"""
trading_simulation.py

PPO-lite based intraday tick simulator for 1-second tick data.
Designed to be driven by an external program that calls:
    sim = TradingSimulation()
    action = sim.add(ts, price, volume, force_close=False)
    df_result = sim.finalize()

Requirements:
- Python 3.9+
- PyTorch 2.8
- pandas, numpy

Notes:
- Single fixed symbol, trade unit 100 shares, no pyramiding.
- add() accepts raw tick: (ts: float, price: float, volume: float).
- Model is saved at models/policy.pth
- Lightweight PPO-style updates on short minibatches.

"""

import os
import math
import time
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Config
# -------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "policy.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRADE_UNIT = 100  # shares
UPDATE_INTERVAL = 256  # number of steps before an update
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
EPOCHS_PPO = 4
BATCH_SIZE = 64

# Reward thresholds (JPY)
REWARD_BONUS_THRESHOLD = 500.0
CLOSE_TARGET = 1000.0


# -------------------------------
# Helper / Networks
# -------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.feature = MLP(obs_dim, 128, 128)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        f = self.feature(x)
        logits = self.actor(f)
        value = self.critic(f).squeeze(-1)
        return logits, value


# -------------------------------
# Trading Simulation
# -------------------------------

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    logp: float
    reward: float
    value: float
    done: bool


class TradingSimulation:
    def __init__(self, model_path: str = MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Small observation: [price, price_diff, log1p(delta_volume), position_flag, entry_price_relative]
        self.obs_dim = 5
        # actions: 0 hold, 1 buy(open long), 2 sell(open short), 3 close
        self.action_dim = 4

        self.model_path = model_path
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        loaded = self._try_load_model()
        if loaded:
            print(f"Loaded existing model from {self.model_path}")
        else:
            print("No valid existing model found — created a new model.")

        # State
        self.reset_state()

        # Replay buffer / trajectory buffer for PPO-lite
        self.buffer: List[Transition] = []
        self.step_counter = 0
        self.model_version = int(time.time())

    def _try_load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            data = torch.load(self.model_path, map_location=DEVICE)
            # Attempt to load state dict
            if 'state_dict' in data:
                self.policy.load_state_dict(data['state_dict'])
                print("Model state dict loaded.")
                return True
            elif isinstance(data, dict):
                # backward compatibility
                self.policy.load_state_dict(data)
                return True
            else:
                return False
        except Exception as e:
            print("Failed to load model (will overwrite). Error:", e)
            return False

    def save_model(self):
        torch.save({'state_dict': self.policy.state_dict(), 'ts': time.time()}, self.model_path)
        print(f"Model saved to {self.model_path}")

    def reset_state(self):
        self.current_ts: Optional[float] = None
        self.current_price: Optional[float] = None
        self.last_volume: Optional[float] = None

        # Position state
        self.position: int = 0  # 0 flat, 1 long, -1 short
        self.entry_price: Optional[float] = None
        self.open_trade_profit: float = 0.0

        # results
        self.results = []  # list of dicts: Time, Price, 売買アクション, Profit

        # For observation history
        self.last_price: Optional[float] = None

    # Observation builder
    def _build_obs(self, price: float, volume: float) -> np.ndarray:
        # price
        p = float(price)
        # price change
        if self.last_price is None:
            dp = 0.0
        else:
            dp = p - float(self.last_price)
        # delta volume
        if self.last_volume is None:
            dvol = 0.0
        else:
            dvol = float(volume) - float(self.last_volume)
            if dvol < 0:
                # reset during trading day boundaries
                dvol = 0.0
        # compress delta volume
        ldvol = float(np.log1p(dvol))

        pos_flag = 0.0
        entry_rel = 0.0
        if self.position != 0 and self.entry_price is not None:
            pos_flag = 1.0 if self.position == 1 else -1.0
            entry_rel = (p - self.entry_price) / max(self.entry_price, 1.0)
        obs = np.array([p, dp, ldvol, pos_flag, entry_rel], dtype=np.float32)

        # keep last values
        self.last_price = p
        self.last_volume = float(volume)
        return obs

    # Select action (stochastic) using policy
    def _select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        x = torch.from_numpy(obs).to(DEVICE).float().unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(x)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action).to(DEVICE)).item()
            value = value.item()
        return action, logp, value

    # Compute reward for step
    def _compute_reward(self, price: float, action: int, force_close: bool) -> float:
        reward = 0.0
        # small time penalty to encourage decisive actions
        reward -= 0.001

        # unrealized P&L if position open
        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:
                unreal = (price - self.entry_price) * TRADE_UNIT
            else:
                unreal = (self.entry_price - price) * TRADE_UNIT
        else:
            unreal = 0.0

        # scale unrealized to more manageable magnitude
        reward += unreal / 10000.0

        # bonus/penalty thresholds
        if unreal >= REWARD_BONUS_THRESHOLD:
            reward += 0.5
        elif unreal < REWARD_BONUS_THRESHOLD:
            reward -= 0.2

        # If force_close or action==close, give realized reward logic handled in add()
        return float(reward)

    # Simple PPO-lite update
    def _update_model(self):
        if len(self.buffer) < 8:
            return
        # Prepare arrays
        obs = np.array([t.obs for t in self.buffer], dtype=np.float32)
        actions = np.array([t.action for t in self.buffer], dtype=np.int64)
        old_logps = np.array([t.logp for t in self.buffer], dtype=np.float32)
        rewards = np.array([t.reward for t in self.buffer], dtype=np.float32)
        values = np.array([t.value for t in self.buffer], dtype=np.float32)

        # compute returns (simple discounted)
        returns = np.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + GAMMA * G
            returns[i] = G
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.from_numpy(obs).to(DEVICE)
        actions_t = torch.from_numpy(actions).to(DEVICE)
        old_logps_t = torch.from_numpy(old_logps).to(DEVICE)
        returns_t = torch.from_numpy(returns).to(DEVICE)
        adv_t = torch.from_numpy(advantages).to(DEVICE)

        dataset = torch.utils.data.TensorDataset(obs_t, actions_t, old_logps_t, returns_t, adv_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)

        for _ in range(EPOCHS_PPO):
            for b_obs, b_actions, b_oldlogp, b_returns, b_adv in loader:
                logits, values = self.policy(b_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                newlogp = dist.log_prob(b_actions)

                ratio = (newlogp - b_oldlogp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = (b_returns - values).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # clear buffer
        self.buffer = []
        # save model
        self.save_model()

    # External API
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """Add one tick. Returns action string.
        Actions: 'hold', 'buy', 'sell', 'close'
        """
        obs = self._build_obs(price, volume)

        # choose action
        action, logp, value = self._select_action(obs)

        action_str = "hold"
        realized_profit = 0.0

        # enforce no pyramiding: only open new position when flat
        if action == 1 and self.position == 0:
            # open long at current price
            self.position = 1
            self.entry_price = price
            action_str = "新規買い"
        elif action == 2 and self.position == 0:
            # open short
            self.position = -1
            self.entry_price = price
            action_str = "新規売り"
        elif action == 3 and self.position != 0:
            # close position at current price
            if self.position == 1:
                realized_profit = (price - self.entry_price) * TRADE_UNIT
            else:
                realized_profit = (self.entry_price - price) * TRADE_UNIT
            action_str = "返済"
            # record profit in results row below
            # clear position
            self.position = 0
            self.entry_price = None
        else:
            action_str = "ホールド"

        # If force_close is set by caller, override: must close any open position
        if force_close and self.position != 0:
            # close at this price
            if self.position == 1:
                fc_profit = (price - self.entry_price) * TRADE_UNIT
            else:
                fc_profit = (self.entry_price - price) * TRADE_UNIT
            realized_profit = fc_profit
            action_str = "強制返済"
            self.position = 0
            self.entry_price = None

        # compute step reward (unrealized) - note realized reward for closure will be larger in buffer
        step_reward = self._compute_reward(price, action, force_close)

        # If we just realized profit, give a bonus/penalty according to thresholds
        if realized_profit != 0.0:
            if realized_profit >= REWARD_BONUS_THRESHOLD:
                step_reward += 1.0 + (realized_profit / 10000.0)
            else:
                step_reward -= 1.0

        # Save one row in results with Profit only when realized (otherwise 0)
        self.results.append({"Time": ts, "Price": price, "売買アクション": action_str, "Profit": realized_profit})

        # store transition for training
        t = Transition(obs=obs, action=action, logp=logp, reward=step_reward, value=value, done=False)
        self.buffer.append(t)
        self.step_counter += 1

        # Update policy periodically or when force_close on last tick
        if self.step_counter >= UPDATE_INTERVAL or force_close:
            self._update_model()
            self.step_counter = 0

        return action_str

    def finalize(self) -> pd.DataFrame:
        # If position still open (should be closed by force_close externally), close at last price
        if self.position != 0 and self.last_price is not None:
            price = float(self.last_price)
            if self.position == 1:
                r = (price - self.entry_price) * TRADE_UNIT
            else:
                r = (self.entry_price - price) * TRADE_UNIT
            self.results.append(
                {"Time": self.current_ts or time.time(), "Price": price, "売買アクション": "最終強制返済", "Profit": r})
            self.position = 0
            self.entry_price = None

        df = pd.DataFrame(self.results, columns=["Time", "Price", "売買アクション", "Profit"]) if len(
            self.results) > 0 else pd.DataFrame(columns=["Time", "Price", "売買アクション", "Profit"])

        # reset results (external driver expects simulator reset)
        self.results = []
        self.buffer = []
        self.step_counter = 0
        # Note: keep model in memory for continued learning

        return df


# -------------------------------
# If run as script, a simple dry-run / unit test with synthetic data
# -------------------------------
if __name__ == "__main__":
    # quick smoke test
    sim = TradingSimulation()
    rng = np.random.RandomState(0)
    vol = 1e6
    price = 5000.0
    ts = 0.0
    for i in range(600):
        ts += 1.0
        # small random walk
        price += rng.normal(scale=1.0)
        vol += abs(int(rng.normal(scale=1000)))
        action = sim.add(ts, float(price), float(vol), force_close=(i == 599))
        if i % 100 == 0:
            print(i, "action:", action)
    df = sim.finalize()
    print(df.head())
    print("Total profit:", df['Profit'].sum())
