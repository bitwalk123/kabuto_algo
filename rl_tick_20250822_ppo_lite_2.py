"""
TradingSimulation (PyTorch PPO-lite / Actor-Critic) sample
- Designed to be driven by an external "別プログラム" which streams ticks using `sim.add(ts, price, volume, force_close=...)`
- Uses ~1-second tick inputs: (ts: float, price: float, volume: float)
- Keeps single position (100 shares). No pyramiding (ナンピン禁止).
- Delta volume is computed inside and compressed by np.log1p.
- Small "PPO-lite" (actor-critic) training performed at finalize() using collected transitions for that episode.
- Model saved/loaded from models/policy.pth

Notes:
- Requires PyTorch 2.8, pandas, numpy
- The file exposes `TradingSimulation` class. The user's driver (別プログラム) can instantiate and call add()/finalize().

This is a compact sample intended for experimentation and parameter search; you should adapt reward shaping,
network sizes, and training schedules to your data and compute budget.
"""

from __future__ import annotations

import os
import math
import collections
from typing import Deque, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = "models/policy.pth"
UNIT_SIZE = 100  # 売買単位
MIN_REWARD_THRESHOLD = 200.0  # 報酬は含み益・収益が200円以上から


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


# -----------------------------
# Small actor-critic networks
# -----------------------------
class Actor(nn.Module):
    def __init__(self, obs_size: int, hidden: int = 64, action_size: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)  # logits


class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# TradingSimulation
# -----------------------------
class TradingSimulation:
    """A lightweight trading simulator with a PPO-lite style trainer.

    Action mapping (int -> string returned by add()):
      0 -> HOLD
      1 -> OPEN_LONG (buy 100)
      2 -> OPEN_SHORT (sell short 100)
      3 -> CLOSE (close any existing position at market)

    Behavioural rules:
      - Only one position at a time. When a position exists, OPEN actions are ignored.
      - CLOSE will close an existing position. If no position exists, CLOSE is treated as HOLD.
      - force_close=True forces a close if a position exists.

    The `add` method returns a string of the action taken immediately.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Running state for feature processing
        self.last_price: Optional[float] = None
        self.last_volume: Optional[float] = None
        self.price_window: Deque[float] = collections.deque(maxlen=120)  # short history for normalization

        # For single position bookkeeping
        self.position: int = 0  # +1 for long 100, -1 for short 100, 0 for flat
        self.entry_price: Optional[float] = None

        # DataFrame results storage
        self.results = []  # list of dicts to construct DataFrame

        # Experience buffer used for training at finalize
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[int] = []
        self.rew_buf: List[float] = []
        self.done_buf: List[bool] = []

        # Networks
        # We'll use 3 features by default: price_return, log1p_delta_volume, price_grad
        self.obs_size = 3
        self.action_size = 4
        self.actor = Actor(self.obs_size, hidden=64, action_size=self.action_size).to(self.device)
        self.critic = Critic(self.obs_size, hidden=64).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4)

        # Model load logic
        self._load_or_init_model()

        # Counters
        self.t = 0

    # -----------------------------
    # Model I/O
    # -----------------------------
    def _load_or_init_model(self):
        ensure_dir(MODEL_PATH)
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=self.device)
                # Simple validity check
                if (
                        "actor_state" in state and
                        "critic_state" in state
                ):
                    self.actor.load_state_dict(state["actor_state"])
                    self.critic.load_state_dict(state["critic_state"])
                    print(f"Loaded existing model from {MODEL_PATH}")
                    return
                else:
                    print(
                        "Model file exists but does not contain expected keys. Initializing new model and overwriting.")
            except Exception as e:
                print("Failed to load model (will create new). Reason:", e)
        else:
            print("No existing model found. Creating new model.")

        # If we reach here, either no model or invalid -> save current (random) model to create file
        self._save_model()

    def _save_model(self):
        ensure_dir(MODEL_PATH)
        torch.save({
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
        }, MODEL_PATH)
        # print(f"Saved model to {MODEL_PATH}")

    # -----------------------------
    # Feature processing
    # -----------------------------
    def _compute_features(self, price: float, volume: float) -> np.ndarray:
        # price_return: normalized price change relative to last price
        if self.last_price is None:
            price_return = 0.0
            price_grad = 0.0
        else:
            price_return = safe_div(price - self.last_price, self.last_price)
            price_grad = price - self.last_price

        # delta_volume -> compress with log1p
        if self.last_volume is None:
            delta_v = 0.0
        else:
            delta_v = max(0.0, volume - self.last_volume)
        log_dv = math.log1p(delta_v)

        # Normalize using running window stats for price_grad and log_dv
        self.price_window.append(price)
        prices = np.array(self.price_window)
        p_mean = float(prices.mean())
        p_std = float(prices.std()) if len(prices) > 1 else 1.0

        price_return_norm = price_return  # already relative
        price_grad_norm = safe_div(price_grad, p_mean) if p_mean != 0 else 0.0

        # NumPy 2.0: use np.ptp instead of prices.ptp()
        ptp_val = np.ptp(prices) if len(prices) > 0 else 1.0
        log_dv_norm = safe_div(log_dv, max(1.0, math.log1p(max(1.0, ptp_val if ptp_val != 0 else 1.0))))

        feat = np.array([price_return_norm, log_dv_norm, price_grad_norm], dtype=np.float32)
        return feat

    # -----------------------------
    # Policy utilities
    # -----------------------------
    def _select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            act = int(probs.argmax(dim=-1).item())
        else:
            m = torch.distributions.Categorical(probs)
            act = int(m.sample().item())
        logp = torch.log(probs[0, act] + 1e-8).item()
        return act, logp

    # -----------------------------
    # Reward function
    # -----------------------------
    def _compute_reward(self, price: float) -> float:
        """Reward uses unrealized P/L while position is open. Only counts beyond MIN_REWARD_THRESHOLD.
        Positive reward only when profit exceeds threshold; small negative step penalty otherwise to discourage endless holding.
        When closing (realizing), a realized profit will be appended as reward in calling code.
        """
        if self.position == 0 or self.entry_price is None:
            return -0.001  # tiny time penalty to encourage actions

        if self.position == 1:
            unreal = (price - self.entry_price) * UNIT_SIZE
        else:
            unreal = (self.entry_price - price) * UNIT_SIZE

        if unreal >= MIN_REWARD_THRESHOLD:
            # reward scaled to thousands to keep magnitudes reasonable
            return float((unreal - MIN_REWARD_THRESHOLD) / 1000.0)
        else:
            # small penalty for not reaching threshold
            return -0.001

    # -----------------------------
    # Public interface
    # -----------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """Called by external driver for each tick. Returns action string taken at this tick."""
        obs = self._compute_features(price, volume)

        # Decide action using current policy
        act, _ = self._select_action(obs, deterministic=False)

        action_str = "HOLD"
        realized_profit = 0.0

        # Enforce trading rules
        if force_close and self.position != 0:
            # force close at this price
            realized_profit = self._close_position(price)
            action_str = "FORCE_CLOSE"
        else:
            if self.position == 0:
                if act == 1:
                    # open long
                    self._open_position(long=True, price=price)
                    action_str = "OPEN_LONG"
                elif act == 2:
                    # open short
                    self._open_position(long=False, price=price)
                    action_str = "OPEN_SHORT"
                else:
                    action_str = "HOLD"
            else:
                # have position
                if act == 3:
                    realized_profit = self._close_position(price)
                    action_str = "CLOSE"
                else:
                    # ignore other open orders while position exists
                    action_str = "HOLD"

        # reward for this timestep (unrealized while open)
        r = self._compute_reward(price)

        # store experience
        self.obs_buf.append(obs.astype(np.float32))
        self.act_buf.append(act)
        self.rew_buf.append(r)
        self.done_buf.append(False)  # episode termination handled in finalize

        # store result row (Profit is realized profit at this tick, else 0)
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": action_str,
            "Profit": float(realized_profit),
        })

        # update history
        self.last_price = price
        self.last_volume = volume
        self.t += 1

        return action_str

    def _open_position(self, long: bool, price: float):
        self.position = 1 if long else -1
        self.entry_price = price
        # No immediate realized profit

    def _close_position(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position == 1:
            profit = (price - self.entry_price) * UNIT_SIZE
        else:
            profit = (self.entry_price - price) * UNIT_SIZE
        # reset position
        self.position = 0
        self.entry_price = None
        return float(profit)

    # -----------------------------
    # Training at episode end (called by driver)
    # -----------------------------
    def finalize(self) -> pd.DataFrame:
        """End of episode: mark done for last transition, perform training (PPO-lite / A2C style), save model, return results DataFrame."""
        # mark last done
        if len(self.done_buf) > 0:
            self.done_buf[-1] = True

        # Compute returns (simple discounted sum) and advantages
        returns = []
        G = 0.0
        gamma = 0.99
        for r, done in zip(reversed(self.rew_buf), reversed(self.done_buf)):
            if done:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)

        if len(returns) == 0:
            df = pd.DataFrame(self.results)
            # reset internal buffers
            self._reset_episode_buffers()
            return df

        obs_arr = torch.tensor(np.stack(self.obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.act_buf, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # A2C style update: policy gradient with value baseline
        # We'll do a few small epochs to keep it "lightweight"
        for _ in range(8):
            logits = self.actor(obs_arr)
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, acts.unsqueeze(1)).squeeze(1)

            values = self.critic(obs_arr)
            advantages = returns_t - values.detach()

            # Policy loss (maximize) -> minimize - advantage * logp
            policy_loss = -(advantages * logp).mean()
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns_t)
            # Entropy bonus to encourage exploration
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()

        # Save model
        self._save_model()

        # Build DataFrame to return
        df = pd.DataFrame(self.results)

        # Reset episode storage for next epoch
        self._reset_episode_buffers()

        return df

    def _reset_episode_buffers(self):
        self.results = []
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        # Note: we intentionally keep last_price/volume/history across days so feature norms don't reset abruptly.


# -----------------------------
# Minimal test-run when executed as __main__
# -----------------------------
if __name__ == "__main__":
    # quick smoke test that can be used to validate library functionality
    print("Smoke test: run a tiny random tick stream")
    sim = TradingSimulation()

    # create synthetic ticks
    ts = 0.0
    price = 3000.0
    volume = 1000.0
    for i in range(200):
        ts += 1.0
        # random walk price
        price += np.random.randn() * 2.0
        volume += max(0.0, np.random.exponential(scale=50.0))
        force_close = (i == 199)
        action = sim.add(ts, price, volume, force_close=force_close)
        if i % 50 == 0:
            print(i, action)

    df = sim.finalize()
    print(df.tail())
    print("Total realized profit:", df["Profit"].sum())
    print("Done.")
