"""
trading_simulation.py

Lightweight PPO-like (actor-critic) trading simulator for intraday tick data (1s resolution).
Interface:
    sim = TradingSimulation()
    action_label = sim.add(ts, price, volume, force_close=False)
    df_result = sim.finalize()  # returns DataFrame with Time, Price, Action, Profit
Behavior:
    - Single instrument, tick-level
    - trade unit = 100 shares
    - margin trading allowed (long/short)
    - no partial fills, no pyramiding: cannot create new position until current closed
    - delta volume preprocessed with log1p scaling internally
    - simple actor-critic (PPO-lite) updates
    - saves/loads model file models/policy.pth
"""

import os
import math
import time
import random
from collections import deque, namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Configurable hyperparameters
# ----------------------------
MODEL_PATH = "models/policy.pth"
TRADE_UNIT = 100            # shares
TICK_PRICE_SCALE = 10000.0  # scale price to ~[0,1]
GAMMA = 0.99
LR = 3e-4
BATCH_UPDATE_STEPS = 128    # accumulate transitions before update
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cpu")  # change to "cuda" if GPU available

Transition = namedtuple("Transition", ["obs", "action", "logp", "reward", "done", "value"])

# ----------------------------
# Neural network (actor-critic)
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

# ----------------------------
# Utility functions
# ----------------------------
def softmax_sample_logits(logits):
    probs = torch.softmax(logits, dim=-1)
    m = torch.distributions.Categorical(probs)
    a = m.sample()
    return a, m.log_prob(a), m.entropy()

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ----------------------------
# TradingSimulation
# ----------------------------
class TradingSimulation:
    """
    Main simulator class. Methods:
      - add(ts, price, volume, force_close=False) -> action_label (str)
      - finalize() -> pandas.DataFrame (Time, Price, Action, Profit)
    """

    def __init__(self):
        # internal state
        self.prev_volume = None
        self.position = 0           # 0=no position, +1=long, -1=short
        self.entry_price = None     # price at which position was opened
        self.open_ts = None

        # results record
        self.results = []  # list of dicts: Time, Price, Action, Profit

        # model & optimizer
        self.obs_dim = 4  # [price_norm, log1p_dvol, pos_flag, unreal_pnl_norm]
        self.n_actions = 3  # 0=hold,1=buy/close-buy (long open / long close),2=sell/close-sell (short open / short close)
        self.model = ActorCritic(self.obs_dim, hidden=128, n_actions=self.n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # experience buffer
        self.buffer = []
        self.step_count = 0

        # load existing model if available
        self._load_or_init_model()

    # ----------------------------
    # Model save/load utilities
    # ----------------------------
    def _load_or_init_model(self):
        ensure_dir(MODEL_PATH)
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state.get("optim", self.optimizer.state_dict()))
                print(f"[Model] Loaded existing model from {MODEL_PATH}. Using loaded model.")
                return
            except Exception as e:
                print(f"[Model] Failed to load model ({e}). Creating new model and will overwrite existing file.")
        else:
            print("[Model] No existing model found. Creating new model.")
        # save initial state
        self._save_model()

    def _save_model(self):
        ensure_dir(MODEL_PATH)
        torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}, MODEL_PATH)

    # ----------------------------
    # Observation construction
    # ----------------------------
    def _make_obs(self, price, delta_vol):
        # price normalization
        p_norm = price / TICK_PRICE_SCALE  # price roughly 1000..10000 -> 0.1 ~ 1.0
        # delta vol scaling: log1p compress
        dv = np.log1p(max(delta_vol, 0.0))
        dv_norm = dv / 10.0  # heuristic scaling to keep values ~0..small
        pos_flag = float(self.position)  # -1,0,1
        # unrealized P/L normalized by trade unit and price scale
        if self.position != 0 and self.entry_price is not None:
            unreal = (price - self.entry_price) * self.position * TRADE_UNIT  # yen
        else:
            unreal = 0.0
        unreal_norm = unreal / 10000.0  # bring to comparable magnitude
        obs = np.array([p_norm, dv_norm, pos_flag, unreal_norm], dtype=np.float32)
        return obs

    # ----------------------------
    # Reward shaping
    # ----------------------------
    def _reward_from_unrealized(self, unreal_yen, realized_yen=0.0):
        """
        Apply non-linear reward scaling per spec:
          - unrealized >= 1000 -> x2
          - >=500 -> x1.5
          - >=200 -> x1.0
          - >0 -> small bonus
          - <0 -> small penalty
        realized profit also added directly.
        Returns scalar reward (float)
        """
        base = 0.0
        # small step reward proportional
        base += unreal_yen / 1000.0  # scale down
        # nonlinear multiplier (applied to base part)
        multiplier = 1.0
        if unreal_yen >= 1000.0:
            multiplier = 2.0
        elif unreal_yen >= 500.0:
            multiplier = 1.5
        elif unreal_yen >= 200.0:
            multiplier = 1.0
        else:
            # tiny bonus/penalty
            if unreal_yen > 0:
                base += 0.05
            elif unreal_yen < 0:
                base -= 0.05
        reward = base * multiplier
        # add realized profit scaled down
        reward += realized_yen / 1000.0
        return float(reward)

    # ----------------------------
    # Action logic and trade execution rules
    # ----------------------------
    def _action_to_label(self, action_int, price, force_close):
        """
        Determine legal action given current position and map to label.
        Returns (label_str, executed_flag, realized_profit_yen)
        """
        realized = 0.0
        executed = False
        label = "HOLD"

        # When flat (position == 0):
        if self.position == 0:
            if action_int == 1:
                # open long at current price
                self.position = 1
                self.entry_price = price  # open at current price
                self.open_ts = None
                executed = True
                label = "OPEN_LONG"
            elif action_int == 2:
                # open short
                self.position = -1
                self.entry_price = price
                self.open_ts = None
                executed = True
                label = "OPEN_SHORT"
            else:
                label = "HOLD"

        else:
            # We have open position: only allow close action (1 closes long, 2 closes short)
            if self.position == 1:
                # long open; to close long we need action==1 OR force_close True
                if action_int == 1 or force_close:
                    # closing long -> realized = (current_price - entry_price) * TRADE_UNIT
                    # Per spec: return sell executed at current_price - tick (call price - 1)
                    exec_price = price - 1.0
                    realized = (exec_price - self.entry_price) * TRADE_UNIT
                    label = "CLOSE_LONG"
                    executed = True
                    self.position = 0
                    self.entry_price = None
                else:
                    label = "HOLD_LONG"
            elif self.position == -1:
                if action_int == 2 or force_close:
                    # closing short -> buy to cover at current_price + tick (price + 1)
                    exec_price = price + 1.0
                    realized = (self.entry_price - exec_price) * TRADE_UNIT
                    label = "CLOSE_SHORT"
                    executed = True
                    self.position = 0
                    self.entry_price = None
                else:
                    label = "HOLD_SHORT"

        return label, executed, realized

    # ----------------------------
    # Core method: add a new tick
    # ----------------------------
    def add(self, ts, price, volume, force_close=False):
        """
        Called by external program for each tick.
        Returns action label (string) representing what the simulator executed at this tick.
        """
        # compute delta volume
        if self.prev_volume is None:
            delta_vol = max(volume, 0.0)
        else:
            delta_vol = max(volume - self.prev_volume, 0.0)
        self.prev_volume = volume

        obs = self._make_obs(price, delta_vol)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # forward
        with torch.no_grad():
            logits, value = self.model(obs_t)
        logits = logits.squeeze(0)
        value = value.item()

        # sample action
        action_t, logp_t, _ = softmax_sample_logits(logits)
        action = int(action_t.item())
        logp = logp_t.item()

        # map action respecting trading rules
        label, executed, realized = self._action_to_label(action, price, force_close)

        # compute unrealized for reward shaping
        if self.position != 0 and self.entry_price is not None:
            unreal_yen = (price - self.entry_price) * self.position * TRADE_UNIT
        else:
            unreal_yen = 0.0

        # reward (if realized -> include realized)
        immediate_reward = self._reward_from_unrealized(unreal_yen, realized)

        # append to results row (Profit only on realized)
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": label,
            "Profit": realized
        })

        # store transition
        self.buffer.append(Transition(obs=obs, action=action, logp=logp, reward=immediate_reward, done=False, value=value))
        self.step_count += 1

        # update model periodically
        if len(self.buffer) >= BATCH_UPDATE_STEPS:
            self._update_model(last_value=0.0)

        # if executed and realized profit, return label
        return label

    # ----------------------------
    # Simple PPO-lite / A2C style update
    # ----------------------------
    def _update_model(self, last_value=0.0):
        """
        Compute returns and advantages from buffer and perform one update (actor+critic).
        This is intentionally lightweight.
        """
        # prepare trajectories
        obs = torch.tensor(np.stack([t.obs for t in self.buffer]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.int64, device=DEVICE)
        old_logps = torch.tensor([t.logp for t in self.buffer], dtype=torch.float32, device=DEVICE)
        rewards = [t.reward for t in self.buffer]
        values = torch.tensor([t.value for t in self.buffer], dtype=torch.float32, device=DEVICE)

        # compute returns (discounted)
        returns = []
        R = last_value
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        advantages = returns - values

        # forward to get current logits and values
        logits, cur_values = self.model(obs)
        logp_all = torch.log_softmax(logits, dim=-1)
        cur_logps = logp_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(torch.softmax(logits, dim=-1) * logp_all).sum(-1).mean()

        # policy loss (use simple surrogate with clipped ratio omitted — PPO-lite)
        ratio = torch.exp(cur_logps - old_logps)
        policy_loss = -(ratio * advantages.detach()).mean()

        # value loss
        value_loss = (returns - cur_values).pow(2).mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        # clear buffer
        self.buffer = []
        # optionally save model
        self._save_model()
        print(f"[Update] Performed model update. Loss={loss.item():.6f}, policy_loss={policy_loss.item():.6f}, value_loss={value_loss.item():.6f}")

    # ----------------------------
    # Finalize simulation: force close any open positions, return DataFrame and reset
    # ----------------------------
    def finalize(self):
        """
        Called at end of tick sequence. If there is open position, it's closed at last known price (apply tick rules).
        Returns DataFrame with columns: Time, Price, Action, Profit (realized profit entries only)
        and resets internal results & state for next epoch.
        """
        # If still have position, force-close at last price recorded in prev result
        if self.position != 0 and len(self.results) > 0:
            last_row = self.results[-1]
            last_price = last_row["Price"]
            # call add with force_close to trigger close; but we must avoid recording duplicate extra row
            label, executed, realized = self._action_to_label(action_int=1 if self.position==1 else 2, price=last_price, force_close=True)
            # record force-close result
            self.results.append({
                "Time": last_row["Time"],
                "Price": last_price,
                "Action": label + "_FORCE",
                "Profit": realized
            })

        # compute totals and build DataFrame
        df = pd.DataFrame(self.results, columns=["Time", "Price", "Action", "Profit"])
        total_profit = df["Profit"].sum() if not df.empty else 0.0
        print(f"[Finalize] Total realized profit: {total_profit:.2f} JPY")

        # if any remaining transitions in buffer, update once more
        if len(self.buffer) > 0:
            self._update_model(last_value=0.0)

        # reset internal state except model
        self.prev_volume = None
        self.position = 0
        self.entry_price = None
        self.open_ts = None
        self.results = []

        return df

# ----------------------------
# If run as script: small self-test
# ----------------------------
if __name__ == "__main__":
    # quick smoke test using synthetic tick data
    sim = TradingSimulation()
    rng = np.random.default_rng(0)
    price = 5000.0
    volume = 1000.0
    records = []
    for i in range(500):
        ts = i
        # simulate small price walk
        price += rng.normal(0, 1.0)
        # simulate cumulative volume
        volume += abs(rng.normal(1000, 300))
        action = sim.add(ts, float(price), float(volume), force_close=False)
        if i % 100 == 0:
            print(f"Tick {i}, price={price:.2f}, action={action}")

    df_result = sim.finalize()
    print(df_result.head())
    print("Done smoke test")
