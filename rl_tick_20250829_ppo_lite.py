# ppo_lite_trading.py
# 要: Python 3.13, torch==2.8.0, numpy==2.2.6, pandas==2.3.2
# 実行例:
#   python ppo_lite_trading.py train  # 学習用の簡易実行（サンプルCSVを読み込む想定）
#   （あるいは学習部分は GUI から呼び出す想定）

import math
import random
from collections import deque
from typing import Deque, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Config / Hyperparameters
# ----------------------------
N_FEATURE = 60  # window size for MA, std, RSI etc.
EPS = 1e-8
SHARES = 100
SLIPPAGE = 1.0  # yen (呼び値1倍)
ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

DEVICE = torch.device("cpu")  # inference/training on CPU as required

# PPO-lite hyperparams (tunable)
PPO_EPOCHS_PER_BATCH = 4
PPO_CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95  # for GAE
LR_POLICY = 3e-4
LR_VALUE = 1e-3
ENTROPY_COEF = 1e-3
VALUE_LOSS_COEF = 0.5
MINIBATCH_SIZE = 256  # if batch smaller, will use full-batch


# ----------------------------
# Utilities: feature calculations
# ----------------------------
def compute_rsi(prices: List[float], period: int) -> float:
    # returns RSI over last `period` prices; if not enough data -> np.nan
    if len(prices) < period + 1:
        return float("nan")
    deltas = np.diff(np.array(prices[-(period + 1):]))
    ups = deltas.clip(min=0)
    downs = -deltas.clip(max=0)
    # Wilder's moving average (simple approx)
    avg_up = ups.mean()
    avg_down = downs.mean()
    if avg_down == 0 and avg_up == 0:
        return 50.0
    if avg_down == 0:
        return 100.0
    rs = avg_up / (avg_down + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


# Running mean & var via deque (windowed)
class RunningStats:
    def __init__(self, window: int):
        self.window = window
        self.deq: Deque[float] = deque(maxlen=window)

    def append(self, x: float):
        self.deq.append(x)

    def mean(self) -> float:
        if len(self.deq) == 0:
            return 0.0
        return float(np.mean(self.deq))

    def var(self) -> float:
        if len(self.deq) <= 1:
            return 0.0
        return float(np.var(self.deq, ddof=0))

    def std(self) -> float:
        return math.sqrt(self.var() + EPS)


# ----------------------------
# Policy & Value Networks (small, CPU friendly)
# ----------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------
# Trading Simulation (推論用 / 記録用)
# ----------------------------
class TradingSimulation:
    """
    Usage:
      sim = TradingSimulation(policy=None)  # for pure simulation w/o policy
      sim = TradingSimulation(policy=loaded_policy)  # for live inference
      action_str = sim.add(ts, price, volume, force_close=False)
      df = sim.finalize()
    """

    def __init__(self, policy: Optional[PolicyNet] = None, value=None, device=DEVICE):
        self.policy = policy
        self.device = device
        # Buffers for features
        self.price_hist: Deque[float] = deque(maxlen=N_FEATURE)
        self.volume_hist: Deque[float] = deque(maxlen=N_FEATURE + 1)
        self.last_volume = 0.0

        self.running_price_stats = RunningStats(window=N_FEATURE)
        self.running_var = 1.0

        # Position management
        self.position = 0  # 0 none, +1 long, -1 short
        self.entry_price = 0.0

        # Records
        self.records = []  # list of (Time, Price, Action, Profit)
        self.unrealized = 0.0

        # Lightweight normalization epsilon
        self.eps = 1e-8

    def _valid_action(self, a: int) -> int:
        # enforce rules: single position only, no pyramiding
        if a == 1 and self.position == 1:  # BUY while long -> HOLD
            return 0
        if a == 2 and self.position == -1:  # SELL while short -> HOLD
            return 0
        if a == 3 and self.position == 0:  # REPAY when no pos -> HOLD
            return 0
        return a

    def _make_features(self, price: float, volume: float) -> Optional[np.ndarray]:
        # update history
        self.price_hist.append(price)
        self.volume_hist.append(volume)

        if len(self.price_hist) < N_FEATURE or len(self.volume_hist) < 2:
            return None  # warmup not ready

        # Δvolume = max(volume - last_volume, 0)
        last_vol = list(self.volume_hist)[-2]
        delta_vol = max(volume - last_vol, 0.0)
        fv_log_dv = math.log1p(delta_vol)

        arr_prices = np.array(self.price_hist)
        ma = arr_prices.mean()
        std = arr_prices.std(ddof=0)
        rsi = compute_rsi(list(self.price_hist), N_FEATURE)
        # running mean/var for z-score (we maintain rolling stats too)
        self.running_price_stats.append(price)
        mean_run = self.running_price_stats.mean()
        var_run = self.running_price_stats.var()
        price_z = (price - mean_run) / math.sqrt(var_run + self.eps)

        features = np.array([fv_log_dv, ma, std, rsi, price_z], dtype=np.float32)
        # optional: add recent returns
        returns_1 = (self.price_hist[-1] - self.price_hist[-2]) / (self.price_hist[-2] + self.eps)
        features = np.concatenate([features, np.array([returns_1], dtype=np.float32)])
        return features  # shape (6,)

    def _select_action(self, features: np.ndarray) -> int:
        # If no policy provided, simple heuristic: HOLD
        if self.policy is None:
            return 0
        x = torch.from_numpy(features).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        action = int(np.argmax(probs))  # deterministic for inference (fast)
        return action

    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        Add one tick; return action string in Japanese mapping.
        - Performs lightweight feature calc and policy inference.
        - If not enough data, returns "HOLD".
        - If force_close True and there's a position, performs immediate REPAY.
        """
        features = self._make_features(price, volume)
        if features is None:
            # Warm-up: not enough data
            self.records.append((ts, price, "HOLD", 0.0))
            self.last_volume = volume
            return "HOLD"

        # If force_close: repay immediately
        if force_close and self.position != 0:
            # treat as REPAY action
            action = 3
        else:
            raw_action = self._select_action(features)
            action = self._valid_action(raw_action)

        profit = 0.0
        # Execute according to rules and apply slippage
        if action == 1:  # BUY (new long)
            if self.position == 0:
                entry = price + SLIPPAGE
                self.position = 1
                self.entry_price = entry
            # else already long -> (shouldn't happen due to _valid_action)
        elif action == 2:  # SELL (new short)
            if self.position == 0:
                entry = price - SLIPPAGE
                self.position = -1
                self.entry_price = entry
        elif action == 3:  # REPAY (close pos)
            if self.position == 1:
                exit_price = price - SLIPPAGE
                profit = (exit_price - self.entry_price) * SHARES
                self.position = 0
                self.entry_price = 0.0
            elif self.position == -1:
                exit_price = price + SLIPPAGE
                profit = (self.entry_price - exit_price) * SHARES
                self.position = 0
                self.entry_price = 0.0
        # For HOLD or new entries no immediate realized profit.
        # Add a small fraction of unrealized P/L as reward bookkeeping:
        self.unrealized = 0.0
        if self.position == 1:
            # mark-to-market long
            self.unrealized = (price - self.entry_price) * SHARES
        elif self.position == -1:
            self.unrealized = (self.entry_price - price) * SHARES

        # We store Profit as realized profit for that tick (unrealized not summed here).
        self.records.append((ts, price, ACTION_MAP[int(action)], profit))
        self.last_volume = volume
        return ACTION_MAP[int(action)]

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records, columns=["Time", "Price", "Action", "Profit"])
        # reset
        self.records = []
        self.position = 0
        self.entry_price = 0.0
        self.unrealized = 0.0
        self.price_hist.clear()
        self.volume_hist.clear()
        self.running_price_stats = RunningStats(window=N_FEATURE)
        return df


# ----------------------------
# PPO-lite Trainer
# ----------------------------
class PPOTrainer:
    def __init__(self, feature_dim: int):
        self.policy = PolicyNet(input_dim=feature_dim).to(DEVICE)
        self.value = ValueNet(input_dim=feature_dim).to(DEVICE)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=LR_POLICY)
        self.opt_value = optim.Adam(self.value.parameters(), lr=LR_VALUE)

    def act_and_record(self, sim: TradingSimulation, ts_list, price_list, volume_list, force_last_repay=True):
        """
        Run through provided ticks using sim but with stochastic policy to collect data.
        Returns trajectory lists for training.
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        # We'll use sim but ensure we don't change its random-deterministic inference logic.
        # Instead, implement sampling from policy here using policy network directly.
        # Create local buffers similar to sim's feature rolling to compute features identically.
        ph = deque(maxlen=N_FEATURE)
        vh = deque(maxlen=N_FEATURE + 1)
        running_stats = RunningStats(window=N_FEATURE)
        position = 0
        entry_price = 0.0

        for i, (ts, price, volume) in enumerate(zip(ts_list, price_list, volume_list)):
            ph.append(price)
            vh.append(volume)
            if len(ph) < N_FEATURE or len(vh) < 2:
                # Warmup
                states.append(None)
                actions.append(0)
                log_probs.append(0.0)
                rewards.append(0.0)
                values.append(0.0)
                dones.append(False)
                running_stats.append(price)
                continue

            # compute features same as sim
            last_vol = list(vh)[-2]
            delta_vol = max(volume - last_vol, 0.0)
            fv_log_dv = math.log1p(delta_vol)
            arr_prices = np.array(ph)
            ma = arr_prices.mean()
            std = arr_prices.std(ddof=0)
            rsi = compute_rsi(list(ph), N_FEATURE)
            running_stats.append(price)
            mean_run = running_stats.mean()
            var_run = running_stats.var()
            price_z = (price - mean_run) / math.sqrt(var_run + EPS)
            returns_1 = (ph[-1] - ph[-2]) / (ph[-2] + EPS)
            feat = np.array([fv_log_dv, ma, std, rsi, price_z, returns_1], dtype=np.float32)
            states.append(feat)

            x = torch.from_numpy(feat).float().to(DEVICE).unsqueeze(0)
            logits = self.policy(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            m = torch.distributions.Categorical(probs)
            a = int(m.sample().item())
            lp = float(m.log_prob(torch.tensor(a)).item())
            v = float(self.value(x).item())

            # ensure action validity relative to current position
            if a == 1 and position == 1:
                a = 0
            if a == 2 and position == -1:
                a = 0
            if a == 3 and position == 0:
                a = 0

            # execute action to compute reward (realized profit)
            reward = 0.0
            if a == 1 and position == 0:
                entry_price = price + SLIPPAGE
                position = 1
            elif a == 2 and position == 0:
                entry_price = price - SLIPPAGE
                position = -1
            elif a == 3 and position != 0:
                if position == 1:
                    exit_price = price - SLIPPAGE
                    reward = (exit_price - entry_price) * SHARES
                else:
                    exit_price = price + SLIPPAGE
                    reward = (entry_price - exit_price) * SHARES
                position = 0
                entry_price = 0.0
            # small per-tick unrealized reward: fraction of unrealized P/L
            unreal = 0.0
            if position == 1:
                unreal = (price - entry_price) * SHARES * 0.01
            elif position == -1:
                unreal = (entry_price - price) * SHARES * 0.01
            total_reward = reward + unreal

            actions.append(a)
            log_probs.append(lp)
            rewards.append(total_reward)
            values.append(v)
            dones.append(False)

            # if last tick and force repay requested -> repay now
            if force_last_repay and i == len(price_list) - 1 and position != 0:
                if position == 1:
                    exit_price = price - SLIPPAGE
                    final_reward = (exit_price - entry_price) * SHARES
                else:
                    exit_price = price + SLIPPAGE
                    final_reward = (entry_price - exit_price) * SHARES
                rewards[-1] += final_reward  # add to last step
                position = 0
                entry_price = 0.0

        return states, actions, log_probs, rewards, values, dones

    def compute_gae_returns(self, rewards: List[float], values: List[float], dones: List[bool]):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            last_gae = delta + GAMMA * LAMBDA * mask * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + np.array(values, dtype=np.float32)
        return returns, advantages

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        # Prepare tensors
        idxs = np.arange(len(states))
        # filter out warmup None states
        valid = [i for i in idxs if states[i] is not None]
        if len(valid) == 0:
            return
        S = torch.tensor(np.stack([states[i] for i in valid]), dtype=torch.float32).to(DEVICE)
        A = torch.tensor([actions[i] for i in valid], dtype=torch.long).to(DEVICE)
        old_lp = torch.tensor([old_log_probs[i] for i in valid], dtype=torch.float32).to(DEVICE)
        R = torch.tensor([returns[i] for i in valid], dtype=torch.float32).to(DEVICE)
        ADV = torch.tensor([advantages[i] for i in valid], dtype=torch.float32).to(DEVICE)
        ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)

        batch_size = S.shape[0]
        mb = min(MINIBATCH_SIZE, batch_size)
        for _ in range(PPO_EPOCHS_PER_BATCH):
            # shuffle minibatches
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, mb):
                end = min(start + mb, batch_size)
                idx = perm[start:end]
                s_b = S[idx]
                a_b = A[idx]
                old_lp_b = old_lp[idx]
                r_b = R[idx]
                adv_b = ADV[idx]

                logits = self.policy(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a_b)
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                # value loss
                values_pred = self.value(s_b)
                value_loss = (r_b - values_pred).pow(2).mean()

                # update policy
                self.opt_policy.zero_grad()
                (policy_loss - ENTROPY_COEF * entropy).backward()
                self.opt_policy.step()

                # update value
                self.opt_value.zero_grad()
                (VALUE_LOSS_COEF * value_loss).backward()
                self.opt_value.step()

    def train_from_dataframe(self, df: pd.DataFrame, epochs: int = 100, save_path_prefix: Optional[str] = None):
        """
        df: DataFrame with columns ["Time", "Price", "Volume"] (single day)
        This function will run multiple epochs over the same df, collecting trajectories and updating policy.
        """
        ts_list = df["Time"].tolist()
        price_list = df["Price"].tolist()
        volume_list = df["Volume"].tolist()

        feature_dim = 6
        for ep in range(epochs):
            states, actions, logps, rewards, values, dones = self.act_and_record(
                sim=None, ts_list=ts_list, price_list=price_list, volume_list=volume_list, force_last_repay=True
            )
            returns, advantages = self.compute_gae_returns(rewards, values, dones)
            self.ppo_update(states, actions, logps, returns, advantages)

            # simple evaluation: run a deterministic pass to compute total realized profit
            sim_eval = TradingSimulation(policy=self.policy)
            for i, (ts, p, v) in enumerate(zip(ts_list, price_list, volume_list)):
                force_close = (i == len(price_list) - 1)
                sim_eval.add(ts, p, v, force_close=force_close)
            df_res = sim_eval.finalize()
            profit = df_res["Profit"].sum()
            print(f"Epoch {ep + 1}/{epochs}  -> Eval Profit: {profit:.0f}")

            if save_path_prefix:
                torch.save(self.policy.state_dict(), f"{save_path_prefix}_policy_epoch{ep + 1}.pth")
                torch.save(self.value.state_dict(), f"{save_path_prefix}_value_epoch{ep + 1}.pth")

        return self.policy, self.value


# ----------------------------
# Example main: training harness matching user's script
# ----------------------------
if __name__ == "__main__":
    import sys
    import os

    # Basic CLI: train <excel_file> <epochs>
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        excel_file = sys.argv[2] if len(sys.argv) >= 3 else "data/tick_20250819_7011.xlsx"
        epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
        print("Loading", excel_file)
        df = pd.read_excel(excel_file)  # columns: Time, Price, Volume
        print(df.head())

        trainer = PPOTrainer(feature_dim=6)
        trainer.train_from_dataframe(df, epochs=epochs, save_path_prefix="models/ppo_lite")

        print("Done training.")
    else:
        # Simple demo: run simulation inference using a random small CSV if provided
        demo_file = "data/tick_demo.csv"
        if not os.path.exists(demo_file):
            # create small synthetic demo
            times = np.arange(0, 300)
            price = 1000 + np.cumsum(np.random.randn(300) * 0.5)
            vol = np.abs(1000 + np.cumsum(np.random.randn(300) * 50))
            df_demo = pd.DataFrame({"Time": times, "Price": price, "Volume": vol})
            os.makedirs("data", exist_ok=True)
            df_demo.to_csv(demo_file, index=False)
        df_demo = pd.read_csv(demo_file)
        sim = TradingSimulation(policy=None)  # no policy => always HOLD
        for i, row in df_demo.iterrows():
            force_close = (i == len(df_demo) - 1)
            a = sim.add(row["Time"], float(row["Price"]), float(row["Volume"]), force_close=force_close)
        res = sim.finalize()
        print(res.head())
        print("Total profit:", res["Profit"].sum())
