# trading_simulation.py
import os
import math
import random
from collections import deque, namedtuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
#  設定（必要なら変更）
# -----------------------------
MODEL_PATH = "models/policy.pth"
DEVICE = torch.device("cpu")  # GPUが使えるなら "cuda" に変更
PRICE_SCALE = 10000.0  # price を正規化するための目安値（価格帯 1,000-10,000 を想定）
VOLUME_LOG_BASE = True

# RL / PPO-lite ハイパーパラ
GAMMA = 0.99
LR = 1e-4
UPDATE_EPOCHS = 4
CLIP_EPS = 0.2
BATCH_SIZE = 64
BUFFER_CAPACITY = 2048
MIN_BUFFER_TO_UPDATE = 128

# 取引の基本ルール
TRADE_UNIT = 100  # 株
TICK = 1.0  # 呼び値（1円）
# -----------------------------

Transition = namedtuple('Transition', ['state', 'action', 'logprob', 'reward', 'next_state', 'done'])

def build_state(price, delta_logvol, position_flag, entry_price_norm):
    # state as numpy array
    return np.array([price, delta_logvol, position_flag, entry_price_norm], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden=64, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)  # logits
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.fc(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)


class TradingSimulation:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        self.device = device
        self.model_path = model_path
        self.buffer = []
        self.buffer_capacity = BUFFER_CAPACITY

        # tick history for delta volume
        self.last_volume = None

        # results log
        self.results = []  # list of dicts for DataFrame

        # position: None or dict {side: 'long'|'short', entry_price: float}
        self.position = None

        # internal experience buffer for updates
        self.memory = []

        # build model
        self.input_dim = 4  # [price_norm, log1p(delta_vol), position_flag, entry_price_norm]
        self.action_dim = 3  # when no position: [hold, open_long, open_short]; when position: we'll interpret outputs
        self.net = ActorCritic(self.input_dim, hidden=64, action_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

        # load or init
        self._load_or_create_model()

        # internal step counter
        self.step = 0

    # -----------------------
    # モデルのロード/保存
    # -----------------------
    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                payload = torch.load(self.model_path, map_location=self.device)
                self.net.load_state_dict(payload['model'])
                self.optimizer.load_state_dict(payload['optim'])
                print(f"[MODEL] Loaded existing model from {self.model_path}.")
            except Exception as e:
                print(f"[MODEL] Failed to load existing model (corrupt?): {e}\nCreating a new model and will overwrite existing file.")
                self._save_model()
        else:
            print("[MODEL] No existing model found — creating a new model.")
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            self._save_model()

    def _save_model(self):
        payload = {'model': self.net.state_dict(), 'optim': self.optimizer.state_dict()}
        torch.save(payload, self.model_path)
        #print(f"[MODEL] Model saved to {self.model_path}.")

    # -----------------------
    # 行動決定 / 推論
    # -----------------------
    def _policy_and_value(self, state_np):
        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item(), probs.detach().cpu().numpy().ravel()

    # -----------------------
    # 報酬関数
    # - 非線形増加（含み益に応じた倍率）
    # -----------------------
    def _compute_step_reward(self, unrealized_pnl):
        # unrealized_pnl is in JPY (for TRADE_UNIT units)
        base = 0.0
        # small bonus/penalty for being positive/negative
        if unrealized_pnl > 0:
            base += 0.1
        elif unrealized_pnl < 0:
            base -= 0.1

        # multiplier depending on thresholds
        mult = 1.0
        if unrealized_pnl >= 1000:
            mult = 2.0
        elif unrealized_pnl >= 500:
            mult = 1.5
        elif unrealized_pnl >= 200:
            mult = 1.0
        else:
            mult = 1.0

        reward = base + (unrealized_pnl / 10000.0) * mult  # scale down to keep magnitude reasonable
        return reward

    # -----------------------
    # 学習（PPO-lite）
    # ここではエピソード途中でもバッファが十分なら更新を行う
    # -----------------------
    def _update(self):
        if len(self.buffer) < MIN_BUFFER_TO_UPDATE:
            return

        # convert to tensors
        states = torch.tensor(np.vstack([t.state for t in self.buffer]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([t.logprob for t in self.buffer], dtype=torch.float32, device=self.device)
        rewards = [t.reward for t in self.buffer]
        dones = [t.done for t in self.buffer]

        # compute discounted returns and advantages (simple)
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(UPDATE_EPOCHS):
            # minibatch sampling
            idxs = np.arange(len(self.buffer))
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), BATCH_SIZE):
                mb_idx = idxs[start:start + BATCH_SIZE]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]

                logits, values = self.net(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                mb_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio for PPO
                ratio = torch.exp(mb_logprobs - mb_old_logprobs)
                advantage = mb_returns - values.detach()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                loss = policy_loss + 0.5 * value_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

        # clear buffer after update
        self.buffer = []
        # save model periodically
        self._save_model()

    # -----------------------
    # add: 別プログラムから逐次データを投入
    # 戻り値: 売買アクション（文字列）
    # -----------------------
    def add(self, ts, price, volume, force_close=False):
        """
        Parameters
        ----------
        ts: float (timestamp in seconds)
        price: float (current price in yen)
        volume: float (cumulative volume)
        force_close: bool (最終時刻で True にして強制返済)
        Returns
        -------
        action_str: str (例: "hold", "open_long", "open_short", "close")
        """
        self.step += 1

        # delta volume
        if self.last_volume is None:
            delta_vol = 0.0
        else:
            delta_vol = max(0.0, volume - self.last_volume)
        self.last_volume = volume
        delta_logvol = float(np.log1p(delta_vol))

        # normalized price and entry price
        price_norm = price / PRICE_SCALE
        entry_price_norm = 0.0
        position_flag = 0.0
        if self.position is not None:
            entry_price_norm = self.position['entry_price'] / PRICE_SCALE
            position_flag = 1.0 if self.position['side'] == 'long' else -1.0

        state = build_state(price_norm, delta_logvol, position_flag, entry_price_norm)

        # decide action via policy
        action_idx, logprob, value, probs = self._policy_and_value(state)
        # interpret action depending on whether a position exists
        action_str = "hold"

        # When no position:
        if self.position is None:
            if action_idx == 0:
                action_str = "hold"
            elif action_idx == 1:
                # open long at current price
                self.position = {'side': 'long', 'entry_price': price}
                action_str = "open_long"
            elif action_idx == 2:
                # open short at current price
                self.position = {'side': 'short', 'entry_price': price}
                action_str = "open_short"
        else:
            # position exists -> options: hold or close (we map action_idx 1 to close for simplicity)
            if action_idx == 1:
                # close position
                # apply repayment price adjustments per spec:
                if self.position['side'] == 'long':
                    # 返済売りは現在価格 - 呼び値で成立
                    close_price = price - TICK
                    realized = (close_price - self.position['entry_price']) * TRADE_UNIT
                else:
                    # 返済買いは現在価格 + 呼び値で成立
                    close_price = price + TICK
                    realized = (self.position['entry_price'] - close_price) * TRADE_UNIT

                # record result row with profit
                self.results.append({
                    'Time': ts,
                    'Price': price,
                    'Action': 'close',
                    'Profit': realized
                })

                # also append trajectory reward for this step (realized)
                reward = realized / 10000.0  # scale
                # push terminal transition
                next_state = build_state(price_norm, delta_logvol, 0.0, 0.0)
                self.buffer.append(Transition(state, 1, logprob, reward, next_state, True))
                self.position = None
                action_str = "close"
            else:
                action_str = "hold"

        # If not closing, compute step reward from unrealized PnL if a position exists (or zero)
        unrealized = 0.0
        if self.position is not None:
            if self.position['side'] == 'long':
                unrealized = (price - self.position['entry_price']) * TRADE_UNIT
            else:
                unrealized = (self.position['entry_price'] - price) * TRADE_UNIT
            step_reward = self._compute_step_reward(unrealized)
            done = False
            next_state = build_state(price_norm, delta_logvol, position_flag, entry_price_norm)
            # store transition with action index used
            self.buffer.append(Transition(state, action_idx, logprob, step_reward, next_state, done))
            # also log current time step (Profit zero unless closed)
            self.results.append({
                'Time': ts,
                'Price': price,
                'Action': action_str,
                'Profit': 0.0
            })
        else:
            # no position: small zero reward for holding / opening
            step_reward = 0.0
            done = False
            next_state = build_state(price_norm, delta_logvol, 0.0, 0.0)
            self.buffer.append(Transition(state, action_idx, logprob, step_reward, next_state, done))
            self.results.append({
                'Time': ts,
                'Price': price,
                'Action': action_str,
                'Profit': 0.0
            })

        # If force_close requested at final tick
        if force_close and self.position is not None:
            # Do forced close immediately (use same repayment rule)
            if self.position['side'] == 'long':
                close_price = price - TICK
                realized = (close_price - self.position['entry_price']) * TRADE_UNIT
            else:
                close_price = price + TICK
                realized = (self.position['entry_price'] - close_price) * TRADE_UNIT

            self.results.append({
                'Time': ts,
                'Price': price,
                'Action': 'forced_close',
                'Profit': realized
            })
            # add final transition
            reward = realized / 10000.0
            self.buffer.append(Transition(next_state, 1, 0.0, reward, next_state, True))
            self.position = None

        # possibly trigger update
        if len(self.buffer) >= MIN_BUFFER_TO_UPDATE:
            self._update()

        return action_str

    # -----------------------
    # finalize: 日の終わりに呼び出して結果を取得
    # -----------------------
    def finalize(self):
        # perform final update if any
        if len(self.buffer) > 0:
            self._update()

        # assemble DataFrame
        df = pd.DataFrame(self.results, columns=['Time', 'Price', 'Action', 'Profit'])
        # reset internal logs
        self.results = []
        self.buffer = []
        self.last_volume = None
        self.position = None
        self.step = 0
        return df

# -----------------------------
#  使い方（別プログラム側）
#  （質問内にあった main のループがそのまま使えます）
# -----------------------------
if __name__ == "__main__":
    # 簡単な動作チェックのためのダミーデータ
    import time
    sim = TradingSimulation()
    # example: simulate 500 ticks of a random walk
    price = 5000.0
    cumvol = 1000.0
    for i in range(500):
        ts = i
        price += random.uniform(-2, 2)  # small movement
        cumvol += random.uniform(0, 100)
        action = sim.add(ts, price, cumvol, force_close=(i == 499))
    df_out = sim.finalize()
    print(df_out.head(20))
    print("Total profit:", df_out["Profit"].sum())
