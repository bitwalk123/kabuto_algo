"""
TradingSimulation: デイトレード向けティック単位強化学習サンプル
- Interfaces:
    sim = TradingSimulation(model_path='models/policy.pth')
    action = sim.add(ts, price, volume, force_close=False)
    df_result = sim.finalize(train=True)  # train: whether to run a lightweight update
- 保存: models/policy.pth
- 依存: torch>=2.0, pandas, numpy

注意:
- このサンプルは学習の『骨組み』を示します。実運用前にハイパーパラメータ調整・検証を行ってください。
"""

import os
import math
import time
import pickle
from collections import deque, namedtuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------- ユーティリティ --------------------------------

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------------- ネットワーク --------------------------------
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        # discrete actions -> we output logits

    def forward(self, x):
        h = self.net(x)
        logits = self.mu(h)
        return logits


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------- 設定 --------------------------------
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

ACTION_MAP = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}


# 実装を簡単にするため、アクションは4種に要約する


# ------------------------- TradingSimulation --------------------------------
class TradingSimulation:
    def __init__(self,
                 model_path='models/policy.pth',
                 device=None,
                 tick_size=1.0,
                 unit=100,
                 gamma=0.99,
                 unrealized_coef=0.01,
                 training_batch_size=256):
        self.model_path = model_path
        self.tick_size = tick_size
        self.slippage = 2 * tick_size  # 呼び値の2倍
        self.unit = unit
        self.gamma = gamma
        self.unrealized_coef = unrealized_coef
        self.training_batch_size = training_batch_size

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部バッファ
        self.reset_state_buffers()

        # ネットワーク
        # 入力特徴量の次元を固定する（Price, MA, Volatility, RSI, log1p(delta_vol)） -> 5
        self.feature_names = ['Price', 'MA', 'Volatility', 'RSI', 'LogDV']
        input_dim = len(self.feature_names)
        self.actor = Actor(input_dim).to(self.device)
        self.critic = Critic(input_dim).to(self.device)
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=3e-4)

        # 経験バッファ
        self.memory = []

        # モデルの読み込み or 新規作成
        self._load_or_init_model()

    def reset_state_buffers(self):
        # ティックで更新するパディング用のデータフレーム (過去の価格を蓄える)
        self.raw_ticks = []  # list of (ts, price, volume)
        self.df_ticks = pd.DataFrame(columns=['Time', 'Price', 'Volume'])

        # ポジション管理
        self.position = 0  # +1 long, -1 short, 0 flat
        self.entry_price = None  # per-share entry price adjusted for slippage
        self.entry_side = None  # 'BUY' or 'SELL'

        # 結果記録
        self.results = []  # each: dict with Time, Price, Action, Profit

        # 内部バッファ
        self.last_volume = None
        self.step_count = 0

    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            try:
                data = torch.load(self.model_path, map_location=self.device)
                self.actor.load_state_dict(data['actor'])
                self.critic.load_state_dict(data['critic'])
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print("Failed to load existing model, creating a new one:", e)
                # overwrite later on save
        else:
            # create parent dir
            odir = os.path.dirname(self.model_path)
            if odir and not os.path.exists(odir):
                os.makedirs(odir, exist_ok=True)
            print("No existing model found — created a new model.")

    # 状態特徴量を作る: pandas 上で rolling を使って簡潔に作る
    def _compute_features(self):
        df = self.df_ticks.copy()
        if len(df) < 2:
            # 初期値
            return None
        df['MA'] = df['Price'].rolling(window=10, min_periods=1).mean()
        df['Volatility'] = df['Price'].rolling(window=10, min_periods=1).std().fillna(0.0)
        df['RSI'] = compute_rsi(df['Price'], window=14).fillna(50.0)
        # delta volume
        df['DeltaVol'] = df['Volume'].diff().fillna(0.0)
        # df['LogDV'] = np.log1p(df['DeltaVol'].clip(min=0.0))
        df['LogDV'] = np.log1p(df['DeltaVol'].clip(lower=0.0))
        # keep latest row
        last = df.iloc[-1]
        features = np.array([last['Price'], last['MA'], last['Volatility'], last['RSI'], last['LogDV']],
                            dtype=np.float32)
        # normalize features (simple): Price scaled by 1e3, MA by 1e3, Volatility by 1e2, RSI by 100, LogDV by log scale
        # Note: you may replace with better scaler for production
        features[0] /= 1000.0
        features[1] /= 1000.0
        features[2] /= 100.0
        features[3] /= 100.0
        # LogDV already compressed
        return features

    def _select_action(self, state):
        # state: numpy array
        st = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        logits = self.actor(st)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        # return action and logprob for learning
        logp = dist.log_prob(torch.tensor(action, device=self.device))
        return action, logp.item()

    def _record_result(self, ts, price, action_str, profit):
        self.results.append({'Time': ts, 'Price': price, 'Action': action_str, 'Profit': profit})

    def add(self, ts, price, volume, force_close=False):
        """Main interface called by "別プログラム".
        - ts: timestamp (float)
        - price: price float
        - volume: cumulative volume float
        - force_close: bool, whether this is end-of-day forced close

        Returns: action string (one of ACTION_MAP)
        """
        # append raw tick
        self.raw_ticks.append((ts, price, volume))
        self.df_ticks.loc[len(self.df_ticks)] = [ts, float(price), float(volume)]
        self.step_count += 1

        # compute features
        features = self._compute_features()
        if features is None:
            # not enough data yet -> HOLD
            self._record_result(ts, price, 'HOLD', 0.0)
            return 'HOLD'

        # choose action using policy
        action_idx, _ = self._select_action(features)
        action_str = ACTION_MAP[action_idx]

        # enforce trading rules: one-unit, no pyramiding
        profit = 0.0
        done = False

        # forced close overrides policy
        if force_close and self.position != 0:
            # close at current price (apply slippage accordingly)
            if self.position == 1:
                # close long -> sell to close -> receive (price - slippage)
                close_price = price - self.slippage
                profit = (close_price - self.entry_price) * self.unit
                self._record_result(ts, price, 'CLOSE(FORCED)', profit)
            elif self.position == -1:
                # close short -> buy to close -> pay (price + slippage)
                close_price = price + self.slippage
                profit = (self.entry_price - close_price) * self.unit
                self._record_result(ts, price, 'CLOSE(FORCED)', profit)
            # clear position
            # add terminal transition
            done = True
            # store transition with large reward on done
            # simple reward bookkeeping: we add a terminal reward entry in memory below
            self.position = 0
            self.entry_price = None
            self.entry_side = None
            # also append mem entry
            if hasattr(self, 'last_state') and self.last_state is not None:
                # create a terminal transition
                self.memory.append(
                    Transition(state=self.last_state, action=action_idx, reward=profit, next_state=None, done=True))
            return 'CLOSE(FORCED)'

        # Normal action handling
        if action_str == 'BUY':
            if self.position == 0:
                # open long: pay (price + slippage)
                entry_price = price + self.slippage
                self.position = 1
                self.entry_price = entry_price
                self.entry_side = 'BUY'
                self._record_result(ts, price, 'BUY', 0.0)
            else:
                # cannot open while having position -> treat as HOLD
                self._record_result(ts, price, 'HOLD', 0.0)
                action_str = 'HOLD'
        elif action_str == 'SELL':
            if self.position == 0:
                # open short: receive (price - slippage) as entry reference
                entry_price = price - self.slippage
                self.position = -1
                self.entry_price = entry_price
                self.entry_side = 'SELL'
                self._record_result(ts, price, 'SELL', 0.0)
            else:
                self._record_result(ts, price, 'HOLD', 0.0)
                action_str = 'HOLD'
        elif action_str == 'CLOSE':
            if self.position == 1:
                # close long -> sell at (price - slippage)
                close_price = price - self.slippage
                profit = (close_price - self.entry_price) * self.unit
                self._record_result(ts, price, 'CLOSE', profit)
                self.position = 0
                self.entry_price = None
                self.entry_side = None
            elif self.position == -1:
                # close short -> buy at (price + slippage)
                close_price = price + self.slippage
                profit = (self.entry_price - close_price) * self.unit
                self._record_result(ts, price, 'CLOSE', profit)
                self.position = 0
                self.entry_price = None
                self.entry_side = None
            else:
                self._record_result(ts, price, 'HOLD', 0.0)
                action_str = 'HOLD'
        else:  # HOLD
            # no trade
            # record unrealized portion as small reward (for training later)
            self._record_result(ts, price, 'HOLD', 0.0)

        # Record experience for learning
        # reward design: realized profit on close; plus small unrealized reward each step
        unrealized = 0.0
        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:
                unrealized = (price - self.entry_price) * self.unit
            else:
                unrealized = (self.entry_price - price) * self.unit
        step_reward = self.unrealized_coef * unrealized

        # If we realized profit this step (profit != 0), include it in reward
        step_reward += profit

        # store transition: state -> action -> reward -> next_state
        next_state = self._compute_features()
        done_flag = False
        self.memory.append(Transition(
            state=self.last_state if hasattr(self, 'last_state') and self.last_state is not None else features,
            action={'HOLD': 0, 'BUY': 1, 'SELL': 2, 'CLOSE': 3}[
                action_str if action_str in ['HOLD', 'BUY', 'SELL', 'CLOSE'] else 'HOLD'],
            reward=step_reward,
            next_state=next_state,
            done=done_flag))

        # update last_state cache
        self.last_state = next_state

        return action_str

    def finalize(self, train=True):
        """Called at end of day per "別プログラム". Returns results dataframe and optionally runs a training update."""
        df_result = pd.DataFrame(self.results)

        # training
        if train and len(self.memory) > 16:
            self._train_from_memory()

        # save model
        self._save_model()

        # reset for next epoch
        self.reset_state_buffers()

        return df_result

    def _train_from_memory(self):
        # Very lightweight actor-critic update using n-step returns from collected memory
        # Convert memory to arrays
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for t in self.memory:
            states.append(t.state)
            actions.append(t.action)
            rewards.append(t.reward)
            next_states.append(
                t.next_state if t.next_state is not None else np.zeros(len(self.feature_names), dtype=np.float32))
            dones.append(t.done)
        states = torch.tensor(np.vstack(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # compute targets (TD(0) simple) or n-step returns
        with torch.no_grad():
            values_next = self.critic(next_states)
            targets = rewards + self.gamma * values_next * (1.0 - dones)

        values = self.critic(states)
        advantages = targets - values

        # actor loss (policy gradient with advantage)
        logits = self.actor(states)
        logp = torch.log_softmax(logits, dim=-1)
        logp_actions = logp[range(len(actions)), actions]
        actor_loss = -(logp_actions * advantages.detach()).mean()

        # critic loss
        critic_loss = nn.MSELoss()(values, targets.detach())

        # step optimizers
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_a.step()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_c.step()

        print(
            f"Trained on {len(self.memory)} transitions. Actor loss={actor_loss.item():.4f}, Critic loss={critic_loss.item():.4f}")

        # clear memory after training
        self.memory = []

    def _save_model(self):
        data = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(data, self.model_path)
        print(f"Model saved to {self.model_path}")


# ------------------------- 実行例 (別プログラム側で使う) ----------------------
if __name__ == '__main__':
    # テスト用の簡易シミュレーション: ダミーデータを流して動作確認
    sim = TradingSimulation()

    # generate dummy tick data (1 second steps)
    np.random.seed(0)
    times = np.arange(0, 600)  # 10 minutes
    prices = 5000 + np.cumsum(np.random.randn(len(times)))  # random walk
    volumes = np.cumsum(np.random.randint(1, 100, size=len(times))).astype(float)

    for i in range(len(times)):
        ts = float(times[i])
        price = float(max(100.0, prices[i]))
        vol = float(volumes[i])
        force_close = (i == len(times) - 1)
        action = sim.add(ts, price, vol, force_close=force_close)
        # print(ts, price, action)

    df_out = sim.finalize(train=True)
    print(df_out.head())
    print('Total profit:', df_out['Profit'].sum())
