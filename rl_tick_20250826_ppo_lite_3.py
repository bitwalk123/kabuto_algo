"""
良さげなモデルだけど少し重い。

PPO-lite ベースのデイトレード強化学習サンプル
- 入力: 秒間ティック (ts, price, volume) を add(ts, price, volume, force_close=False) で通知
- 使い方: 別プログラムから行単位で add を呼び、最後に finalize() を呼ぶ
- 要件を満たすように設計

依存パッケージ:
- numpy, pandas, gymnasium, torch

保存ファイル:
- models/policy.pth

注意:
- 本サンプルは学習を軽量化した "PPO-lite"（簡易Actor-Critic）を実装しています。
- 実運用前に報酬設計やハイパーパラメータを必ず調整してください。
"""

import os
import math
import random
from collections import deque, namedtuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# --- 定数 / 設定 ---
TICK = 1  # 呼び値 (円)
SLIPPAGE = 2 * TICK  # 指定どおり呼び値の2倍
UNIT = 100  # 売買単位（株）
POSITION_NONE = 0
POSITION_LONG = 1
POSITION_SHORT = -1

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "policy.pth")

# --- ユーティリティ ---

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- ネットワーク ---
class ActorCritic(nn.Module):
    def __init__(self, obs_size, hidden_size=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

# --- 経験バッファ（1日分） ---
Experience = namedtuple('Experience', ['obs', 'action', 'logp', 'reward', 'done', 'value'])

class ReplayBuffer:
    def __init__(self):
        self.data = []

    def push(self, *args):
        self.data.append(Experience(*args))

    def clear(self):
        self.data = []

    def __len__(self):
        return len(self.data)

# --- トレーディングシミュレーション ---
class TradingSimulation:
    def __init__(self, device='cpu'):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.device = torch.device(device)

        # 内部状態
        self.reset_episode_state()

        # モデル関連
        self.obs_size = 6  # price, ma10, vol10, rsi14, log1p(delta_volume), position_indicator
        self.n_actions = 4  # 購入・売却・返済・ホールド（詳細は下）
        self.model = ActorCritic(self.obs_size, hidden_size=128, n_actions=self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.buffer = ReplayBuffer()

        # モデル読み込み or 新規作成
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                print(f"Loaded existing model from {MODEL_PATH}")
                self.model_loaded = True
            except Exception as e:
                print(f"Failed to load model (will create new). Error: {e}")
                self.model_loaded = False
        else:
            print("No existing model found — created a new model.")
            self.model_loaded = False

        # ハイパーパラ
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.adv_norm_eps = 1e-8
        self.batch_epochs = 4
        self.clip_eps = 0.2

    def reset_episode_state(self):
        # 入力シリーズを保持
        self.df_buf = pd.DataFrame(columns=["Time", "Price", "Volume"])
        self.last_volume = None

        # ポジション情報
        self.position = POSITION_NONE
        self.position_price = 0.0  # 新規建玉価格（約定価格）
        self.position_size = 0

        # 結果の記録
        self.records = []  # list of dict

        # トレーニング用
        self.buffer = ReplayBuffer()

    # 外部から1ティック通知されるインターフェイス
    def add(self, ts: float, price: float, volume: float, force_close: bool = False):
        # 生データを貯める
        self.df_buf.loc[len(self.df_buf)] = [ts, float(price), float(volume)]

        # 前処理: 特徴量計算
        obs = self._compute_observation()

        # モデルから行動決定
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=self.device)).item()
            value = value.item()

        # action mapping: 0=buy (新規買い or 新規売り depending on position), 1=sell, 2=close (返済), 3=hold
        # For simplicity and better learning we will interpret 0 as 新規買い, 1 as 新規売り, 2 as 返済, 3 as ホールド

        # 実際の取引処理
        reward, executed_action, profit = self._execute_action(action, price, force_close)

        # バッファに保存
        self.buffer.push(obs, action, logp, reward, force_close, value)

        # 記録
        rec_action_str = self._action_to_str(executed_action)
        self.records.append({
            'Time': ts,
            'Price': price,
            'Action': rec_action_str,
            'Profit': profit
        })

        # 返り値はアクション文字列（要求に沿う）
        return rec_action_str

    def _compute_observation(self):
        # 最新の df_buf を用いて特徴量を計算して、観測ベクトルを返す
        df = self.df_buf.copy()
        df['Price'] = df['Price'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # delta volume
        if len(df) >= 2:
            delta_v = df['Volume'].diff().fillna(0.0).iloc[-1]
        else:
            delta_v = 0.0
        log_dv = math.log1p(max(delta_v, 0.0))

        # MA and Volatility using window=10
        if len(df) >= 10:
            ma10 = df['Price'].rolling(window=10).mean().iloc[-1]
            vol10 = df['Price'].rolling(window=10).std().iloc[-1]
        else:
            ma10 = df['Price'].mean()
            vol10 = df['Price'].std() if len(df) > 1 else 0.0

        # RSI
        if len(df) >= 2:
            rsi14 = compute_rsi(df['Price'], window=14).iloc[-1]
            if pd.isna(rsi14):
                rsi14 = 50.0
        else:
            rsi14 = 50.0

        price = float(df['Price'].iloc[-1])
        ma10 = 0.0 if pd.isna(ma10) else float(ma10)
        vol10 = 0.0 if pd.isna(vol10) else float(vol10)

        pos_ind = 0.0
        if self.position == POSITION_LONG:
            pos_ind = 1.0
        elif self.position == POSITION_SHORT:
            pos_ind = -1.0

        # 正規化: price はそのままにしておくが学習安定のため価格を除算することも検討
        obs = np.array([price, ma10, vol10, rsi14, log_dv, pos_ind], dtype=np.float32)
        # 簡易スケーリング
        obs[0] = obs[0] / 10000.0  # price scale
        obs[1] = obs[1] / 10000.0
        obs[2] = obs[2] / 10000.0
        obs[3] = obs[3] / 100.0
        obs[4] = obs[4] / (1.0 + obs[4]) if obs[4] >= 0.0 else obs[4]

        return obs

    def _execute_action(self, action_idx, price, force_close):
        # action_idx: 0=buy,1=sell,2=close,3=hold
        executed = 'ホールド'
        realized_profit = 0.0

        if force_close and self.position != POSITION_NONE:
            # 強制返済はその時の市場価格で決済
            if self.position == POSITION_LONG:
                close_price = price - SLIPPAGE  # 返済売り
                realized_profit = (close_price - self.position_price) * UNIT
                executed = '返済売り（強制）'
            elif self.position == POSITION_SHORT:
                close_price = price + SLIPPAGE  # 返済買い
                realized_profit = (self.position_price - close_price) * UNIT
                executed = '返済買い（強制）'
            # reset position
            self.position = POSITION_NONE
            self.position_price = 0.0
            self.position_size = 0
            return realized_profit, executed, realized_profit

        if action_idx == 0:  # 新規買い
            if self.position == POSITION_NONE:
                # 約定条件: (price + SLIPPAGE)
                exec_price = price + SLIPPAGE
                self.position = POSITION_LONG
                self.position_price = exec_price
                self.position_size = UNIT
                executed = '新規買い'
            else:
                executed = 'ホールド'  # ルール: 建玉がある間は新規トレードしない

        elif action_idx == 1:  # 新規売り
            if self.position == POSITION_NONE:
                exec_price = price - SLIPPAGE
                self.position = POSITION_SHORT
                self.position_price = exec_price
                self.position_size = UNIT
                executed = '新規売り'
            else:
                executed = 'ホールド'

        elif action_idx == 2:  # 返済
            if self.position == POSITION_LONG:
                close_price = price - SLIPPAGE
                realized_profit = (close_price - self.position_price) * UNIT
                executed = '返済売り'
                # reset
                self.position = POSITION_NONE
                self.position_price = 0.0
                self.position_size = 0
            elif self.position == POSITION_SHORT:
                close_price = price + SLIPPAGE
                realized_profit = (self.position_price - close_price) * UNIT
                executed = '返済買い'
                self.position = POSITION_NONE
                self.position_price = 0.0
                self.position_size = 0
            else:
                executed = 'ホールド'

        else:  # hold
            executed = 'ホールド'

        # 含み益の一部を小さく報酬に与える（例: 1%）
        unrealized = 0.0
        if self.position == POSITION_LONG:
            unrealized = (price - self.position_price) * UNIT
        elif self.position == POSITION_SHORT:
            unrealized = (self.position_price - price) * UNIT

        reward = realized_profit + 0.01 * unrealized
        return reward, executed, realized_profit

    def _action_to_str(self, s):
        return s

    # finalize: エピソード終了時に呼ばれる。学習（モデル更新）/ 結果返却 を行う
    def finalize(self, train=True):
        # DataFrame を作成して返す
        df_result = pd.DataFrame(self.records)

        # 学習を行う
        if train and len(self.buffer) > 0:
            print(f"Training on {len(self.buffer)} steps")
            self._update_model()
            # モデル保存
            torch.save(self.model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

        # リセット
        self.reset_episode_state()
        return df_result

    def _update_model(self):
        # シンプルな A2C / PPO-lite の実装（オンポリシー）
        # バッファからテンソル作成
        obs = np.array([e.obs for e in self.buffer.data], dtype=np.float32)
        actions = np.array([e.action for e in self.buffer.data], dtype=np.int64)
        rewards = np.array([e.reward for e in self.buffer.data], dtype=np.float32)
        dones = np.array([e.done for e in self.buffer.data], dtype=np.float32)
        values = np.array([e.value for e in self.buffer.data], dtype=np.float32)

        # discounted returns
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.adv_norm_eps)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # perform several epochs
        for epoch in range(self.batch_epochs):
            logits, value_pred = self.model(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logp_all = dist.log_prob(actions_t)

            # policy loss (PPO clipped)
            ratio = torch.exp(logp_all - logp_all.detach())  # simplified (since old logp not stored)
            # Note: we don't store old logp for simplicity; this is a LIGHT approximation => behave like A2C
            policy_loss = -(adv_t * logp_all).mean()

            value_loss = (returns_t - value_pred).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        print("Update finished.")


# --- 単体テスト / 実行例 ---
if __name__ == '__main__':
    # デモ用: CSV から読み込んで sim を回す
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='data/tick_sample.csv')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # サンプル CSV が存在しない場合はランダムデータを作る
    if not os.path.exists(args.file):
        print(f"{args.file} not found. Creating synthetic sample...")
        ts = np.arange(0, 10 * 60)  # 10 minutes of per-second ticks
        price = np.cumsum(np.random.randn(len(ts)) * 0.5) + 5000
        volume = np.cumsum(np.abs(np.random.randn(len(ts)) * 1000)).astype(float)
        df = pd.DataFrame({"Time": ts, "Price": price, "Volume": volume})
        os.makedirs(os.path.dirname(args.file), exist_ok=True)
        df.to_csv(args.file, index=False)

    df = pd.read_csv(args.file)

    sim = TradingSimulation()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for i, row in df.iterrows():
            ts = row['Time']
            price = row['Price']
            volume = row['Volume']
            is_last = (i == len(df) - 1)
            action = sim.add(ts, price, volume, force_close=is_last)
        df_result = sim.finalize(train=True)
        profit = df_result['Profit'].sum()
        print(f"Epoch {epoch} total profit: {profit}")

    print('done')
