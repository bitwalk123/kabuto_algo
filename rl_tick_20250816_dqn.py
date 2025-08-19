#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日次で tick CSV（Time, Price）を読み込み、
単一銘柄・単一ポジション（100株固定）・ナンピン禁止の条件で
DQN による強化学習を行う最小サンプル。

特徴:
- 1ティックずつ逐次処理（分足などへの変換なし）
- 指値=現在値で約定する前提（信用売買可、空売り可）
- 100株のみ建てられる。建玉を返済するまで次の新規は不可
- 手数料は考慮しない
- 学習は毎日別 CSV を読み込んで継続。本体モデルと経験リプレイを保存/再利用
- 出力: trade_results.csv（Time, Price, 売買アクション, 報酬額）

使い方:
    python rl_tick_dqn.py tick_YYYYMMDD_XXXX.csv

保存ファイル:
    - dqn_model.pt: DQN モデルの重み
    - replay.pkl   : 経験リプレイ
    - trade_results.csv: 実行日のトレードログ

依存:
    pip install pandas numpy torch

注意:
    ・本サンプルは研究/教育目的の骨格コードです。実運用には一切使用しないでください。
"""

import os
import sys
import pickle
from collections import deque, namedtuple
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ====== 環境の固定シード（簡易） ======
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====== 定数 ======
POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = -1

ACTION_HOLD = 0  # そのまま
ACTION_OPEN_LONG = 1  # 新規買い（フラット時のみ有効）
ACTION_OPEN_SHORT = 2  # 新規売り（フラット時のみ有効）
ACTION_CLOSE = 3  # 返済（建玉ありのときのみ有効）

ACTION_NAMES_JP = {
    ACTION_HOLD: "保持",
    ACTION_OPEN_LONG: "新規買い",
    ACTION_OPEN_SHORT: "新規売り",
    ACTION_CLOSE: "返済",
}

SHARES = 100  # 100株単位

# ====== 経験リプレイ ======
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'mask'))


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(*args)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ====== DQN モデル ======
class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ====== 環境 ======
class TickTradingEnv:
    """Tick 毎に状態→行動→報酬を返す簡易環境。
    状態: 直近 N ティックのリターン（標準化） + 建玉状態（-1/0/1）
    行動: [保持, 新規買い, 新規売り, 返済]
    報酬: (次ティック価格 - 現在価格) * ポジション * 100株 （次状態への遷移で得る）
    取引価格は常に現在の価格（指値=現在値で即約定）
    """

    def __init__(self, times: np.ndarray, prices: np.ndarray, window: int = 20):
        assert (len(times) == len(prices) and len(prices) >= window + 2), "価格系列が短すぎます（少なくとも window+2 必要）"
        self.times = times.astype(int)
        self.prices = prices.astype(float)
        self.window = window
        self.i = window  # 現在のインデックス（状態は過去 window を使う）
        self.position = POSITION_FLAT
        self.entry_price = None
        self.done = False
        self.ret_buf = deque(maxlen=window)
        # 初期化: 最初の window 区間のリターン
        for k in range(1, window + 1):
            r = self.prices[k] / self.prices[k - 1] - 1.0
            self.ret_buf.append(r)

    def _state(self):
        # リターン標準化
        rets = np.array(self.ret_buf, dtype=np.float32)
        mu = rets.mean() if len(rets) else 0.0
        sd = rets.std() if len(rets) else 1.0
        if sd < 1e-8: sd = 1.0
        norm = (rets - mu) / sd
        pos = np.array([self.position], dtype=np.float32)
        return np.concatenate([norm, pos], axis=0)  # shape: [window+1]

    def _action_mask(self):
        # 無効行動をマスク（1=有効, 0=無効）
        mask = np.ones(4, dtype=np.float32)
        if self.position == POSITION_FLAT:
            mask[ACTION_CLOSE] = 0.0
        else:
            mask[ACTION_OPEN_LONG] = 0.0
            mask[ACTION_OPEN_SHORT] = 0.0
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("環境は終了しています")
        t = self.times[self.i]
        px = self.prices[self.i]

        # 行動適用（無効行動は罰則を与え、状態遷移は HOLD と同等に）
        mask = self._action_mask()
        invalid = mask[action] < 0.5
        penalty = -0.0001  # ごく小さな罰則

        realized = 0.0
        if not invalid:
            if action == ACTION_OPEN_LONG and self.position == POSITION_FLAT:
                self.position = POSITION_LONG
                self.entry_price = px
            elif action == ACTION_OPEN_SHORT and self.position == POSITION_FLAT:
                self.position = POSITION_SHORT
                self.entry_price = px
            elif action == ACTION_CLOSE and self.position != POSITION_FLAT:
                # ここで即時に実現損益を計上する設計もあり得るが、
                # 本サンプルでは一貫性のため報酬は "次ティックの差分PnL" に統一。
                self.position = POSITION_FLAT
                self.entry_price = None
            # HOLD は何もしない

        # 次ティックへ（報酬は価格差 * ポジション * 株数）
        next_i = self.i + 1
        next_px = self.prices[next_i]
        reward = ((next_px - px) * (
                    SHARES * (1 if self.position == POSITION_LONG else -1 if self.position == POSITION_SHORT else 0)))
        if invalid:
            reward += penalty

        # 次状態の準備
        self.i = next_i
        # リターンを更新
        self.ret_buf.append(self.prices[self.i] / self.prices[self.i - 1] - 1.0)

        # 終了判定
        if self.i >= len(self.prices) - 1:
            self.done = True

        info = {
            'time': t,
            'price': px,
            'action_mask': mask.copy(),
            'invalid': invalid,
        }
        return self._state(), float(reward), self.done, info

    def reset(self):
        self.i = self.window
        self.position = POSITION_FLAT
        self.entry_price = None
        self.done = False
        self.ret_buf.clear()
        for k in range(1, self.window + 1):
            r = self.prices[k] / self.prices[k - 1] - 1.0
            self.ret_buf.append(r)
        return self._state()


# ====== エージェント（DQN） ======
class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int = 4, lr: float = 1e-3, gamma: float = 0.999, batch_size: int = 64,
                 eps_start: float = 0.10, eps_end: float = 0.01, eps_decay: float = 0.9995, target_tau: float = 0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q = QNet(state_dim, n_actions).to(self.device)
        self.target = QNet(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_tau = target_tau
        self.n_actions = n_actions

    def act(self, state: np.ndarray, mask: np.ndarray):
        # ε-greedy + 行動マスク（無効行動は -inf に）
        if np.random.rand() < self.eps:
            # 有効行動からランダム選択
            valid_idx = np.where(mask > 0.5)[0]
            a = int(np.random.choice(valid_idx)) if len(valid_idx) else 0
            return a
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s).squeeze(0).cpu().numpy()
            q_masked = q + (np.log(mask + 1e-8) - np.log(1.0))  # mask=0 → -inf 相当
            return int(np.argmax(q_masked))

    def update(self, replay: ReplayBuffer):
        if len(replay) < self.batch_size:
            return 0.0
        batch = replay.sample(self.batch_size)
        state = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        mask = torch.tensor(np.array(batch.mask), dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_sa = self.q(state).gather(1, action)

        # max_a' Q_target(s',a') with mask
        with torch.no_grad():
            q_next_all = self.target(next_state)
            q_next_all[mask < 0.5] = -1e9
            q_next, _ = q_next_all.max(dim=1, keepdim=True)
            target = reward + (1.0 - done) * self.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()

        # Soft update target
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.copy_(self.target_tau * p.data + (1 - self.target_tau) * tp.data)

        # ε 減衰
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return float(loss.item())

    def save(self, path: str):
        torch.save({
            'model': self.q.state_dict(),
            'target': self.target.state_dict(),
            'eps': self.eps,
        }, path)

    def load(self, path: str):
        obj = torch.load(path, map_location=self.device)
        self.q.load_state_dict(obj['model'])
        self.target.load_state_dict(obj.get('target', obj['model']))
        self.eps = obj.get('eps', self.eps)


# ====== 学習実行 ======

def run_day(csv_path: str,
            model_path: str = 'dqn_model.pt',
            replay_path: str = 'replay.pkl',
            window: int = 20,
            lr: float = 1e-3,
            gamma: float = 0.999,
            batch_size: int = 64,
            steps_per_update: int = 1,
            max_train_steps_per_tick: int = 1,
            ):
    # CSV 読み込み（ヘッダ: Time, Price）
    df = pd.read_csv(csv_path)
    if not set(['Time', 'Price']).issubset(df.columns):
        raise ValueError('CSV に Time, Price 列が必要です')
    times = df['Time'].to_numpy()
    prices = df['Price'].to_numpy(dtype=float)

    env = TickTradingEnv(times, prices, window=window)
    state = env.reset()

    state_dim = len(state)
    agent = DQNAgent(state_dim=state_dim, n_actions=4, lr=lr, gamma=gamma, batch_size=batch_size)

    # 既存モデル/リプレイのロード
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model: {model_path} (eps={agent.eps:.4f})")
    else:
        print("No existing model. Training from scratch.")

    if os.path.exists(replay_path):
        try:
            with open(replay_path, 'rb') as f:
                replay = pickle.load(f)
            if not isinstance(replay, ReplayBuffer):
                raise ValueError('Invalid replay file')
            print(f"Loaded replay: {replay_path} (size={len(replay)})")
        except Exception as e:
            print(f"Replay load failed: {e}; creating new buffer.")
            replay = ReplayBuffer()
    else:
        replay = ReplayBuffer()

    # ログ出力用
    logs = []  # dict: time, price, action_name, reward

    total_steps = 0
    episode_done = False

    while not episode_done:
        mask = env._action_mask()
        action = agent.act(state, mask)
        next_state, reward, done, info = env.step(action)

        # リプレイへ
        replay.push(state, action, reward, next_state, done, mask)

        # 学習
        for _ in range(max_train_steps_per_tick):
            agent.update(replay)

        # ログへ
        logs.append({
            'Time': int(info['time']),
            'Price': float(info['price']),
            '売買アクション': ACTION_NAMES_JP[action] + ("（無効）" if info['invalid'] else ""),
            '報酬額': float(reward),
        })

        state = next_state
        total_steps += 1
        episode_done = done

    # エピソード終了時、建玉が残っていたら強制返済を記録（報酬は 0 とする）
    # ※ 学習報酬は各ティック差分で既に計上されているため、ここで実現損益を加算しない
    if env.position != POSITION_FLAT:
        last_time = int(env.times[env.i])
        last_price = float(env.prices[env.i])
        logs.append({
            'Time': last_time,
            'Price': last_price,
            '売買アクション': ACTION_NAMES_JP[ACTION_CLOSE] + "（強制）",
            '報酬額': 0.0,
        })
        # 内部状態もフラットに戻しておく（学習への影響はなし）
        env.position = POSITION_FLAT
        env.entry_price = None

    # 保存
    agent.save(model_path)
    with open(replay_path, 'wb') as f:
        pickle.dump(replay, f)

    # 結果 CSV 出力
    out_df = pd.DataFrame(logs)
    out_df.to_csv('trade_results.csv', index=False)

    # 概要プリント
    total_reward = out_df['報酬額'].sum()
    realized_like = total_reward  # 本サンプルでは差分PnLの合計として定義
    print(f"Steps: {total_steps}, Replay: {len(replay)}, Eps: {agent.eps:.4f}")
    print(f"日合計 報酬額(差分PnL合計): {realized_like:.2f} 円")

    return out_df


# ====== エントリーポイント ======
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rl_tick_dqn.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    run_day(csv_path)
