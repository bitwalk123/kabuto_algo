"""
強化学習（DQN）を使ったデイトレード売買シミュレーション・サンプル
- Python でそのまま動く単一ファイルのサンプル
- PyTorch を利用（torch>=1.8 推奨）

使い方（例）:
  python rl_tick_dqn_20250817.py --data_dir /mnt/data --symbol 7011 --epochs 10

出力:
  - trade_results.csv (指定ディレクトリに出力)
  - dqn_model_{symbol}.pth (モデル保存)

注意:
  - tick CSV は列 Time, Price を持つことを想定
  - 売買単位: 100株 (変更可)
  - 約定手数料は考慮しない
  - ナンピンしない（建玉がある間は新規建てを行わない）

"""

import os
import argparse
import glob
import random
import math
import csv
from collections import deque, namedtuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------- 環境定義 ---------------------------

TradeEvent = namedtuple('TradeEvent', ['epoch', 'time', 'price', 'action', 'reward'])

class SimpleTickEnv:
    """シンプルなティック環境
    - state: 直近 window_size の価格を正規化して与える
    - actions: 0=Hold, 1=OpenLong, 2=OpenShort, 3=ClosePosition
    - 100株単位固定
    - 信用売買（ショート）を許可
    - ポジションがある間は新規建て不可（ナンピン禁止）
    - 報酬: 建玉返済（Close）時に確定 （円、株数を勘案）
    """
    def __init__(self, prices, times, window_size=10, lot=100):
        assert len(prices) == len(times)
        self.prices = prices
        self.times = times
        self.window = window_size
        self.lot = lot
        self.reset()

    def reset(self):
        self.idx = self.window  # 最初に観測可能なインデックス
        self.position = 0  # 0=none, +1=long, -1=short
        self.entry_price = None
        self.done = False
        return self._get_state()

    def _get_state(self):
        # 直近 window の価格を最後の価格で割って正規化（1.0 が最新）
        window_prices = self.prices[self.idx-self.window:self.idx]
        base = window_prices[-1]
        arr = np.array(window_prices) / base - 1.0
        # 追加情報: position をスケールして入れる
        pos_feat = np.array([self.position], dtype=np.float32)
        return np.concatenate([arr.astype(np.float32), pos_feat], axis=0)

    def step(self, action):
        """
        action の適用. 返り値: next_state, reward, done, info(dict)
        """
        reward = 0.0
        info = {}
        price = self.prices[self.idx]
        time = self.times[self.idx]

        # 行動の解釈
        if action == 0:  # Hold
            pass
        elif action == 1:  # Open Long
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                info['event'] = 'open_long'
            else:
                # 無効行動 -> ペナルティは与えないが何もしない
                info['event'] = 'open_long_ignored'
        elif action == 2:  # Open Short
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                info['event'] = 'open_short'
            else:
                info['event'] = 'open_short_ignored'
        elif action == 3:  # Close Position
            if self.position != 0:
                # 建玉返済: 損益確定
                if self.position == 1:
                    pnl = (price - self.entry_price) * self.lot
                else:
                    pnl = (self.entry_price - price) * self.lot
                reward = float(pnl)
                info['event'] = 'close'
                info['pnl'] = pnl
                # ポジション解消
                self.position = 0
                self.entry_price = None
            else:
                info['event'] = 'close_ignored'
        else:
            raise ValueError('Unknown action')

        # 位置を進める
        self.idx += 1
        if self.idx >= len(self.prices):
            self.done = True

            # 最終ティックでポジションが残っていれば強制決済（終値で決済）
            if self.position != 0:
                final_price = self.prices[-1]
                if self.position == 1:
                    pnl = (final_price - self.entry_price) * self.lot
                else:
                    pnl = (self.entry_price - final_price) * self.lot
                reward += float(pnl)
                info['final_close_pnl'] = pnl
                self.position = 0
                self.entry_price = None

        next_state = self._get_state() if not self.done else None
        return next_state, reward, self.done, info, time, price

# --------------------------- DQN 定義 ---------------------------

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------- トレーニング / 実行 ---------------------------

def train_on_file(filepath, agent, target_agent, optimizer, replay, device, args, epoch, trade_log):
    df = pd.read_csv(filepath)
    assert 'Time' in df.columns and 'Price' in df.columns
    prices = df['Price'].values.astype(np.float32)
    times = df['Time'].values

    env = SimpleTickEnv(prices, times, window_size=args.window, lot=args.lot)
    state = env.reset()

    step_count = 0
    epsilon = max(args.eps_end, args.eps_start * (args.eps_decay ** epoch))

    while True:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # 行動選択
        if random.random() < epsilon:
            action = random.randrange(args.n_actions)
        else:
            with torch.no_grad():
                q = agent(state_t)
                action = int(q.argmax().item())

        next_state, reward, done, info, time, price = env.step(action)

        # 結果のログ（建玉返済時に報酬が生じたらログ行を追加）
        if info.get('event') in ('open_long', 'open_short'):
            trade_log.append(TradeEvent(epoch, int(time), float(price), info['event'], 0.0))
        if info.get('event') == 'close':
            pnl = info.get('pnl', 0.0)
            trade_log.append(TradeEvent(epoch, int(time), float(price), 'close', float(pnl)))

        # リプレイバッファに追加
        replay.push(state, action, reward, next_state if next_state is not None else np.zeros_like(state), done)

        state = next_state if next_state is not None else state

        # 学習
        if len(replay) >= args.batch_size:
            batch_state, batch_action, batch_reward, batch_next, batch_done = replay.sample(args.batch_size)
            bs = torch.tensor(batch_state, dtype=torch.float32, device=device)
            ba = torch.tensor(batch_action, dtype=torch.long, device=device).unsqueeze(1)
            br = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
            bnext = torch.tensor(batch_next, dtype=torch.float32, device=device)
            bdone = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = agent(bs).gather(1, ba)
            with torch.no_grad():
                max_next = target_agent(bnext).max(1)[0].unsqueeze(1)
                target_q = br + (1 - bdone) * args.gamma * max_next

            loss = nn.functional.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_count += 1
        if done:
            break

    return trade_log


def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='CSV ファイルがあるディレクトリ')
    parser.add_argument('--symbol', type=str, default='7011')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--lot', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 固定
    args.n_actions = 4  # hold, open long, open short, close
    args.batch_size = args.batch_size
    args.gamma = args.gamma
    args.window = args.window
    args.lot = args.lot
    args.eps_start = args.eps_start
    args.eps_end = args.eps_end
    args.eps_decay = args.eps_decay
    args.tau = args.tau

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ファイル一覧
    pattern = os.path.join(args.data_dir, f'tick_*_{args.symbol}.csv')
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        # 直接指定ファイル名が与えられている場合を想定
        alt = os.path.join(args.data_dir, f'tick_*.csv')
        files = sorted(glob.glob(alt))
        if len(files) == 0:
            raise FileNotFoundError(f'No tick CSV files found in {args.data_dir} for symbol {args.symbol}')

    print(f'Found {len(files)} file(s). Using first for demo: {files[0]}')

    # モデル準備
    input_dim = args.window + 1  # window 価格 + position flag
    output_dim = args.n_actions

    agent = DQN(input_dim, output_dim).to(device)
    target_agent = DQN(input_dim, output_dim).to(device)
    target_agent.load_state_dict(agent.state_dict())

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    replay = ReplayBuffer(capacity=20000)

    model_path = os.path.join(args.data_dir, f'dqn_model_{args.symbol}.pth')
    if args.load_model and os.path.exists(model_path):
        print('Loading existing model', model_path)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        target_agent.load_state_dict(agent.state_dict())

    trade_log = []

    # エポック単位でファイルを順に回す（通常は日々のファイルを順に学習させる）
    for epoch in range(args.epochs):
        print(f'=== Epoch {epoch+1}/{args.epochs} ===')
        # 今回は単純に同じ日ファイルを使って学習を重ねる実装
        for filepath in files:
            trade_log = train_on_file(filepath, agent, target_agent, optimizer, replay, device, args, epoch, trade_log)
            # ターゲットネットワークを少し更新
            soft_update(target_agent, agent, args.tau)

        # エポックごとにモデル保存
        if args.save_model:
            torch.save(agent.state_dict(), model_path)
            print('Saved model to', model_path)

    # トレードログを CSV に出力
    out_path = os.path.join(args.data_dir, 'trade_results.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Time', 'Price', '売買アクション', '報酬額'])
        for ev in trade_log:
            writer.writerow([ev.epoch, ev.time, ev.price, ev.action, ev.reward])

    print('Finished. trade_results.csv written to', out_path)

if __name__ == '__main__':
    main()
