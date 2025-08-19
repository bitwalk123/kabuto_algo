import os
import random
import numpy as np
import pandas as pd
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ---- 強化学習用 Q-Network ----
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---- DQN エージェント ----
class DQNAgent:
    def __init__(self, state_size=1, action_size=3, model_path="trading_model.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.1
        self.lr = 0.001
        self.batch_size = 32
        self.model_path = model_path

        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("✅ 学習済みモデルを読み込みました")
        else:
            print("⚡ 新しいモデルを初期化しました")

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)


# ---- 売買シミュレーション環境 ----
class TradingSimulation:
    def __init__(self):
        self.agent = DQNAgent()
        self.position = 0  # +100 (買い), -100 (売り), 0 (なし)
        self.entry_price = None
        self.results = []
        self.last_price = None

    def add(self, ts, price, force_close=False):
        state = [price]
        action = self.agent.act(state)

        reward = 0
        action_str = "ホールド"

        if self.position == 0:  # ノーポジ
            if action == 1:  # 新規買い
                self.position = 100
                self.entry_price = price
                action_str = "新規買い"
            elif action == 2:  # 新規売り
                self.position = -100
                self.entry_price = price
                action_str = "新規売り"

        else:  # ポジションあり
            if (self.position > 0 and action == 2) or (self.position < 0 and action == 1) or force_close:
                # 決済
                pnl = (price - self.entry_price) * self.position
                reward = pnl
                self.position = 0
                self.entry_price = None
                action_str = "決済"

        # 記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "売買アクション": action_str,
            "報酬額": reward
        })

        # 強化学習用メモリ更新
        next_state = [price]
        done = 1.0 if force_close else 0.0
        self.agent.remember(state, action, reward, next_state, done)
        self.agent.replay()

        self.last_price = price
        return action_str

    def finalize(self, filename="trade_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        self.agent.save()
        # print(f"✅ 結果を {filename} に保存しました")
        return df


# ---- 動作サンプル ----
if __name__ == "__main__":
    sim = TradingSimulation()
    ticks = [(i, 1000 + np.sin(i / 5) * 10) for i in range(60)]  # ダミー: 60秒分の株価
    for ts, price in ticks:
        action = sim.add(ts, price)
        print(ts, price, action)

    sim.finalize()
