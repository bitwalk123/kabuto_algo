import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# PPO-lite Actor-Critic ネットワーク
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.actor(h), self.critic(h)

class TradingSimulation:
    def __init__(self, model_file="ppo_model.pth"):
        self.state_dim = 3  # Price, Volume(diff), Profit
        self.action_dim = 3  # buy, sell, hold
        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.model_file = model_file
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file))
        self.reset()

    def reset(self):
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.prev_volume = 0
        self.history = []
        self.memory = []

    def get_state(self, price, volume):
        # 出来高を差分化 & 正規化
        vol_diff = max(volume - self.prev_volume, 0)
        self.prev_volume = volume
        vol_norm = np.log1p(vol_diff) / 10.0
        price_norm = price / 10000.0
        profit_norm = self.total_profit / 100000.0
        return np.array([price_norm, vol_norm, profit_norm], dtype=np.float32)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits, value = self.model(state_tensor)
        logits = torch.clamp(logits, -10, 10)  # 安定化
        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            action = 2  # hold
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action, value

    def add(self, ts, price, volume, force_close=False):
        state = self.get_state(price, volume)
        action, value = self.select_action(state)

        reward = 0
        action_label = "ホールド"

        if action == 0:  # buy
            if self.position == 0:
                self.position = 100
                self.entry_price = price
                action_label = "新規買い"
            else:
                action_label = "買い（無効）"
        elif action == 1:  # sell
            if self.position > 0:
                profit = (price - self.entry_price) * self.position
                self.total_profit += profit
                reward = profit / 1000.0
                self.position = 0
                action_label = "返済売り"
            else:
                action_label = "売り（無効）"

        if force_close and self.position > 0:
            profit = (price - self.entry_price) * self.position
            self.total_profit += profit
            reward += profit / 1000.0
            self.position = 0
            action_label = "強制返済"

        unrealized = 0
        if self.position > 0:
            unrealized = (price - self.entry_price) * self.position / 1000.0
            reward += unrealized * 0.01

        self.history.append({
            "Time": ts,
            "Price": price,
            "売買アクション": action_label,
            "Reward": reward,
            "Profit": self.total_profit
        })

        self.memory.append((state, action, reward, value))
        return action_label

    def finalize(self):
        if len(self.history) == 0:
            return pd.DataFrame(columns=["Time", "Price", "売買アクション", "Reward", "Profit"])

        df_result = pd.DataFrame(self.history)

        # 学習処理 (簡易版 PPO)
        returns = []
        G = 0
        for _, _, reward, value in reversed(self.memory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        for (state, action, reward, value), G in zip(self.memory, returns):
            logits, val = self.model(torch.tensor(state))
            probs = torch.softmax(torch.clamp(logits, -10, 10), dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action))
            advantage = G - val.item()
            actor_loss = -log_prob * advantage
            critic_loss = advantage**2
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.save(self.model.state_dict(), self.model_file)
        self.reset()
        return df_result

# 単体テスト用
if __name__ == "__main__":
    sim = TradingSimulation()
    data = [
        (1, 100, 1000),
        (2, 101, 1500),
        (3, 102, 2000),
        (4, 101, 2200),
    ]
    for i, (t, p, v) in enumerate(data):
        sim.add(t, p, v, force_close=(i == len(data)-1))
    df = sim.finalize()
    print(df)
    print("最終損益", df["Profit"].iloc[-1])
