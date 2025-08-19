import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        prob = self.actor(state)
        value = self.critic(state)
        return prob, value


class TradingSimulation:
    def __init__(self, model_path="models/ppo_7011_20250819.pt", lr=1e-3, gamma=0.99):
        self.model_path = model_path
        self.gamma = gamma

        self.model = ActorCritic()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 状態管理
        self.position = 0
        self.entry_price = None
        self.results = []

        # 正規化用
        self.price_window = []
        self.prev_volume = None  # 出来高差分用

        # 学習バッファ
        self.buffer = []

    def _preprocess(self, price, volume):
        # --- price: rolling 標準化 ---
        self.price_window.append(price)
        if len(self.price_window) > 100:
            self.price_window.pop(0)
        p_mean = np.mean(self.price_window)
        p_std = np.std(self.price_window) if np.std(self.price_window) > 0 else 1
        norm_price = (price - p_mean) / p_std

        # --- volume: 差分に変換して log1p ---
        if self.prev_volume is None:
            delta_volume = 0
        else:
            delta_volume = max(0, volume - self.prev_volume)
        self.prev_volume = volume
        norm_volume = np.log1p(delta_volume)

        return np.array([norm_price, norm_volume, self.position], dtype=np.float32)

    def add(self, ts, price, volume, force_close=False):
        state = self._preprocess(price, volume)
        state_tensor = torch.tensor(state).unsqueeze(0)

        with torch.no_grad():
            probs, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        reward = 0.0
        action_str = "HOLD"

        if self.position == 0:
            if action == 1:
                self.position = 1
                self.entry_price = price
                action_str = "BUY"
            elif action == 2:
                self.position = -1
                self.entry_price = price
                action_str = "SELL"
        else:
            if (self.position == 1 and action == 2) or force_close:
                reward = (price - self.entry_price) * 100
                self.position = 0
                action_str = "SELL_CLOSE" if not force_close else "FORCE_CLOSE"
            elif (self.position == -1 and action == 1) or force_close:
                reward = (self.entry_price - price) * 100
                self.position = 0
                action_str = "BUY_CLOSE" if not force_close else "FORCE_CLOSE"

        self.buffer.append((state, action, reward, value.item()))
        self.results.append(
            {"Time": ts, "Price": price, "Action": action_str, "Reward": reward}
        )
        return action_str

    def finalize(self):
        self._train()
        torch.save(self.model.state_dict(), self.model_path)

        df_result = pd.DataFrame(self.results)

        # --- 状態リセット ---
        self.results = []
        self.buffer = []
        self.position = 0
        self.entry_price = None
        self.price_window = []
        self.prev_volume = None

        return df_result

    def _train(self):
        if not self.buffer:
            return
        states, actions, rewards, values = zip(*self.buffer)

        # --- Warning 解消: list→np.array→tensor ---
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - values

        probs, vals = self.model(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(vals.squeeze(), returns)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
