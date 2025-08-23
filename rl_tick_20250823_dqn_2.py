import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim


# --- 強化学習用の軽量ネットワーク ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# --- Gym 環境定義 ---
class TradingEnv(gym.Env):
    """
    状態: [正規化株価変化, log差分出来高, ポジション情報]
    アクション: 0=ホールド, 1=新規買い, 2=新規売り, 3=返済
    報酬: 含み益 + 実現益
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super(TradingEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.current_price = None
        self.prev_price = None
        self.prev_volume = None
        self.position = 0  # 0=ノーポジ, 1=買い, -1=売り
        self.entry_price = 0
        self.done = False

    def reset(self, *, seed=None, options=None):
        self.current_price = None
        self.prev_price = None
        self.prev_volume = None
        self.position = 0
        self.entry_price = 0
        self.done = False
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action, price, volume, force_close=False):
        reward = 0.0
        profit = 0.0

        # 株価変化率
        price_change = 0.0 if self.prev_price is None else (price - self.prev_price) / self.prev_price
        # 出来高の差分を log1p
        vol_diff = 0.0 if self.prev_volume is None else np.log1p(volume - self.prev_volume)

        # アクション処理
        if action == 1 and self.position == 0:  # 新規買い
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # 新規売り
            self.position = -1
            self.entry_price = price
        elif (action == 3 and self.position != 0) or force_close:  # 返済
            if self.position == 1:  # 買い決済
                profit = (price - self.entry_price) * 100
            elif self.position == -1:  # 売り決済
                profit = (self.entry_price - price) * 100
            reward += profit
            self.position = 0
            self.entry_price = 0

        # 含み益を報酬に加える
        if self.position == 1:
            reward += (price - self.entry_price) * 100
        elif self.position == -1:
            reward += (self.entry_price - price) * 100

        self.prev_price = price
        self.prev_volume = volume

        obs = np.array([price_change, vol_diff, float(self.position)], dtype=np.float32)
        return obs, reward, self.done, False, {"profit": profit}


# --- シミュレーション管理クラス ---
class TradingSimulation:
    def __init__(self, model_path="models/policy.pth"):
        self.env = TradingEnv()
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.policy_net = QNetwork(input_dim, output_dim).to(self.device)

        if os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_path))
                print("既存モデルを読み込みました:", model_path)
            except Exception:
                print("既存モデルが無効のため再生成します")
                self.policy_net = QNetwork(input_dim, output_dim).to(self.device)
        else:
            print("新規モデルを生成します")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.results = pd.DataFrame(columns=["Time", "Price", "Action", "Profit", "Reward"])
        self.state, _ = self.env.reset()

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def add(self, ts, price, volume, force_close=False):
        action = self.select_action(self.state)

        next_state, reward, done, _, info = self.env.step(action, price, volume, force_close)
        profit = info["profit"]

        # Q学習更新
        state_t = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_t)
        next_q_values = self.policy_net(next_state_t)

        target = q_values.clone()
        target[0, action] = reward + 0.99 * next_q_values.max().item()

        loss = self.criterion(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state = next_state

        # 結果を記録
        self.results.loc[len(self.results)] = [ts, price, action, profit, reward]
        return action

    def finalize(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        df_out = self.results.copy()
        self.results = pd.DataFrame(columns=["Time", "Price", "Action", "Profit", "Reward"])
        self.state, _ = self.env.reset()
        return df_out
