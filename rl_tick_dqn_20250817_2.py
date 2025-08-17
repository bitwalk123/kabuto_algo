import os
import random
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ====================
# DQN ネットワーク定義
# ====================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ====================
# 環境クラス
# ====================
class TradingEnv:
    def __init__(self, df, unit=100):
        self.df = df.reset_index(drop=True)
        self.unit = unit
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0   # 0: ノーポジ, 1: 買い, -1: 売り
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        price = self.df.loc[self.current_step, "Price"]
        return np.array([price, self.position], dtype=np.float32)

    def step(self, action):
        """
        action: 0=ホールド, 1=買い, 2=売り, 3=決済
        """
        price = self.df.loc[self.current_step, "Price"]
        reward = 0
        action_str = "ホールド"

        if action == 1 and self.position == 0:  # 新規買い
            self.position = 1
            self.entry_price = price
            action_str = "新規買い"
        elif action == 2 and self.position == 0:  # 新規売り
            self.position = -1
            self.entry_price = price
            action_str = "新規売り"
        elif action == 3 and self.position != 0:  # 決済
            if self.position == 1:
                reward = (price - self.entry_price) * self.unit
            elif self.position == -1:
                reward = (self.entry_price - price) * self.unit
            self.position = 0
            self.entry_price = 0
            action_str = "決済"

        # ステップ進める
        self.current_step += 1

        # 最終行なら強制決済処理して終了
        if self.current_step >= len(self.df):
            if self.position != 0:
                if self.position == 1:
                    reward += (price - self.entry_price) * self.unit
                elif self.position == -1:
                    reward += (self.entry_price - price) * self.unit
                action_str = "強制決済"
                self.position = 0
                self.entry_price = 0
            self.done = True
            return np.array([price, 0], dtype=np.float32), reward, self.done, action_str

        # 通常の遷移
        next_state = self._get_state()
        return next_state, reward, self.done, action_str


# ====================
# DQN エージェント
# ====================
class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path="dqn_model.pth"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # 既存モデル読み込み
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("既存モデルを読み込みました。")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).clone().detach()
            target_f[action] = target

            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)


# ====================
# シミュレーション実行
# ====================
def run_simulation(csv_path, epochs=3, model_path="dqn_model.pth"):
    df = pd.read_csv(csv_path)
    env = TradingEnv(df)
    agent = DQNAgent(state_dim=2, action_dim=4, model_path=model_path)

    results = []
    for epoch in range(epochs):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, action_str = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            # インデックスの範囲を保証
            idx = min(env.current_step - 1, len(df) - 1)
            results.append({
                "Epoch": epoch,
                "Time": df.loc[idx, "Time"],
                "Price": df.loc[idx, "Price"],
                "売買アクション": action_str,
                "報酬額": reward
            })

            state = next_state

        agent.save()
        print(f"Epoch {epoch} 終了")

    results_df = pd.DataFrame(results)
    results_df.to_csv("trade_results.csv", index=False)
    print("trade_results.csv に出力しました。")


# ====================
# 実行例
# ====================
if __name__ == "__main__":
    run_simulation("data/tick_20250602_7011.csv", epochs=3)
