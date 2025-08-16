import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import numpy as np
from collections import deque
import os


# --- A. DQNエージェントクラス ---
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Qネットワークの構築
        self.fc1 = nn.Linear(self.state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.action_size)

        # 最適化手法の定義
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.forward(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor).detach())

            target_f = self.forward(state_tensor)
            target_f[0][action] = target

            output = self.forward(state_tensor)
            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()  # 評価モードに設定


# --- B. シミュレーション実行スクリプト ---

# ファイルの読み込み
file_path = "tick_20250602_7011.csv"
df = pd.read_csv(file_path)

# エージェントと環境の定義
STATE_SIZE = 10  # 過去10ティックの価格変動を状態とする
ACTION_SIZE = 3  # 0:Buy, 1:Sell, 2:Hold
BATCH_SIZE = 32
TRADE_UNIT = 100


def get_state(data, t, n_ticks):
    """過去n_ticksの価格変動を状態として取得する"""
    if t < n_ticks:
        return np.zeros(n_ticks)
    state = data['Price'].iloc[t - n_ticks: t].values
    return state - state[0]  # 価格変動を計算


def run_simulation(data, agent, episodes=1):
    results = []

    for e in range(episodes):
        print(f"--- Episode {e + 1}/{episodes} ---")
        position = "None"
        entry_price = 0

        for t in range(1, len(data)):
            state = get_state(data, t, STATE_SIZE)

            # ポジションを保有している場合は、決済アクションを追加
            current_price = data['Price'].iloc[t]
            action_map = {0: "Buy", 1: "Sell", 2: "Hold"}

            if position == "None":
                action = agent.act(state)
                action_name = action_map[action]

                if action == 0:  # Buy
                    position = "Buy"
                    entry_price = current_price
                    reward = 0
                    results.append([data['Time'].iloc[t], current_price, "Buy", 0, "Buy", entry_price])
                    #print(f"Time: {data['Time'].iloc[t]}, Price: {current_price}, Action: Buy")
                elif action == 1:  # Sell
                    position = "Sell"
                    entry_price = current_price
                    reward = 0
                    results.append([data['Time'].iloc[t], current_price, "Sell", 0, "Sell", entry_price])
                    #print(f"Time: {data['Time'].iloc[t]}, Price: {current_price}, Action: Sell")
                else:  # Hold
                    results.append([data['Time'].iloc[t], current_price, "Hold", 0, "None", 0])

            else:  # Position is held
                action = agent.act(state)
                action_name = action_map[action]

                if position == "Buy" and action == 1:  # Close Buy
                    profit = (current_price - entry_price) * TRADE_UNIT
                    reward = profit
                    position = "None"
                    results.append([data['Time'].iloc[t], current_price, "Close_Buy", profit, "None", 0])
                    #print(f"Time: {data['Time'].iloc[t]}, Price: {current_price}, Action: Close_Buy, Profit: {profit}")

                elif position == "Sell" and action == 0:  # Close Sell
                    profit = (entry_price - current_price) * TRADE_UNIT
                    reward = profit
                    position = "None"
                    results.append([data['Time'].iloc[t], current_price, "Close_Sell", profit, "None", 0])
                    #print(f"Time: {data['Time'].iloc[t]}, Price: {current_price}, Action: Close_Sell, Profit: {profit}")
                else:  # Hold
                    reward = 0
                    results.append([data['Time'].iloc[t], current_price, "Hold", 0, position, entry_price])

            # 学習
            next_state = get_state(data, t + 1, STATE_SIZE) if t + 1 < len(data) else np.zeros(STATE_SIZE)
            done = True if t == len(data) - 1 else False

            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

    return pd.DataFrame(results, columns=['Time', 'Price', '売買アクション', '報酬額', 'ポジション', '建玉価格'])


# --- メインの実行部分 ---
if __name__ == "__main__":
    # モデルのファイルパスを定義
    model_file_path = "trading_model.pth"

    # エージェントのインスタンスを作成
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    # ファイルが存在する場合、モデルを読み込む
    if os.path.exists(model_file_path):
        print(f"Loading pre-trained model from {model_file_path}")
        agent.load_model(model_file_path)
        # 学習済みモデルを読み込んだ後は、探索率を低く設定することが一般的
        agent.epsilon = 0.1  # 例として0.1に設定
    else:
        print(f"Model file {model_file_path} not found. Starting with a new model.")

    # シミュレーションと学習の実行
    trade_results_df = run_simulation(df, agent, episodes=10)  # 10エピソードで学習

    # 学習済みモデルの保存
    agent.save_model(model_file_path)

    # 結果の出力
    trade_results_df.to_csv("trade_results.csv", index=False)
    print("Simulation complete. Results saved to trade_results.csv")