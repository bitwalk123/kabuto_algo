import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque


# ======================
# 環境定義
# ======================
class TradingEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_balance=1_000_000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # 行動空間: 0 = ホールド, 1 = 買い, 2 = 売り
        self.action_space = spaces.Discrete(3)

        # 状態: 価格・出来高・特徴量 + ポジション
        n_features = 6  # price, volume, MA, volatility, RSI, position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(window_size, n_features),
                                            dtype=np.float32)

        # 内部状態
        self.balance = None
        self.position = None
        self.entry_price = None
        self.current_step = None
        self.done = None

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        self.done = False
        return self._get_state()

    def _get_state(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        state = np.column_stack([
            window["Price"].values,
            window["Volume"].values,
            window["MA"].values,
            window["Volatility"].values,
            window["RSI"].values,
            np.full(self.window_size, self.position)
        ])
        return state.astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.current_step]["Price"]

        reward = 0
        if action == 1:  # 買い
            if self.position == 0:
                self.position = 1
                self.entry_price = price
            elif self.position == -1:
                reward = self.entry_price - price
                self.balance += reward
                self.position = 0
        elif action == 2:  # 売り
            if self.position == 0:
                self.position = -1
                self.entry_price = price
            elif self.position == 1:
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0
        else:  # ホールド
            if self.position == 1:
                reward = price - self.entry_price
            elif self.position == -1:
                reward = self.entry_price - price

        # 報酬をバランスでスケーリング
        reward = reward / 100.0
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")


# ======================
# 強化学習モデル (DQN)
# ======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# リプレイバッファ
# ======================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)


# ======================
# 学習関数
# ======================
def train_dqn(env, num_episodes=50, batch_size=64, gamma=0.99, lr=1e-3,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)

    epsilon = epsilon_start
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state).unsqueeze(0).float())
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).long()
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        target_net.load_state_dict(policy_net.state_dict())
        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

    return policy_net, rewards_history


# ======================
# 特徴量生成
# ======================
def add_features(df):
    df["MA"] = df["Price"].rolling(window=10).mean()
    df["Volatility"] = df["Price"].rolling(window=10).std()
    df["RSI"] = compute_rsi(df["Price"], 14)
    df = df.fillna(0)
    return df


def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ======================
# 実行サンプル
# ======================
if __name__ == "__main__":
    # サンプルデータ（ダミー）
    np.random.seed(0)
    prices = np.cumsum(np.random.randn(200)) + 100
    volumes = np.random.randint(100, 1000, size=200)
    df = pd.DataFrame({"Price": prices, "Volume": volumes})
    df = add_features(df)

    env = TradingEnv(df, window_size=30)
    model, history = train_dqn(env, num_episodes=20)
