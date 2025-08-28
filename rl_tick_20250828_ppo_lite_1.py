import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==== PPO-lite Actor-Critic ====
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, model_path="models/policy.pth", lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.model_path = model_path
        self.device = torch.device("cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if os.path.exists(model_path):
            try:
                self.policy.load_state_dict(torch.load(model_path))
                print("既存モデルを読み込みました:", model_path)
            except Exception:
                print("既存モデルが無効。新しいモデルを生成して上書きします。")
                torch.save(self.policy.state_dict(), model_path)
        else:
            print("モデルが存在しません。新規生成します。")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.policy.state_dict(), model_path)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs, states, next_states, dones):
        returns, G = [], 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)

        _, values = self.policy(states)
        values = values.squeeze()
        advantages = returns - values.detach()

        # actor loss
        ratios = torch.exp(log_probs - log_probs.detach())
        actor_loss = -(torch.min(ratios * advantages,
                                 torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages)).mean()

        # critic loss
        critic_loss = nn.MSELoss()(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.save(self.policy.state_dict(), self.model_path)


# ==== Trading Simulation ====
class TradingSimulation:
    def __init__(self):
        self.results = []
        self.position = None   # ("buy" or "sell")
        self.entry_price = None
        self.shares = 100
        self.slippage = 1.0

        # agent
        self.state_dim = 6  # [price, log_volume, MA, Vol, RSI, position_flag]
        self.action_dim = 4  # buy, sell, close, hold
        self.agent = PPOAgent(self.state_dim, self.action_dim)

        # buffers
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones = [], []

        # for feature calc
        self.df_buffer = pd.DataFrame(columns=["Price", "Volume"])
        self.last_volume = None

    def _features(self, ts, price, volume):
        # 差分出来高
        delta_vol = 0 if self.last_volume is None else max(volume - self.last_volume, 0)
        self.last_volume = volume

        self.df_buffer.loc[ts] = [price, volume]
        df = self.df_buffer

        ma = df["Price"].rolling(60).mean().iloc[-1] if len(df) >= 60 else price
        vol = df["Price"].rolling(60).std().iloc[-1] if len(df) >= 60 else 0

        # RSI
        if len(df) >= 60:
            delta = df["Price"].diff()
            gain = delta.where(delta > 0, 0).rolling(60).mean()
            loss = -delta.where(delta < 0, 0).rolling(60).mean()
            rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-6)
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        log_vol = np.log1p(delta_vol)
        pos_flag = 0 if self.position is None else (1 if self.position == "buy" else -1)

        return np.array([price, log_vol, ma, vol, rsi, pos_flag], dtype=np.float32)

    def add(self, ts, price, volume, force_close=False):
        state = self._features(ts, price, volume)
        action, log_prob = self.agent.select_action(state)

        reward, profit, action_name = 0, 0, "ホールド"

        if self.position is None:  # no position
            if action == 0:  # buy
                self.position = "buy"
                self.entry_price = price + self.slippage
                action_name = "新規買い"
            elif action == 1:  # sell
                self.position = "sell"
                self.entry_price = price - self.slippage
                action_name = "新規売り"
        else:
            if action == 2 or force_close:  # close
                if self.position == "buy":
                    exit_price = price - self.slippage
                    profit = (exit_price - self.entry_price) * self.shares
                    action_name = "返済売り" if not force_close else "返済売り（強制）"
                elif self.position == "sell":
                    exit_price = price + self.slippage
                    profit = (self.entry_price - exit_price) * self.shares
                    action_name = "返済買い" if not force_close else "返済買い（強制）"

                reward = profit
                self.position, self.entry_price = None, None

        # 含み益も一部を報酬に加える
        if self.position is not None:
            if self.position == "buy":
                unrealized = (price - self.entry_price) * self.shares
            else:
                unrealized = (self.entry_price - price) * self.shares
            reward += 0.01 * unrealized

        # 記録
        self.results.append({"Time": ts, "Price": price, "Action": action_name, "Profit": profit})

        # バッファに保存
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(force_close)

        return action_name

    def finalize(self):
        if len(self.results) == 0:
            return pd.DataFrame()

        # PPO update
        self.agent.update(self.rewards, self.log_probs, self.states, self.states, self.dones)

        df_result = pd.DataFrame(self.results)
        self.results = []
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones = [], []
        self.df_buffer = pd.DataFrame(columns=["Price", "Volume"])
        self.last_volume = None
        return df_result
