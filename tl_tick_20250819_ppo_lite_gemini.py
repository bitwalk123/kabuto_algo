import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# PPO-liteモデルの定義
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actorネットワーク: 状態 -> 行動確率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Criticネットワーク: 状態 -> 価値
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)


class TradingSimulation:
    def __init__(self, model_path="ppo_model.pth"):
        self.model_path = model_path
        self.state_dim = 2  # price, volume
        self.action_dim = 3  # buy, hold, sell
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルの読み込みまたは新規作成
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)

        try:
            self.actor_critic.load_state_dict(torch.load(self.model_path))
            print(f"モデルを読み込みました: {self.model_path}")
        except FileNotFoundError:
            print(f"新しいモデルを作成します: {self.model_path}")

        self.reset_simulation()

    def reset_simulation(self):
        # シミュレーション状態のリセット
        self.results = []
        self.position = 0  # 1:買い, -1:売り, 0:なし
        self.entry_price = 0
        self.last_ts = 0
        self.last_volume = 0
        self.memory = []  # 強化学習の経験バッファ (state, action, reward, next_state, done)
        self.is_finalized = False

    def _get_state(self, price, volume):
        # 出来高を対数変換
        log_volume = np.log(1 + (volume - self.last_volume)) if volume > self.last_volume else 0

        # 株価と出来高を正規化 (ここでは簡易的なMin-Maxスケーリングを想定)
        price_norm = (price - 1000) / 9000
        volume_norm = log_volume / 1000  # 適切な最大値で正規化

        # 状態ベクトル
        return torch.tensor([price_norm, volume_norm], dtype=torch.float).to(self.device)

    def add(self, ts, price, volume, force_close=False):
        if self.is_finalized:
            self.reset_simulation()

        state = self._get_state(price, volume)

        action_probs, value = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        # 行動の実行と報酬の計算
        reward = 0
        action_name = "HOLD"

        if self.position == 0:
            if action.item() == 0:  # 買い
                self.position = 1
                self.entry_price = price
                action_name = "BUY"
            elif action.item() == 2:  # 売り
                self.position = -1
                self.entry_price = price
                action_name = "SELL"
        elif force_close:
            if self.position == 1:  # 買いポジションの強制返済
                reward = (price - self.entry_price) * 100
                action_name = "CLOSE_BUY"
            elif self.position == -1:  # 売りポジションの強制返済
                reward = (self.entry_price - price) * 100
                action_name = "CLOSE_SELL"
            self.position = 0

        elif self.position == 1 and action.item() == 2:  # 買いポジションの返済
            reward = (price - self.entry_price) * 100
            self.position = 0
            action_name = "CLOSE_BUY"
        elif self.position == -1 and action.item() == 0:  # 売りポジションの返済
            reward = (self.entry_price - price) * 100
            self.position = 0
            action_name = "CLOSE_SELL"

        # 経験の記録 (ここでは簡易的に状態、行動、報酬のみ記録)
        self.memory.append((state.detach(), action.detach(), reward, value.detach()))

        # 結果の記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": action_name,
            "Reward": reward
        })

        self.last_ts = ts
        self.last_volume = volume

        return action_name

    def finalize(self):
        # 強化学習の学習プロセス
        if not self.memory:
            return pd.DataFrame(self.results)

        states, actions, rewards, values = zip(*self.memory)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        values = torch.cat(values).squeeze().to(self.device)

        # PPOの学習ロジック (簡易版)
        returns = self._compute_returns(rewards)
        advantages = returns - values

        # 損失の計算とバックプロパゲーション
        actor_loss = -(Categorical(self.actor_critic(torch.stack(states))[0]).log_prob(
            torch.stack(actions)) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # モデルの保存
        torch.save(self.actor_critic.state_dict(), self.model_path)

        df_result = pd.DataFrame(self.results)
        self.is_finalized = True
        return df_result

    def _compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float).to(self.device)
