import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import pandas as pd


# ------------------------------
# PPO-lite Actor-Critic ネットワーク
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, action_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)  # 行動: 0=ホールド, 1=買い, 2=売り
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


# ------------------------------
# Trading Simulation
# ------------------------------
class TradingSimulation:
    def __init__(self, model_path="ppo_model.pth", lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.gamma = gamma

        # PPO-lite ネットワーク
        self.model = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 学習履歴
        self.memory = []

        # トレード状態
        self.position = 0  # 0=ノーポジ, 1=買い, -1=売り
        self.entry_price = None
        self.results = []  # 取引ログ

        # モデル再利用
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("学習済みモデルを読み込みました:", model_path)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def add(self, ts, price, volume, force_close=False):
        # 状態ベクトル: [価格, 出来高, ポジション]
        state = [price, volume, float(self.position)]

        # 行動選択
        action, log_prob, value = self.select_action(state)

        reward = 0
        trade_action = "ホールド"

        # 売買ロジック
        if force_close and self.position != 0:
            # 強制返済
            reward = (price - self.entry_price) * 100 * self.position
            trade_action = "強制返済"
            self.position = 0
            self.entry_price = None

        elif self.position == 0:
            if action == 1:  # 新規買い
                self.position = 1
                self.entry_price = price
                trade_action = "新規買い"
            elif action == 2:  # 新規売り
                self.position = -1
                self.entry_price = price
                trade_action = "新規売り"

        elif self.position == 1:  # 買い持ち
            if action == 2:  # 売却（返済）
                reward = (price - self.entry_price) * 100
                trade_action = "返済売り"
                self.position = 0
                self.entry_price = None

        elif self.position == -1:  # 売り持ち
            if action == 1:  # 買い戻し（返済）
                reward = (self.entry_price - price) * 100
                trade_action = "返済買い"
                self.position = 0
                self.entry_price = None

        # メモリに保存（学習用）
        self.memory.append((state, action, reward, log_prob, value))

        # ログ記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "売買アクション": trade_action,
            "報酬額": reward
        })

        return trade_action

    def finalize(self, output_file="trade_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        # PPO-lite 学習更新
        self.update_model()

        # 保存
        torch.save(self.model.state_dict(), self.model_path)
        print("モデルを保存しました:", self.model_path)

        # --- ここでリセット ---
        results_copy = self.results
        self.results = []

        return pd.DataFrame(results_copy)

    def update_model(self):
        if not self.memory:
            return

        states, actions, rewards, log_probs, values = zip(*self.memory)

        # Tensor 変換
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        values = torch.cat(values).squeeze(-1)

        # 割引累積報酬
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Advantage
        advantages = returns - values.detach()

        # PPO オブジェクティブ
        logits, new_values = self.model(states)
        dist = Categorical(F.softmax(logits, dim=-1))
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values.squeeze(-1), returns)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []  # メモリクリア
        print("学習更新完了: Loss =", loss.item())
