import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque


# 簡易 PPO-lite Actor-Critic モデル
class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        # 出力は [0: HOLD, 1: BUY, 2: SELL]
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class TradingSimulation:
    def __init__(self, model_path="models/trade_model.pt", device="cpu"):
        self.device = device
        self.model_path = model_path
        self.state_dim = 3  # [price, Δvolume(log1p), position]
        self.action_dim = 3  # HOLD / BUY / SELL
        self.hidden_dim = 64

        # モデル読み込み or 新規作成
        self.model = ActorCritic(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("既存モデルを読み込みました:", self.model_path)
            except Exception as e:
                print("既存モデルの読み込みに失敗、新規作成:", e)
                self.model = ActorCritic(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        else:
            print("新規モデルを作成")

        # シミュレーションの状態
        self.position = 0   # +100株 (買い保有), -100株 (売り保有), 0 (ノーポジ)
        self.entry_price = None
        self.last_volume = None
        self.results = []
        self.memory = []  # PPO-lite 用タスクバッファ

    def _preprocess(self, price, volume):
        """価格・出来高の特徴量整形"""
        if self.last_volume is None:
            delta_volume = 0.0
        else:
            delta_volume = volume - self.last_volume
        self.last_volume = volume

        log_delta_vol = np.log1p(delta_volume)
        state = np.array([price / 10000.0, log_delta_vol / 10.0, self.position / 100.0], dtype=np.float32)
        return state

    def _select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, dist.log_prob(torch.tensor(action)).unsqueeze(0), value

    def _calc_reward(self, price, force_close=False):
        """建玉と価格から報酬を計算"""
        reward = 0.0
        closed = False

        if self.position != 0:
            profit = (price - self.entry_price) * self.position
            # position=+100株 → profit=(price-entry_price)*100
            # position=-100株 → profit=(entry_price-price)*100
            profit *= 1  # 100株単位、そのまま扱う

            # 強制返済 or 含み益が一定以上
            if force_close or (profit >= 200):
                if profit >= 1000:
                    reward = profit * 2.0
                elif profit >= 500:
                    reward = profit * 1.5
                elif profit >= 200:
                    reward = profit * 1.0
                else:
                    reward = profit * 0.1

                closed = True
                self.position = 0
                self.entry_price = None

                return reward, closed, profit

            # 含み益・含み損に応じて微妙な補正
            if profit > 0:
                reward = 0.01
            elif profit < 0:
                reward = -0.01

        return reward, closed, 0.0

    def add(self, ts, price, volume, force_close=False):
        """ティックデータを 1 行受け取る"""
        state = self._preprocess(price, volume)
        action, log_prob, value = self._select_action(state)

        reward, closed, realized_profit = self._calc_reward(price, force_close)

        # 実際の建玉操作
        if self.position == 0 and not force_close:
            if action == 1:  # BUY
                self.position = 100
                self.entry_price = price
                trade_action = "BUY"
            elif action == 2:  # SELL
                self.position = -100
                self.entry_price = price
                trade_action = "SELL"
            else:
                trade_action = "HOLD"
        else:
            trade_action = "HOLD"
            if closed:
                trade_action = "CLOSE"

        # 結果を記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": trade_action,
            "Profit": realized_profit
        })

        # メモリに蓄積（簡易 PPO-lite）
        self.memory.append((state, action, reward, log_prob, value))

        return trade_action

    def finalize(self):
        """最終まとめ → 学習 → DataFrameで返す"""
        df = pd.DataFrame(self.results)

        # 簡易 PPO-lite 学習
        if len(self.memory) > 1:
            states, actions, rewards, log_probs, values = zip(*self.memory)
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            old_log_probs = torch.cat(log_probs).to(self.device)
            values = torch.cat(values).squeeze(-1)

            # 収益合計を advantage として単純化
            returns = rewards.cumsum(dim=0)
            advantages = returns - values.detach()

            new_probs, new_values = self.model(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.save(self.model.state_dict(), self.model_path)

        # 結果リセット（次 epoch の準備）
        self.results = []
        self.memory = []
        self.position = 0
        self.entry_price = None
        self.last_volume = None

        return df
