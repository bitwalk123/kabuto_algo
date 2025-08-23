import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


# --- PPO-lite Actor-Critic モデル ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value = self.critic_head(x)
        return action_probs, state_value


# --- TradingSimulation クラス ---
class TradingSimulation:
    def __init__(self, model_path="ppo_model.pth", profit_threshold=200, loss_threshold=-500):
        # シミュレーション設定
        self.model_path = model_path
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.tick_count = 0
        self.prev_volume = None
        self.position = None
        self.entry_price = None
        self.entry_ts = None
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.profit_log = []

        # 強化学習モデルのセットアップ
        self.state_dim = 2  # [log1p(delta_volume), price_normalized]
        self.action_dim = 3  # 0: 何もしない, 1: 買い, 2: 売り

        # モデルの読み込みまたは新規作成
        self.policy_net = ActorCritic(self.state_dim, self.action_dim)
        if os.path.exists(self.model_path):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path))
                self.policy_net.eval()
                print("既存の学習モデルを読み込みました。")
            except RuntimeError:
                print("既存モデルの形式が無効です。新しいモデルを生成します。")
                self.policy_net = ActorCritic(self.state_dim, self.action_dim)
                self._save_model()
        else:
            print("学習モデルが存在しないため、新しいモデルを生成します。")
            self.policy_net = ActorCritic(self.state_dim, self.action_dim)
            self._save_model()

        # Optimizerの設定
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def add(self, ts, price, volume, force_close=False):
        self.tick_count += 1

        # 出来高の加工: 累積出来高 -> 差分出来高 -> log1p変換
        if self.prev_volume is not None:
            delta_volume = volume - self.prev_volume
        else:
            delta_volume = 0
        self.prev_volume = volume

        # 状態の生成: 出来高情報をlog1pで圧縮し、株価は正規化
        log_delta_volume = np.log1p(delta_volume)
        normalized_price = (price - 1000) / 9000  # 株価を0-1の範囲に正規化 (1000-10000のオーダーを想定)
        state = torch.tensor([log_delta_volume, normalized_price], dtype=torch.float32)

        # モデルに状態を入力し、行動を決定
        action_probs, _ = self.policy_net(state)
        action = torch.multinomial(action_probs, 1).item()

        reward = 0
        current_profit = 0

        # ポジションを保有している場合
        if self.position is not None:
            if self.position == "BUY":
                current_profit = (price - self.entry_price) * 100
            elif self.position == "SELL":
                current_profit = (self.entry_price - price) * 100

            # --- 報酬設計の変更箇所 (提案2) ---
            # 利益確定の報酬を段階的に設定
            if current_profit >= 1000:
                reward = current_profit * 2  # 1000円以上はより大きな報酬
                # 利益確定後の処理
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": f"返済 ({self.position})",
                    "Profit": current_profit
                })
                self.position = None
                self.entry_price = None
                self.entry_ts = None
                action = 0
            elif current_profit >= 500:
                reward = current_profit * 1.5  # 500円以上は少し多めに
                # 利益確定後の処理
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": f"返済 ({self.position})",
                    "Profit": current_profit
                })
                self.position = None
                self.entry_price = None
                self.entry_ts = None
                action = 0
            elif current_profit >= self.profit_threshold:
                reward = current_profit * 1.0  # 200円以上で基本的な報酬
                # 利益確定後の処理
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": f"返済 ({self.position})",
                    "Profit": current_profit
                })
                self.position = None
                self.entry_price = None
                self.entry_ts = None
                action = 0
            # 損切り
            elif current_profit <= self.loss_threshold:
                reward = current_profit * 2  # 損失額の2倍をペナルティとする
                # 損切り後の処理
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": f"損切り ({self.position})",
                    "Profit": current_profit
                })
                self.position = None
                self.entry_price = None
                self.entry_ts = None
                action = 0
            # 強制返済
            elif force_close:
                reward = current_profit
                # 強制返済後の処理
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": f"強制返済 ({self.position})",
                    "Profit": current_profit
                })
                self.position = None
                self.entry_price = None
                self.entry_ts = None
                action = 0
            # 利益確定でも損切りでもない中間状態の報酬
            else:
                if current_profit > 0:
                    reward = 1  # 利益が出ている間はわずかなボーナス
                else:
                    reward = -1  # 含み損を抱えている間はわずかなペナルティ

        # ポジションを保有していない場合、新規建玉を検討
        else:
            if action == 1:  # 買い
                self.position = "BUY"
                self.entry_price = price
                self.entry_ts = ts
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": "新規買い",
                    "Profit": np.nan
                })
            elif action == 2:  # 売り
                self.position = "SELL"
                self.entry_price = price
                self.entry_ts = ts
                self.profit_log.append({
                    "Time": ts,
                    "Price": price,
                    "Action": "新規売り",
                    "Profit": np.nan
                })

        # 経験の記録 (学習のため)
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # 最後に、モデルの学習と保存
        if force_close:
            self._update_model()
            self._save_model()

        return action

    def finalize(self):
        # 最終結果のDataFrameを作成
        df_results = pd.DataFrame(self.profit_log)
        # 履歴をリセット
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.profit_log.clear()
        self.position = None
        self.entry_price = None
        self.entry_ts = None
        self.prev_volume = None

        return df_results

    def _update_model(self):
        if not self.state_history:
            return

        # PPO-liteの学習ロジック
        states = torch.stack(self.state_history)
        actions = torch.tensor(self.action_history, dtype=torch.long)
        rewards = torch.tensor(self.reward_history, dtype=torch.float32)

        # 割引報酬の計算 (GAEなどを使わずシンプルに)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Actor-Critic Lossの計算
        action_probs, state_values = self.policy_net(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1))).squeeze(-1)
        advantages = returns - state_values.squeeze(-1)

        # PPO Clipping objective (軽量化のためシンプルに)
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(state_values.squeeze(-1), returns)
        loss = policy_loss + 0.5 * value_loss

        # バックプロパゲーションとパラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
