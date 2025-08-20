import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ハイパーパラメータ
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64
ENTROPY_BETA = 0.01

# 定数
LOT_SIZE = 100
MODEL_PATH = "ppo_trading_model.pth"

# 状態空間の定義
# [正規化された株価, 正規化された差分出来高, ポジション状態, 含み損益]
STATE_DIM = 4
ACTION_DIM = 3  # 0: 何もしない, 1: 買い, 2: 売り


# モデルの定義
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.memory = []

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def get_value(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.critic(state).item()

    def add_experience(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def update(self):
        if not self.memory:
            return

        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        values = self.critic(states)
        next_values = self.critic(next_states)

        # GAE (Generalized Advantage Estimation) を利用した利得の計算
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + GAMMA * GAE_LAMBDA * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]

        returns = advantages + values

        for _ in range(PPO_EPOCHS):
            batch_indices = np.random.choice(len(self.memory), PPO_BATCH_SIZE, replace=True)
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Actor の更新
            new_log_probs = torch.log(self.actor(batch_states).gather(1, batch_actions))
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy項を追加して探索を促進
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=1).mean()
            actor_loss = actor_loss - ENTROPY_BETA * entropy

            # Critic の更新
            critic_loss = nn.functional.mse_loss(self.critic(batch_states), batch_returns)

            # 2つの損失を合計して一度に backward を実行する
            total_loss = actor_loss + critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory.clear()


class TradingSimulation:
    def __init__(self, lot_size=LOT_SIZE):
        self.lot_size = lot_size
        self.ppo = PPO(STATE_DIM, ACTION_DIM)
        self.position = 0  # 1: 買建, -1: 売建, 0: ポジションなし
        self.entry_price = 0.0
        self.last_price = 0.0
        self.last_volume = 0.0
        self.last_ts = 0.0
        self.results = []
        self.initial_tick_data = False

        # prev_stateなどの変数を初期化
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None

        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH)
                self.ppo.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.ppo.critic.load_state_dict(checkpoint['critic_state_dict'])
                logging.info(f"既存のモデルを読み込みました: {MODEL_PATH}")
            except (IOError, KeyError, RuntimeError) as e:
                logging.error(f"モデルの読み込みに失敗しました。新しいモデルを生成します: {e}")
                self.create_new_model()
        else:
            self.create_new_model()

    def create_new_model(self):
        self.ppo = PPO(STATE_DIM, ACTION_DIM)
        logging.info("新しいモデルを生成しました")

    def _get_state(self, price, volume):
        # 出来高の差分を計算し、np.log1pで正規化
        diff_volume = volume - self.last_volume
        normalized_volume = np.log1p(diff_volume) if diff_volume >= 0 else 0

        # 株価の正規化（例: 過去の平均値や標準偏差を使用する。ここでは単純な比率を使用）
        # 実際にはより洗練された方法が必要です
        normalized_price = price / 10000.0

        # 含み損益の計算と正規化
        if self.position == 1:
            unrealized_profit = (price - self.entry_price) * self.lot_size
        elif self.position == -1:
            unrealized_profit = (self.entry_price - price) * self.lot_size
        else:
            unrealized_profit = 0

        normalized_unrealized_profit = unrealized_profit / 10000.0  # 適切な値に正規化

        # 状態の構築
        state = [
            normalized_price,
            normalized_volume,
            self.position,
            normalized_unrealized_profit
        ]
        return state

    def add(self, ts, price, volume, force_close=False):
        """
        別プログラムからティックデータを受け取り、売買アクションを決定する
        :param ts: タイムスタンプ
        :param price: 株価
        :param volume: 累積出来高
        :param force_close: 強制返済フラグ
        :return: 売買アクション (0: HOLD, 1: BUY, 2: SELL)
        """
        # 初回データ処理
        if not self.initial_tick_data:
            self.last_price = price
            self.last_volume = volume
            self.last_ts = ts
            self.initial_tick_data = True

            # 最初の状態とアクションを初期化
            state = self._get_state(price, volume)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = self.ppo.get_action(state)

            self.prev_state = state_tensor.detach().squeeze().numpy()
            self.prev_action = action
            self.prev_log_prob = log_prob

            # 最初のティックでは取引はしない
            self.results.append({
                "Time": ts,
                "Price": price,
                "Action": "HOLD",
                "Reward": 0,
                "Profit": 0
            })
            return 0  # 何もしない

        # 強制返済処理
        if force_close and self.position != 0:
            reward, profit = self._calculate_realized_profit(price)
            logging.info(f"最終時刻のため建玉を強制返済。実現損益: {profit}")

            # 強制返済時の学習データ追加
            state = self._get_state(price, volume)
            self.ppo.add_experience(self.prev_state, self.prev_action, self.prev_log_prob, reward, state, True)

            self.results.append({
                "Time": ts,
                "Price": price,
                "Action": "FORCE_CLOSE",
                "Reward": reward,
                "Profit": profit
            })
            self.position = 0
            self.entry_price = 0.0
            return 0

        # 状態の取得
        state = self._get_state(price, volume)

        # ポジションを保持している場合の報酬計算 (含み益)
        if self.position != 0:
            reward = self._calculate_unrealized_profit_reward(price)
        else:
            reward = 0

        # アクションの決定
        action, log_prob = self.ppo.get_action(state)

        # 売買アクションの実行
        realized_profit = 0
        executed_action_str = "HOLD"

        # ポジションを持っていない場合
        if self.position == 0:
            if action == 1:  # 買い
                self.position = 1
                self.entry_price = price
                executed_action_str = "BUY"
                logging.info(f"買い注文。時刻: {ts}, 株価: {price}, ポジション: {self.position}")
            elif action == 2:  # 売り
                self.position = -1
                self.entry_price = price
                executed_action_str = "SELL"
                logging.info(f"売り注文。時刻: {ts}, 株価: {price}, ポジション: {self.position}")

        # ポジションを持っている場合
        else:
            if (self.position == 1 and action == 2) or (self.position == -1 and action == 1):
                # ポジションの反対アクションで返済
                reward, realized_profit = self._calculate_realized_profit(price)

                # 建玉を返済したので、ポジションをクリア
                self.position = 0
                self.entry_price = 0.0
                executed_action_str = "CLOSE"
                logging.info(f"返済注文。時刻: {ts}, 株価: {price}, 実現損益: {realized_profit}")

            # `add`メソッドの戻り値として、実際の取引アクションを返す
            # ポジションを保有している間はHOLDを返す
            action = 0

        # PPOの学習データに追加
        # prev_stateをグラフから切り離すことで、次回のbackward()でエラーが出ないようにする
        self.ppo.add_experience(self.prev_state, self.prev_action, self.prev_log_prob, reward, state, False)

        # 状態の更新
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.prev_state = state_tensor.detach().squeeze().numpy()
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.last_price = price
        self.last_volume = volume
        self.last_ts = ts

        # 結果を記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": executed_action_str,
            "Reward": reward,
            "Profit": realized_profit
        })

        return action

    def _calculate_unrealized_profit_reward(self, current_price):
        """含み益に基づいた報酬を計算"""
        if self.position == 1:  # 買建
            unrealized_profit = (current_price - self.entry_price) * self.lot_size
        elif self.position == -1:  # 売建
            unrealized_profit = (self.entry_price - current_price) * self.lot_size
        else:
            return 0

        return unrealized_profit

    def _calculate_realized_profit(self, exit_price):
        """建玉返済時の実現損益と報酬を計算"""
        if self.position == 1:  # 買建玉の返済
            profit = (exit_price - self.entry_price) * self.lot_size
        elif self.position == -1:  # 売建玉の返済
            profit = (self.entry_price - exit_price) * self.lot_size
        else:
            profit = 0

        return profit, profit  # この例では実現損益と報酬を同値とする

    def finalize(self):
        """
        シミュレーションを終了し、結果のデータフレームを返す
        """
        self.ppo.update()  # Epochの終わりに学習
        df_results = pd.DataFrame(self.results)
        self.results = []  # 次のEpochのためにリセット
        self.initial_tick_data = False  # 次のEpochのためにリセット

        # モデルの保存
        torch.save({
            'actor_state_dict': self.ppo.actor.state_dict(),
            'critic_state_dict': self.ppo.critic.state_dict(),
        }, MODEL_PATH)
        logging.info(f"モデルを保存しました: {MODEL_PATH}")

        return df_results


if __name__ == "__main__":
    # この部分はユーザーが提供した「別プログラム」のコード例
    excel_file = os.path.join("data", "tick_20250819_7011.xlsx")
    epochs = 10

    # ダミーのティックデータファイルを作成
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(excel_file):
        dummy_data = {
            "Time": pd.to_datetime(pd.date_range("2025-08-19 09:00:00", periods=500, freq="s")),
            "Price": np.random.normal(5000, 10, 500).cumsum() + 5000,
            "Volume": np.arange(1, 501).cumsum() * 1000
        }
        pd.DataFrame(dummy_data).to_excel(excel_file, index=False)
        print(f"ダミーのExcelファイルを作成しました: {excel_file}")

    # 結果保存ディレクトリ
    if not os.path.exists("results"):
        os.makedirs("results")

    # ティックデータを読み込む
    df = pd.read_excel(excel_file)
    print("読み込んだティックデータ:")
    print(df.head())

    # シミュレータ・インスタンス
    sim = TradingSimulation()

    print("ティックファイル:", excel_file)
    # 繰り返し学習
    for epoch in range(epochs):
        # 1行ずつシミュレーションに流す
        for i, row in df.iterrows():
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]

            # 最後の行だけ強制返済フラグを立てる
            force_close = (i == len(df) - 1)
            action = sim.add(ts, price, volume, force_close=force_close)
            # print(f"Time: {ts}, Price: {price}, Action: {action}")

        # 結果を保存
        df_result = sim.finalize()
        print(f"Epoch: {epoch}, 総実現損益: {df_result['Profit'].sum():.2f}")
        df_result.to_csv(os.path.join("results", f"trade_results_epoch_{epoch}.csv"))