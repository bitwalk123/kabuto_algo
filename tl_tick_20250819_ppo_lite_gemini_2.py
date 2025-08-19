import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os


## PPO-lite Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Actor-Critic Network
        self.actor = self.create_actor_network(state_dim, action_dim)
        self.critic = self.create_critic_network(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def create_actor_network(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def create_critic_network(self, state_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        # 確率に基づいてアクションを選択
        action = torch.multinomial(action_probs, 1).item()
        return action

    def learn(self, states, actions, rewards, next_states, dones):
        # バッチ処理のためのテンソル化
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # ターゲット値の計算
        next_values = self.critic(next_states).detach()
        targets = rewards + self.gamma * next_values * (1 - dones)

        # 価値関数の損失計算と更新
        values = self.critic(states)
        critic_loss = nn.MSELoss()(values, targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # PPOのActor更新（ここでは簡略化）
        # 実際のPPOでは、古いポリシーと新しいポリシーの比率を計算し、クリッピングを行います
        # 簡易実装のため、ここではActorの損失を価値関数のアドバンテージに基づいて計算します。
        with torch.no_grad():
            advantages = targets - values

        old_action_probs = self.actor(states).gather(1, actions.unsqueeze(1)).detach()
        new_action_probs = self.actor(states).gather(1, actions.unsqueeze(1))

        ratio = new_action_probs / old_action_probs

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


## TradingSimulation
class TradingSimulation:
    def __init__(self, model_path="ppo_model.pth"):
        self.model_path = model_path
        self.state_dim = 2  # 状態空間: [直前の価格差, 出来高の変化率]
        self.action_dim = 3  # 3つの行動: 0=何もしない, 1=買い, 2=売り

        # モデルの読み込みと初期化
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        if os.path.exists(self.model_path):
            try:
                self.agent.load_model(self.model_path)
                print(f"✅ 既存の学習モデルを読み込みました: {self.model_path}")
            except Exception as e:
                print(f"⚠️ 既存モデルの読み込みに失敗しました ({e})。新しいモデルを生成します。")
                os.remove(self.model_path)
                self.agent = PPOAgent(self.state_dim, self.action_dim)
                print("🆕 新しい学習モデルを生成しました。")
        else:
            print("🆕 学習モデルが存在しないため、新しいモデルを生成しました。")

        # シミュレーション状態の初期化
        self.position = 0  # 0=ポジションなし, 1=買い, -1=売り
        self.entry_price = 0  # 建玉価格
        self.realized_profit = 0  # 実現損益
        self.results = []

        # 過去のデータを保持（状態の生成用）
        self.last_price = 0
        self.last_volume = 0
        self.last_state = None
        self.last_action = None

    def add(self, ts, price, volume, force_close=False):
        """
        別プログラムからティックデータを受け取り、売買アクションを決定する
        """
        # 最初のティックデータの場合、過去データと状態を初期化
        if self.last_price == 0:
            self.last_price = price
            self.last_volume = volume
            return "初期化"

        # 状態（State）の生成
        # price_change: 直前の株価からの変化率
        price_change = (price - self.last_price) / self.last_price if self.last_price != 0 else 0

        # volume_change: 出来高の増分を対数で正規化
        # 累計出来高なので、増加分のみ考慮。ログを取ることで桁数を抑える
        volume_increment = volume - self.last_volume
        if volume_increment > 0:
            volume_change = np.log1p(volume_increment)  # log(1+x)で0の時も対応
        else:
            volume_change = 0

        state = [price_change * 100, volume_change]  # 変化率をパーセントにするなど調整可能

        # 報酬（Reward）の計算と学習
        reward = 0
        if self.last_state is not None and self.last_action is not None:
            # 前回の行動に対する報酬を計算
            if self.last_action == 1:  # 買いの場合
                reward = (price - self.entry_price) * self.position * 100  # 含み益
            elif self.last_action == 2:  # 売り(空売り)の場合
                reward = (self.entry_price - price) * self.position * 100  # 含み益

            # 含み益をそのまま報酬にすると、値が大きくなりすぎるため、正規化またはスケーリングが必要
            reward = reward / 1000  # 例：1000円あたりの含み益を報酬とする

            # 既存のポジションがあれば、含み損益を報酬に加える
            if self.position != 0:
                unrealized_profit = (price - self.entry_price) * self.position * 100
                reward += unrealized_profit / 100000  # 含み益の報酬を加算

            # エピソードの終了判定
            done = False
            self.agent.learn([self.last_state], [self.last_action], [reward], [state], [done])

        # 強制返済フラグの処理
        if force_close and self.position != 0:
            self.close_position(price)
            action_desc = "強制返済"
            self.results.append({
                "Time": ts,
                "Price": price,
                "Action": action_desc,
                "Reward": reward,
                "Profit": self.realized_profit
            })
            self.position = 0
            self.entry_price = 0
            self.last_state = state
            self.last_action = 0  # 強制返済後のアクションは0(何もしない)としておく
            return action_desc

        # PPOエージェントに行動を決定させる
        action = self.agent.get_action(state)

        # 売買ロジック
        action_desc = "何もしない"

        if action == 1 and self.position == 0:
            self.open_position(price, "buy")
            action_desc = "買い"
        elif action == 2 and self.position == 0:
            self.open_position(price, "sell")
            action_desc = "売り"
        elif action == 1 and self.position == -1:
            self.close_position(price)
            action_desc = "買い返済"
        elif action == 2 and self.position == 1:
            self.close_position(price)
            action_desc = "売り返済"

        # 結果を記録
        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": action_desc,
            "Reward": reward,
            "Profit": self.realized_profit if action_desc.endswith("返済") else None
        })

        # 状態を更新
        self.last_price = price
        self.last_volume = volume
        self.last_state = state
        self.last_action = action

        return action_desc

    def open_position(self, price, side):
        self.entry_price = price
        self.position = 1 if side == "buy" else -1

    def close_position(self, price):
        if self.position == 1:  # 買いポジションを返済
            profit = (price - self.entry_price) * 100
        elif self.position == -1:  # 売りポジションを返済
            profit = (self.entry_price - price) * 100
        else:
            profit = 0

        self.realized_profit += profit
        self.position = 0

    def finalize(self):
        """
        シミュレーション終了時に結果を返す
        """
        # 最終状態の学習
        if self.last_state is not None and self.last_action is not None:
            # 最終的な報酬の計算 (含み益などを清算)
            final_reward = 0
            if self.position != 0:
                final_reward += self.realized_profit  # 実現損益を最終報酬に加える

            self.agent.learn([self.last_state], [self.last_action], [final_reward], [self.last_state],
                             [True])  # done=Trueでエピソード終了を通知

        # モデルを保存
        self.agent.save_model(self.model_path)

        df_result = pd.DataFrame(self.results)
        # 次のEpochのために状態をリセット
        self.results = []
        self.position = 0
        self.entry_price = 0
        self.realized_profit = 0
        self.last_price = 0
        self.last_volume = 0
        self.last_state = None
        self.last_action = None

        return df_result


## 別プログラム（メインスクリプト）
if __name__ == "__main__":
    # 実際にはデータファイルパスを適切に設定してください
    # Windowsで動くことを考慮して、パス区切り文字はos.path.joinを使うのが安全
    # Excelの読み込みにはopenpyxlが必要です（pip install openpyxl）

    excel_file = "data/tick_20250818_7011.xlsx"
    epochs = 10

    # ティックデータを読み込む
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"エラー: 指定されたファイルが見つかりません: {excel_file}")
        exit()

    print(df)

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
            # print(ts, price, action)

        # 結果を保存
        df_result = sim.finalize()
        total_profit = df_result["Profit"].sum() if "Profit" in df_result.columns and not df_result[
            "Profit"].isnull().all() else 0
        print(f"Epoch: {epoch}, 総収益: {total_profit}円")
        df_result.to_csv(f"trade_results_{epoch}.csv")
