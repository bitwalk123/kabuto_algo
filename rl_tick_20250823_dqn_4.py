import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 簡易 DQN ネットワーク
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TradingSimulation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0           # 建玉 (1:買い, -1:売り, 0:ノーポジ)
        self.entry_price = None     # 建玉価格
        self.realized_pnl = 0       # 実現損益（円単位、整数）
        self.trades = []            # 取引履歴

    def add(self, ts, price, volume, force_close=False):
        """
        1ティック分のデータを処理して行動を返す
        ts: 時刻
        price: 約定価格
        volume: 出来高
        force_close: 最終行など強制クローズ
        """
        action = "HOLD"
        reward = 0.0

        # --- 強制クローズ処理 ---
        if force_close and self.position != 0:
            pnl = int((price - self.entry_price) * self.position)
            self.realized_pnl += pnl
            reward = pnl / 1000.0
            self.trades.append({"Time": ts, "Action": "FORCE_CLOSE", "Price": price, "Profit": pnl})
            self.position = 0
            self.entry_price = None
            return action

        # --- 売買ロジック（ここでは超シンプル：ランダム） ---
        # 実際には強化学習のポリシーで決める
        if np.random.rand() < 0.01:   # 1% の確率で新規建玉 or クローズ
            if self.position == 0:
                # 新規建玉
                self.position = np.random.choice([1, -1])  # 買い or 売り
                self.entry_price = price
                action = "BUY" if self.position == 1 else "SELL"
                self.trades.append({"Time": ts, "Action": action, "Price": price, "Profit": 0})
            else:
                # クローズ
                pnl = int((price - self.entry_price) * self.position)
                self.realized_pnl += pnl
                reward = pnl / 1000.0
                action = "CLOSE"
                self.trades.append({"Time": ts, "Action": action, "Price": price, "Profit": pnl})
                self.position = 0
                self.entry_price = None

        return action

    def finalize(self):
        """
        学習1エポック終了後に呼び出して結果をまとめる
        """
        df_result = pd.DataFrame(self.trades)
        # 次のエポック用にリセット
        self.reset()
        return df_result