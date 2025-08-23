import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
#from stable_baselines3.common.exceptions import NoSupportError

MODEL_PATH = "models/trading_dqn.zip"

# -------------------------
# 強化学習環境
# -------------------------
class TradingEnv(gym.Env):
    """
    アクション:
        0: ホールド
        1: 新規買い (Buy)
        2: 新規売り (Sell)
        3: 返済 (Close)

    観測量:
        - 株価 (正規化)
        - log1p(差分出来高)
        - 建玉の方向 (1=買い, -1=売り, 0=ノーポジ)
        - 建玉の含み益 (正規化)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # 観測空間 (株価, 出来高, ポジション, 含み益)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # アクション空間
        self.action_space = spaces.Discrete(4)

        self.reset_env_vars()

    def reset_env_vars(self):
        self.current_price = None
        self.last_volume = None
        self.position = 0  # 1=買い, -1=売り, 0=ノーポジ
        self.entry_price = 0
        self.total_profit = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env_vars()
        obs = np.zeros(4, dtype=np.float32)
        return obs, {}

    def step(self, action, price, volume, force_close=False):
        reward = 0.0
        realized_profit = 0.0

        # 差分出来高
        if self.last_volume is None:
            vol_diff = 0
        else:
            vol_diff = max(volume - self.last_volume, 0)
        self.last_volume = volume
        vol_feature = np.log1p(vol_diff)

        # アクション処理
        if action == 1 and self.position == 0:  # 新規買い
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # 新規売り
            self.position = -1
            self.entry_price = price
        elif action == 3 and self.position != 0:  # 返済
            if self.position == 1:
                realized_profit = (price - self.entry_price) * 100
            elif self.position == -1:
                realized_profit = (self.entry_price - price) * 100
            self.total_profit += realized_profit
            reward += realized_profit
            self.position = 0
            self.entry_price = 0

        # 強制返済
        if force_close and self.position != 0:
            if self.position == 1:
                realized_profit = (price - self.entry_price) * 100
            elif self.position == -1:
                realized_profit = (self.entry_price - price) * 100
            self.total_profit += realized_profit
            reward += realized_profit
            self.position = 0
            self.entry_price = 0
            self.done = True

        # 含み益も報酬に追加
        unrealized = 0
        if self.position == 1:
            unrealized = (price - self.entry_price) * 100
        elif self.position == -1:
            unrealized = (self.entry_price - price) * 100
        reward += unrealized * 0.01  # 含み益を弱めに加点

        # 観測
        obs = np.array([
            price / 10000.0,   # 正規化株価
            vol_feature,       # log1p差分出来高
            float(self.position),
            unrealized / 10000.0
        ], dtype=np.float32)

        return obs, reward, self.done, False, {"profit": realized_profit, "action": action}


# -------------------------
# シミュレーション本体
# -------------------------
class TradingSimulation:
    def __init__(self):
        self.env = TradingEnv()
        self.results = []

        # モデル読み込み / 新規作成
        if os.path.exists(MODEL_PATH):
            try:
                self.model = DQN.load(MODEL_PATH, env=self.env)
                print("既存モデルを読み込みました:", MODEL_PATH)
            except (ValueError, NoSupportError):
                print("既存モデルが無効のため、新規作成します。")
                self.model = self._create_new_model()
        else:
            print("既存モデルが存在しないため、新規作成します。")
            self.model = self._create_new_model()

    def _create_new_model(self):
        model = DQN("MlpPolicy", self.env, verbose=0, learning_rate=1e-3, buffer_size=5000)
        model.save(MODEL_PATH)
        return model

    def add(self, ts, price, volume, force_close=False):
        obs = np.array([price / 10000.0, 0.0, float(self.env.position), 0.0], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)

        obs, reward, done, _, info = self.env.step(action, price, volume, force_close)

        self.results.append({
            "Time": ts,
            "Price": price,
            "Action": action,
            "Profit": info["profit"],
            "Reward": reward,
        })
        return action

    def finalize(self):
        df = pd.DataFrame(self.results)
        self.results = []  # リセット
        # モデルを保存
        self.model.save(MODEL_PATH)
        return df
