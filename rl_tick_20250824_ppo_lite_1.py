# trading_simulation.py
# -*- coding: utf-8 -*-
"""
デイトレ用・ティックベース PPO-lite（Actor-Critic）サンプル

要件:
- add(ts, price, volume, force_close=False) を1秒ティックごとに呼び出す
- 返り値は売買アクション（日本語文字列）
- 特徴量は add 内で生成（価格:MA/Volatility/RSI, 出来高: Δvolume→log1p）
- 100株・信用売買・ナンピン禁止・返済時は1円スリッページ
- 非線形報酬（含み益が大きいほど倍率UP）
- 結果DataFrame(Time, Price, Action, Profit) を finalize() で取得してリセット
- 学習モデルは自動ロード/保存（無効なら再生成し上書き）

依存:
  pip install torch pandas numpy
  （gymnasium不要）
"""
from __future__ import annotations
import os
import math
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ------------------------------
# ユーティリティ: オンライン標準化（Welford）
# ------------------------------
class OnlineScaler:
    def __init__(self, n_features: int, eps: float = 1e-6):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.eps = eps

    def partial_fit(self, x: np.ndarray):
        # x shape: (n_features,)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        var = np.where(self.n > 1, self.M2 / max(self.n - 1, 1), 1.0)
        std = np.sqrt(var) + self.eps
        return (x - self.mean) / std

    def state_dict(self) -> Dict:
        return {"n": self.n, "mean": self.mean.tolist(), "M2": self.M2.tolist(), "eps": self.eps}

    def load_state_dict(self, state: Dict):
        self.n = int(state.get("n", 0))
        self.mean = np.array(state.get("mean", [0] * len(self.mean)), dtype=np.float64)
        self.M2 = np.array(state.get("M2", [0] * len(self.M2)), dtype=np.float64)
        self.eps = float(state.get("eps", 1e-6))


# ------------------------------
# テクニカル指標
# ------------------------------

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------------------
# モデル
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_actions: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


# ------------------------------
# バッファ（PPO/GAE）
# ------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(self, state, action, logprob, reward, done, value):
        self.states.append(state.copy())
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self):
        self.__init__()


# ------------------------------
# 取引シミュレーション（学習器内蔵）
# ------------------------------
@dataclass
class Position:
    side: int  # 0: flat, +1: long, -1: short
    entry_price: float


class TradingSimulation:
    """別プログラムから add() を1秒ごとに呼ぶシミュレーション/学習クラス"""

    def __init__(self,
                 model_dir: str = "models",
                 model_file: str = "ppo_7011_20250824.pt",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_clip: float = 0.2,
                 lr: float = 3e-4,
                 update_after_steps: int = 128,
                 update_epochs: int = 2,
                 hidden: int = 128,
                 device: Optional[str] = None):
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, model_file)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.lr = lr
        self.update_after_steps = update_after_steps
        self.update_epochs = update_epochs
        self.hidden = hidden
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 特徴量用バッファ
        self.window = 14  # RSIに合わせて≥14
        self.prices = deque(maxlen=60)  # 余裕を持って保持
        self.volumes = deque(maxlen=2)
        self._prev_cum_volume: Optional[float] = None

        # 結果ログ
        self._rows: List[Dict] = []

        # 建玉状態
        self.position = Position(side=0, entry_price=0.0)
        self.TRADE_SIZE = 100  # 100株固定
        self.TICK = 1.0  # 呼び値（1円）

        # アクション定義
        # 0: ホールド, 1: 新規買い, 2: 新規売り, 3: 返済売り, 4: 返済買い
        self.n_actions = 5

        # 入力次元（特徴量）: [price, dprice, ma10, vol10, rsi14, log1p(delta_vol)]
        self.input_dim = 6

        # 正規化器
        self.scaler = OnlineScaler(self.input_dim)

        # モデル
        self.model = ActorCritic(self.input_dim, hidden=self.hidden, n_actions=self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # バッファ
        self.buf = RolloutBuffer()
        self._step_count = 0
        self._episode_started = False

        # モデルのロード or 新規作成
        self._load_or_init_model()

    # --------------------------
    # モデルロード/保存
    # --------------------------
    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                meta = ckpt.get("meta", {})
                if meta.get("input_dim") == self.input_dim and meta.get("n_actions") == self.n_actions:
                    self.model.load_state_dict(ckpt["model"]) \
                        if isinstance(ckpt, dict) and "model" in ckpt else self.model.load_state_dict(ckpt)
                    if "scaler" in ckpt:
                        self.scaler.load_state_dict(ckpt["scaler"])
                    print("既存モデルを読み込みました: ", self.model_path)
                else:
                    print("既存モデルは無効（入出力次元不一致）。新規作成して上書きします。")
                    self._save_model()
            except Exception as e:
                print("既存モデルの読み込みに失敗: ", e)
                print("新規モデルを作成します。")
                self._save_model()
        else:
            print("既存モデルが見つかりません。新規モデルを作成します。")
            self._save_model()

    def _save_model(self):
        ckpt = {
            "model": self.model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "meta": {
                "input_dim": self.input_dim,
                "n_actions": self.n_actions,
                "hidden": self.hidden,
                "created_at": time.time(),
            }
        }
        torch.save(ckpt, self.model_path)
        # print("Model saved to", self.model_path)

    # --------------------------
    # 観測生成
    # --------------------------
    def _build_features(self, price: float, cum_volume: float) -> Optional[np.ndarray]:
        self.prices.append(price)

        # 差分出来高
        if self._prev_cum_volume is None:
            delta_vol = 0.0
        else:
            dv = cum_volume - self._prev_cum_volume
            delta_vol = max(dv, 0.0)  # 欠損/リセット耐性
        self._prev_cum_volume = cum_volume

        log_dv = math.log1p(delta_vol)

        # pandasでテクニカル
        s = pd.Series(list(self.prices), dtype=float)
        ma10 = s.rolling(window=10, min_periods=1).mean().iloc[-1]
        vol10 = s.rolling(window=10, min_periods=1).std(ddof=0).iloc[-1]
        rsi14 = compute_rsi(s, window=14).fillna(50.0).iloc[-1]

        # 価格差分
        dprice = 0.0 if len(self.prices) < 2 else (self.prices[-1] - self.prices[-2])

        x = np.array([price, dprice, ma10, vol10, rsi14, log_dv], dtype=np.float64)
        self.scaler.partial_fit(x)
        x_norm = self.scaler.transform(x).astype(np.float32)
        return x_norm

    # --------------------------
    # アクション選択
    # --------------------------
    def _select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        self.model.eval()
        with torch.no_grad():
            st = torch.from_numpy(state).unsqueeze(0).to(self.device)
            logits, value = self.model(st)
            # 無効なアクションをマスク（建玉状況に応じて）
            mask = self._action_mask()
            logits = logits + torch.log(mask)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item())

    def _action_mask(self) -> torch.Tensor:
        # 0:ホールドは常に可
        # flat: 新規買い/新規売りのみ可（返済×）
        # long: 返済売りのみ可（新規×）
        # short: 返済買いのみ可
        mask = torch.ones(self.n_actions, device=self.device)
        if self.position.side == 0:
            mask[3] = 1e-8  # 返済売り不可
            mask[4] = 1e-8  # 返済買い不可
        elif self.position.side == +1:
            mask[1] = 1e-8  # 新規買い不可
            mask[2] = 1e-8  # 新規売り不可
            mask[4] = 1e-8  # 返済買い不可
        elif self.position.side == -1:
            mask[1] = 1e-8
            mask[2] = 1e-8
            mask[3] = 1e-8
        return mask

    # --------------------------
    # 約定/損益ロジック
    # --------------------------
    def _execute(self, ts: float, price: float, action_id: int, force_close: bool) -> Tuple[str, float, float]:
        action_str = ["ホールド", "新規買い", "新規売り", "返済売り", "返済買い"][action_id]
        realized_profit = 0.0

        # 強制返済フラグが立ったら、アクションに関わらずクローズ
        if force_close and self.position.side != 0:
            if self.position.side == +1:
                exit_price = price - self.TICK  # 返済売り
                realized_profit = (exit_price - self.position.entry_price) * self.TRADE_SIZE
                action_str = "返済売り"
            elif self.position.side == -1:
                exit_price = price + self.TICK  # 返済買い
                realized_profit = (self.position.entry_price - exit_price) * self.TRADE_SIZE
                action_str = "返済買い"
            self.position = Position(0, 0.0)
            self._rows.append({"Time": ts, "Price": price, "Action": action_str, "Profit": realized_profit})
            return action_str, realized_profit, 0.0

        # 通常のアクション
        if self.position.side == 0:
            if action_str == "新規買い":
                self.position = Position(+1, price)
            elif action_str == "新規売り":
                self.position = Position(-1, price)
            # 新規時点では確定損益なし
            self._rows.append({"Time": ts, "Price": price, "Action": action_str, "Profit": 0.0})
        elif self.position.side == +1:
            if action_str == "返済売り":
                exit_price = price - self.TICK
                realized_profit = (exit_price - self.position.entry_price) * self.TRADE_SIZE
                self.position = Position(0, 0.0)
                self._rows.append({"Time": ts, "Price": price, "Action": action_str, "Profit": realized_profit})
            else:
                # ホールド（または無効アクションはマスク済）
                self._rows.append({"Time": ts, "Price": price, "Action": "ホールド", "Profit": 0.0})
                action_str = "ホールド"
        elif self.position.side == -1:
            if action_str == "返済買い":
                exit_price = price + self.TICK
                realized_profit = (self.position.entry_price - exit_price) * self.TRADE_SIZE
                self.position = Position(0, 0.0)
                self._rows.append({"Time": ts, "Price": price, "Action": action_str, "Profit": realized_profit})
            else:
                self._rows.append({"Time": ts, "Price": price, "Action": "ホールド", "Profit": 0.0})
                action_str = "ホールド"

        # 含み損益（この時点の評価）
        unrealized = 0.0
        if self.position.side == +1:
            unrealized = (price - self.position.entry_price) * self.TRADE_SIZE
        elif self.position.side == -1:
            unrealized = (self.position.entry_price - price) * self.TRADE_SIZE
        return action_str, realized_profit, unrealized

    # --------------------------
    # 報酬設計（非線形増加 + 微小ペナルティ/ボーナス）
    # --------------------------
    def _compute_reward(self, unrealized: float, d_equity: float) -> float:
        # 基本は評価損益の増分（delta equity）を報酬とする
        reward = d_equity / 100.0  # スケーリング（学習安定化）
        # 現在の含み益水準に応じた倍率
        if unrealized >= 1000:
            reward *= 2.0
        elif unrealized >= 500:
            reward *= 1.5
        elif unrealized >= 200:
            reward *= 1.0
        # わずかなボーナス/ペナルティ（現在値ベース）
        if unrealized > 0:
            reward += 0.01
        elif unrealized < 0:
            reward -= 0.01
        return float(reward)

    # --------------------------
    # 公開API: add
    # --------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """ティックを1件処理し、アクションを返す"""
        state = self._build_features(price, volume)
        # 学習開始前のウォームアップ（特徴量が安定しない最初の数ティック）
        if state is None:
            self._rows.append({"Time": ts, "Price": price, "Action": "ホールド", "Profit": 0.0})
            return "ホールド"

        # アクション選択
        action_id, logp, value = self._select_action(state)

        # 実行 & 損益計算
        prev_unrealized = 0.0
        if self.position.side == +1:
            prev_unrealized = (price - self.position.entry_price) * self.TRADE_SIZE
        elif self.position.side == -1:
            prev_unrealized = (self.position.entry_price - price) * self.TRADE_SIZE

        action_str, realized_profit, unrealized = self._execute(ts, price, action_id, force_close)

        # 実行後の含み損益（ホールド時のみ評価変化）
        d_equity = unrealized - prev_unrealized
        reward = self._compute_reward(unrealized, d_equity)

        # done（エピソード終端）
        done = bool(force_close)

        # バッファに格納
        self.buf.add(state, action_id, logp, reward, done, value)
        self._step_count += 1

        # 一定ステップごと、または終端で更新
        if self._step_count % self.update_after_steps == 0 or done:
            self._update(value_bootstrap=0.0 if done else None)

        return action_str

    # --------------------------
    # PPO-lite 更新
    # --------------------------
    def _compute_gae(self, rewards, values, dones, last_value: float = 0.0):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        returns = adv + np.array(values, dtype=np.float32)
        return adv, returns

    def _update(self, value_bootstrap: Optional[float]):
        if len(self.buf.states) == 0:
            return

        states = torch.tensor(np.array(self.buf.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buf.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.buf.logprobs, dtype=torch.float32, device=self.device)
        rewards = np.array(self.buf.rewards, dtype=np.float32)
        dones = np.array(self.buf.dones, dtype=np.bool_)
        values = np.array(self.buf.values, dtype=np.float32)

        # ブートストラップ値（エピソード継続中の更新では現在値を再計算）
        if value_bootstrap is None:
            with torch.no_grad():
                _, v_last = self.model(states[-1:].detach())
                value_bootstrap = float(v_last.item())
        adv, ret = self._compute_gae(rewards, values, dones, last_value=value_bootstrap)

        advantages = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-6), dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret, dtype=torch.float32, device=self.device)

        # PPO更新
        for _ in range(self.update_epochs):
            logits, value_pred = self.model(states)
            # アクションマスクは学習時は適用しない（分布の正規化が崩れるため）。
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value_pred, returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.buf.clear()
        # 更新のたびにモデルを保存（簡便）
        self._save_model()

    # --------------------------
    # 公開API: finalize
    # --------------------------
    def finalize(self) -> pd.DataFrame:
        """結果DataFrameを返して内部状態をリセット。呼び出し後に学習モデルは保存済み。"""
        df = pd.DataFrame(self._rows, columns=["Time", "Price", "Action", "Profit"])
        # リセット（ただしモデル・スケーラは維持して継続学習可能に）
        self._rows.clear()
        self.prices.clear()
        self.volumes.clear()
        self._prev_cum_volume = None
        self.position = Position(0, 0.0)
        self._step_count = 0
        self.buf.clear()
        return df


# --------------
# 簡易テスト
# --------------
if __name__ == "__main__":
    # ダミーデータで動作確認
    sim = TradingSimulation()
    ts0 = 1_700_000_000
    price = 1000.0
    cumv = 0.0
    np.random.seed(0)
    for i in range(600):  # 10分相当
        ts = ts0 + i
        price += np.random.randn() * 2.0  # ランダムウォーク
        price = max(100.0, price)
        cumv += np.random.randint(0, 2000)
        force = (i == 599)
        a = sim.add(ts, float(price), float(cumv), force_close=force)
        # print(ts, price, cumv, a)
    df_res = sim.finalize()
    print(df_res.tail())
