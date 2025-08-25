# -*- coding: utf-8 -*-
"""
PPO-lite デイトレ売買シミュレーション（ティック 1秒、単一銘柄）

要件対応ポイント:
- add(ts, price, volume, force_close=False) を外部から毎秒呼び出す想定
- 生のティックから以下の特徴量を add 内で生成
  * Δvolume -> np.log1p(Δvolume)
  * 価格ベースの特徴量: MA(10), Volatility(10), RSI(14)
- 呼び値=1円、売買単位=100株、ナンピン禁止（ポジションは常に最大1単位）
- 信用売買の約定価格ルールに対応
  * 新規売り: price - 1 / 新規買い: price + 1
  * 返済売り: price - 1 / 返済買い: price + 1
- 報酬設計: 確定益（建玉解消時）+ α * 含み益（途中）
- PPO-lite: 小規模 Actor-Critic（3アクション: Long/CloseShort, Short/CloseLong, Hold）
- 既存モデルのロード/保存/検証と標準出力メッセージ
- finalize() で結果 DataFrame を返却し、内部状態を次エポック用にリセット

依存: torch, numpy, pandas
CPUのみで動作可（GPUがあれば自動使用）
"""

from __future__ import annotations
import os
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ===================== ユーティリティ =====================

def rsi_from_prices(prices: np.ndarray, window: int = 14) -> float:
    """RSI(14) を numpy で都度計算（軽量のため簡易版）。
    prices: 最新が末尾。
    """
    if len(prices) < window + 1:
        return np.nan
    delta = np.diff(prices[-(window + 1):])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def zscore(x: float, mean: float, std: float) -> float:
    if std <= 1e-8:
        return 0.0
    return (x - mean) / std


# ===================== PPO Lite モデル =====================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, action_dim: int = 3):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.base(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value.squeeze(-1)


@dataclass
class Transition:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor


class PPOLite:
    def __init__(
            self,
            obs_dim: int,
            action_dim: int = 3,
            lr: float = 3e-4,
            gamma: float = 0.99,
            clip_eps: float = 0.2,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            update_epochs: int = 4,
            batch_size: int = 256,
            hidden: int = 64,
            device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, hidden, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.buffer: List[Transition] = []

    def policy(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """観測から行動をサンプル。戻り値: (action_idx, logprob, value)"""
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            probs = torch.distributions.Categorical(logits=logits)
            act = probs.sample()
            logp = probs.log_prob(act)
        return int(act.item()), float(logp.item()), float(value.item())

    def push(self, obs, act, logp, val, rew, done):
        self.buffer.append(
            Transition(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(act, dtype=torch.long),
                torch.tensor(logp, dtype=torch.float32),
                torch.tensor(val, dtype=torch.float32),
                torch.tensor(rew, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32),
            )
        )

    def _compute_returns_adv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # GAEではなく単純モンテカルロを採用（Lite）
        rews = torch.stack([t.rew for t in self.buffer])
        vals = torch.stack([t.val for t in self.buffer])
        dones = torch.stack([t.done for t in self.buffer])
        acts = torch.stack([t.act for t in self.buffer])
        logps_old = torch.stack([t.logp for t in self.buffer])

        returns = torch.zeros_like(rews)
        G = 0.0
        for i in reversed(range(len(rews))):
            G = rews[i] + self.gamma * G * (1.0 - dones[i])
            returns[i] = G
        adv = returns - vals
        # 標準化で安定化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return returns, adv, acts, logps_old, vals

    def maybe_update(self):
        if len(self.buffer) < self.batch_size:
            return
        returns, adv, acts, logps_old, _vals = self._compute_returns_adv()

        obs = torch.stack([t.obs for t in self.buffer]).to(self.device)
        acts = acts.to(self.device)
        logps_old = logps_old.to(self.device)
        returns = returns.to(self.device)
        adv = adv.to(self.device)

        ds = torch.utils.data.TensorDataset(obs, acts, logps_old, returns, adv)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

        self.net.train()
        for _ in range(self.update_epochs):
            for b_obs, b_act, b_logp_old, b_ret, b_adv in loader:
                b_obs = b_obs.to(self.device)
                logits, values = self.net(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, b_ret)
                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

        # 更新後にバッファクリア
        self.buffer.clear()

    def act_and_record(self, obs: np.ndarray, reward: float, done: bool) -> int:
        # 直前の obs, act の保存は外部から push 済み想定。ここは観測から次の行動を決めるだけでも可。
        act, logp, val = self.policy(obs)
        return act


# ===================== シミュレーション本体 =====================

class TradingSimulation:
    def __init__(
            self,
            lot_size: int = 100,
            tick_size: float = 1.0,
            unrealized_weight: float = 0.05,  # 含み益を報酬に入れる重み
            update_every: int = 256,  # PPO 更新バッチ閾値（PPOLite.batch_size と整合）
            model_path: str = "models/ppo_trader.pth",
            feature_window_ma: int = 10,
            feature_window_vol: int = 10,
            feature_window_rsi: int = 14,
            zscore_window: int = 100,
            epsilon: float = 0.05,  # ε-greedy 探索率
            seed: int = 42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.lot = lot_size
        self.tick = tick_size
        self.unreal_w = unrealized_weight
        self.model_path = model_path
        self.update_every = update_every

        # ローリング計算用
        self.prices = deque(maxlen=max(feature_window_ma, feature_window_vol, feature_window_rsi, zscore_window) + 2)
        self.volumes = deque(maxlen=4)  # Δ計算用（直近）

        self.window_ma = feature_window_ma
        self.window_vol = feature_window_vol
        self.window_rsi = feature_window_rsi
        self.zscore_window = zscore_window

        # 結果
        self.records: List[Tuple[float, float, str, float]] = []  # Time, Price, Action, Profit

        # ポジション状態
        self.position: Optional[str] = None  # 'long' or 'short' or None
        self.entry_price: Optional[float] = None

        # 学習器
        self.obs_dim = 6  # [price_z, ma_z, vol_z, rsi/100, dprice_z, log1p(dv)_z]
        self.agent = PPOLite(obs_dim=self.obs_dim, batch_size=update_every)

        # モデルロード
        self._load_or_init_model()

        # 直近観測（初期化）
        self.prev_obs: Optional[np.ndarray] = None
        self.prev_logp: Optional[float] = None
        self.prev_val: Optional[float] = None
        self.prev_act: Optional[int] = None
        self.prev_ts: Optional[float] = None
        self.prev_price: Optional[float] = None

        # 集計用（ステップ間報酬の蓄積）
        self.step_counter = 0

    # -------------- モデルのロード/保存 --------------
    def _load_or_init_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location="cpu")
                self.agent.net.load_state_dict(state["model"])  # type: ignore
                print("既存モデルを読み込みました:", self.model_path)
            except Exception as e:
                print("既存モデルが無効のため新規作成します:", e)
                self._save_model(new_model=True)
        else:
            print("有効な既存モデルが見つからないため新規作成します。")
            self._save_model(new_model=True)

    def _save_model(self, new_model: bool = False):
        torch.save({"model": self.agent.net.state_dict()}, self.model_path)
        if new_model:
            print("No valid existing model found — created a new model.")
        else:
            # print("Model saved to", self.model_path)
            pass

    # -------------- 観測ベクトル生成 --------------
    def _build_obs(self, price: float, volume: float) -> np.ndarray:
        self.prices.append(price)
        self.volumes.append(volume)

        # Δvolume と log1p 圧縮
        if len(self.volumes) >= 2:
            dvol = max(0.0, self.volumes[-1] - self.volumes[-2])
        else:
            dvol = 0.0
        dv_feature = math.log1p(dvol)

        # 移動平均/ボラ（10）
        arr = np.array(self.prices, dtype=np.float32)
        ma = np.nan
        vol = np.nan
        if len(arr) >= self.window_ma:
            ma = float(arr[-self.window_ma:].mean())
        if len(arr) >= self.window_vol:
            vol = float(arr[-self.window_vol:].std(ddof=0))

        # RSI(14)
        rsi = rsi_from_prices(arr, self.window_rsi)

        # dprice（直近差分）
        if len(arr) >= 2:
            dprice = float(arr[-1] - arr[-2])
        else:
            dprice = 0.0

        # z-score 正規化（直近 zscore_window）
        zs = arr[-self.zscore_window:]
        m = float(zs.mean()) if len(zs) > 0 else price
        s = float(zs.std(ddof=0) + 1e-6)

        price_z = zscore(price, m, s)
        ma_z = 0.0 if math.isnan(ma) else zscore(ma, m, s)
        vol_z = 0.0 if math.isnan(vol) else vol / (s + 1e-6)  # ボラはスケール合わせ
        rsi_scaled = 0.0 if math.isnan(rsi) else (rsi / 100.0)
        dprice_z = dprice / (s + 1e-6)

        # dvol の正規化: 過去dv の平均/分散を dvol_stats で近似（ここでは価格と同じ s で粗くスケール）
        dv_z = (dv_feature - 1.0)  # log1p の中心化の簡易版

        obs = np.array([price_z, ma_z, vol_z, rsi_scaled, dprice_z, dv_z], dtype=np.float32)
        return obs

    # -------------- 取引ロジック（約定/損益計算） --------------
    def _fill_and_update(self, ts: float, price: float, action_idx: int, force: bool = False) -> str:
        """行動を実売買アクションに変換し、必要なら損益を確定。戻り値はアクション名（日本語）。"""
        action_name = "ホールド"
        realized = 0.0
        if action_idx == 2:
            # HOLD
            pass
        elif action_idx == 0:
            # Go Long / Close Short
            if self.position is None:
                # 新規買い（ price + tick ）
                self.position = 'long'
                self.entry_price = price + self.tick
                action_name = "新規買い"
            elif self.position == 'short':
                # 返済買い（ price + tick ）
                exit_price = price + self.tick
                realized = (self.entry_price - exit_price) * self.lot  # entry は売り価格（短）
                self.position = None
                self.entry_price = None
                action_name = "返済買い（強制)" if force else "返済買い"
            else:
                action_name = "ホールド"
        elif action_idx == 1:
            # Go Short / Close Long
            if self.position is None:
                # 新規売り（ price - tick ）
                self.position = 'short'
                self.entry_price = price - self.tick
                action_name = "新規売り"
            elif self.position == 'long':
                # 返済売り（ price - tick ）
                exit_price = price - self.tick
                realized = (exit_price - self.entry_price) * self.lot
                self.position = None
                self.entry_price = None
                action_name = "返済売り（強制)" if force else "返済売り"
            else:
                action_name = "ホールド"

        # 記録（確定益は建玉解消時のみ >0 になる）
        if realized != 0.0:
            self.records.append((ts, price, action_name, realized))
        else:
            self.records.append((ts, price, action_name, 0.0))
        return action_name

    def _unrealized_pnl(self, price: float) -> float:
        if self.position is None or self.entry_price is None:
            return 0.0
        if self.position == 'long':
            return (price - self.entry_price) * self.lot
        else:
            return (self.entry_price - price) * self.lot

    # -------------- 公開 API --------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        外部の「別プログラム」から毎秒呼び出される。
        返り値: 実行した売買アクション（日本語）
        """
        obs = self._build_obs(price, volume)

        # 観測が十分に揃うまで学習/行動は保守的に Hold
        if np.isnan(obs).any():
            action_idx = 2  # Hold
            action_name = self._fill_and_update(ts, price, action_idx)
            self.prev_obs = obs
            self.prev_price = price
            self.prev_ts = ts
            return action_name

        # 方策に従って行動サンプル（ε-greedy 探索）
        if np.random.rand() < 0.05:  # self.epsilon（固定値で簡略）
            action_idx = np.random.randint(0, 3)
            # 値/ログ確率は後で再計算
            logits, val = self.agent.net(torch.tensor(obs).float().unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor([action_idx])).item()
            value = float(val.item())
        else:
            action_idx, logp, value = self.agent.policy(obs)

        # 途中報酬（含み益の一部）
        unreal = self._unrealized_pnl(price)
        step_reward = self.unreal_w * unreal

        # 強制返済
        if force_close and self.position is not None:
            # サイドに応じて反対側の行動に置換
            action_idx = 0 if self.position == 'short' else 1
            action_name = self._fill_and_update(ts, price, action_idx, force=True)
            # 強制決済の確定益を報酬に加算
            realized = self.records[-1][3]
            total_reward = step_reward + realized
            done = 1.0
            self.agent.push(obs, action_idx, logp, value, total_reward, done)
            self.agent.maybe_update()
            # 次エポックに向けて最終行も観測として扱う
            self.prev_obs = obs
            self.prev_price = price
            self.prev_ts = ts
            return action_name

        # 通常フロー: 約定/損益
        prev_pos = self.position
        action_name = self._fill_and_update(ts, price, action_idx)
        realized = self.records[-1][3]

        # 確定益（建玉解消時）を報酬に加算
        total_reward = step_reward + realized
        done = 0.0

        # 学習バッファに記録
        self.agent.push(obs, action_idx, logp, value, total_reward, done)
        self.agent.maybe_update()

        self.prev_obs = obs
        self.prev_price = price
        self.prev_ts = ts
        self.step_counter += 1
        return action_name

    def finalize(self) -> pd.DataFrame:
        """結果 DataFrame を返し、内部の結果記録のみリセット（モデル/学習器は保持）。"""
        df = pd.DataFrame(self.records, columns=["Time", "Price", "Action", "Profit"])
        # エポック終了時点でモデル保存
        self._save_model(new_model=False)
        # ログをリセット（次エポックへ）
        self.records.clear()
        return df


# ---------------------- 使い方メモ ----------------------
"""
from your_module import TradingSimulation

sim = TradingSimulation()
for i, row in df.iterrows():
    ts = row["Time"]; price = row["Price"]; volume = row["Volume"]
    force_close = (i == len(df) - 1)
    action = sim.add(ts, price, volume, force_close=force_close)

result_df = sim.finalize()
print(result_df.head())
"""

if __name__ == "__main__":
    # 簡易自己テスト（ダミーデータ）: 実運用では別プログラムから add() を呼ぶ
    rng = np.random.default_rng(0)
    base = 5000.0
    prices = base + np.cumsum(rng.normal(0, 2, size=1000))
    vols = np.cumsum(np.maximum(0, rng.poisson(1000, size=1000)).astype(float))
    times = np.arange(1000).astype(float)

    sim = TradingSimulation()
    for i in range(len(prices)):
        ts = times[i];
        price = float(round(prices[i], 0));
        volume = float(vols[i])
        force = (i == len(prices) - 1)
        sim.add(ts, price, volume, force_close=force)
    df = sim.finalize()
    print(df.tail())
