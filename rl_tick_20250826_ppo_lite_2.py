# -*- coding: utf-8 -*-
"""
PPO-lite デイトレ売買シミュレーション（ティック1秒・単一銘柄・3アクション）

要求仕様:
- 外部プログラムから毎秒 `add(ts, price, volume, force_close=False)` を呼び出す
- 生ティックから特徴量を内部で生成
  * Δvolume = max(0, volume_t - volume_{t-1})
  * log1p(Δvolume) で圧縮
  * 価格ベース: MA(10), Volatility(10), RSI(14)
  * 追加: price z-score, dprice z-score
- 信用売買・呼び値=1円・売買単位=100株・ナンピン禁止（常に最大1単位の建玉）
  * 新規買い/返済買いは `price + 1`
  * 新規売り/返済売りは `price - 1`
- アクションは 3 種: [買い, 売り, ホールド]
  * ノーポジ時: 買い→新規買い / 売り→新規売り
  * 買い持ち時: 売り→返済売り / 買い→ホールド
  * 売り持ち時: 買い→返済買い / 売り→ホールド
- 報酬: 確定益 + α×含み益（α=0.1 デフォルト）
- 既存モデルのロード/保存/検証と標準出力
- `finalize()` で結果 DataFrame（Time, Price, Action, Profit）を返し、記録をリセット

依存: torch, numpy, pandas  （CPUのみでOK）
"""

# from __future__ import annotations
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


# ---------------- ユーティリティ ----------------

def rsi_from_prices(prices: np.ndarray, window: int = 14) -> float:
    if len(prices) < window + 1:
        return float('nan')
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


# ---------------- PPO-Lite ----------------

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
            gamma: float = 0.95,
            clip_eps: float = 0.2,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            update_epochs: int = 4,
            batch_size: int = 512,
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
        self.buffer: List[Transition] = []

    def policy(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
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

    def _compute_returns_adv(self):
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return returns, adv, acts, logps_old

    def maybe_update(self):
        if len(self.buffer) < self.batch_size:
            return
        returns, adv, acts, logps_old = self._compute_returns_adv()
        obs = torch.stack([t.obs for t in self.buffer]).to(self.device)
        acts = acts.to(self.device)
        logps_old = logps_old.to(self.device)
        returns = returns.to(self.device)
        adv = adv.to(self.device)

        ds = torch.utils.data.TensorDataset(obs, acts, logps_old, returns, adv)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        self.net.train()
        for _ in range(self.update_epochs):
            for b_obs, b_act, b_logp_old, b_ret, b_adv in loader:
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

        self.buffer.clear()


# ---------------- シミュレーション本体 ----------------

class TradingSimulation:
    """3アクション = [買い(0), 売り(1), ホールド(2)] の軽量PPOトレーナー内蔵シミュレータ"""

    def __init__(
            self,
            lot_size: int = 100,
            tick_size: float = 1.0,
            unrealized_weight: float = 0.1,
            update_every: int = 512,
            model_path: str = "models/ppo_trader_3act.pth",
            feature_window_ma: int = 10,
            feature_window_vol: int = 10,
            feature_window_rsi: int = 14,
            zscore_window: int = 200,
            seed: int = 42,
    ):
        np.random.seed(seed);
        torch.manual_seed(seed)

        self.lot = lot_size
        self.tick = tick_size
        self.unreal_w = unrealized_weight
        self.model_path = model_path
        self.update_every = update_every

        # ローリングバッファ
        self.prices = deque(maxlen=max(feature_window_ma, feature_window_vol, feature_window_rsi, zscore_window) + 2)
        self.volumes = deque(maxlen=4)

        self.window_ma = feature_window_ma
        self.window_vol = feature_window_vol
        self.window_rsi = feature_window_rsi
        self.zscore_window = zscore_window

        # 結果レコード
        self.records: List[Tuple[float, float, str, float]] = []  # Time, Price, Action(JP), Profit

        # ポジション
        self.position: Optional[str] = None  # 'long' or 'short'
        self.entry_price: Optional[float] = None

        # 観測次元: [price_z, ma_z, vol_z, rsi/100, dprice_z, log1p(dv)_z]
        self.obs_dim = 6
        self.agent = PPOLite(obs_dim=self.obs_dim, batch_size=update_every)

        self._load_or_init_model()

    # ---- モデル I/O ----
    def _load_or_init_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        created = False
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location="cpu")
                self.agent.net.load_state_dict(state["model"])  # type: ignore
                print("既存モデルを読み込みました:", self.model_path)
            except Exception as e:
                print("既存モデルが無効のため新規作成します:", e)
                created = True
        else:
            print("有効な既存モデルが見つからないため新規作成します。")
            created = True
        if created:
            torch.save({"model": self.agent.net.state_dict()}, self.model_path)
            print("No valid existing model found — created a new model.")

    def _save_model(self):
        torch.save({"model": self.agent.net.state_dict()}, self.model_path)
        print("Model saved to", self.model_path)

    # ---- 観測生成 ----
    def _build_obs(self, price: float, volume: float) -> np.ndarray:
        self.prices.append(price)
        self.volumes.append(volume)

        # Δvolume -> log1p
        if len(self.volumes) >= 2:
            dvol = max(0.0, self.volumes[-1] - self.volumes[-2])
        else:
            dvol = 0.0
        dv_log = math.log1p(dvol)

        arr = np.array(self.prices, dtype=np.float32)
        ma = float('nan');
        vol = float('nan')
        if len(arr) >= self.window_ma:
            ma = float(arr[-self.window_ma:].mean())
        if len(arr) >= self.window_vol:
            vol = float(arr[-self.window_vol:].std(ddof=0))
        rsi = rsi_from_prices(arr, self.window_rsi)
        dprice = float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0

        # z-score（価格系列で近似スケール）
        zs = arr[-self.zscore_window:]
        m = float(zs.mean()) if len(zs) else price
        s = float(zs.std(ddof=0) + 1e-6)

        price_z = zscore(price, m, s)
        ma_z = 0.0 if math.isnan(ma) else zscore(ma, m, s)
        vol_z = 0.0 if math.isnan(vol) else vol / (s + 1e-6)
        rsi_scaled = 0.0 if math.isnan(rsi) else (rsi / 100.0)
        dprice_z = dprice / (s + 1e-6)
        dv_z = (dv_log - 1.0)  # 簡易中心化

        return np.array([price_z, ma_z, vol_z, rsi_scaled, dprice_z, dv_z], dtype=np.float32)

    # ---- PnL ----
    def _unrealized_pnl(self, price: float) -> float:
        if self.position is None or self.entry_price is None:
            return 0.0
        if self.position == 'long':
            return (price - self.entry_price) * self.lot
        else:
            return (self.entry_price - price) * self.lot

    # ---- 約定/損益反映 ----
    def _apply_action(self, ts: float, price: float, act_idx: int, force: bool = False) -> str:
        # act_idx: 0=買い, 1=売り, 2=ホールド
        action_name = "ホールド"
        realized = 0.0

        if act_idx == 0:  # 買い
            if self.position is None:
                # 新規買い（price+tick）
                self.position = 'long'
                self.entry_price = price + self.tick
                action_name = "買い"
            elif self.position == 'short':
                # 返済買い（price+tick）
                exit_price = price + self.tick
                realized = (self.entry_price - exit_price) * self.lot
                self.position = None
                self.entry_price = None
                action_name = "買い"  # 3アクション表記に統一
        elif act_idx == 1:  # 売り
            if self.position is None:
                # 新規売り（price-tick）
                self.position = 'short'
                self.entry_price = price - self.tick
                action_name = "売り"
            elif self.position == 'long':
                # 返済売り（price-tick）
                exit_price = price - self.tick
                realized = (exit_price - self.entry_price) * self.lot
                self.position = None
                self.entry_price = None
                action_name = "売り"  # 3アクション表記に統一
        else:
            action_name = "ホールド"

        self.records.append((ts, price, action_name, realized))
        return action_name

    # ---- 公開API ----
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        obs = self._build_obs(price, volume)

        # 観測が未充足ならホールド
        if np.isnan(obs).any():
            action_idx = 2
            self._apply_action(ts, price, action_idx)
            return "ホールド"

        # 強制返済なら、建玉に応じて反対アクションを選ぶ
        if force_close and self.position is not None:
            action_idx = 0 if self.position == 'short' else 1
            logits, val = self.agent.net(torch.tensor(obs).float().unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor([action_idx])).item()
            value = float(val.item())
            action_name = self._apply_action(ts, price, action_idx, force=True)
            realized = self.records[-1][3]
            reward = self.unreal_w * self._unrealized_pnl(price) + realized
            self.agent.push(obs, action_idx, logp, value, reward, 1.0)
            self.agent.maybe_update()
            return action_name

        # 方策からサンプル
        act_idx, logp, val = self.agent.policy(obs)
        action_name = self._apply_action(ts, price, act_idx)

        realized = self.records[-1][3]
        reward = self.unreal_w * self._unrealized_pnl(price) + realized
        self.agent.push(obs, act_idx, logp, val, reward, 0.0)
        self.agent.maybe_update()
        return action_name

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records, columns=["Time", "Price", "Action", "Profit"])
        self._save_model()
        self.records.clear()
        return df


# ---------------- 使い方（例） ----------------
"""
# 別プログラム側
sim = TradingSimulation()
for i, row in df.iterrows():
    ts = row["Time"]; price = row["Price"]; volume = row["Volume"]
    force_close = (i == len(df) - 1)
    action = sim.add(ts, price, volume, force_close=force_close)

result = sim.finalize()
print(result.tail())
"""

if __name__ == "__main__":
    # ダミーデータの簡易動作テスト
    rng = np.random.default_rng(0)
    n = 2000
    base = 5000.0
    prices = np.round(base + np.cumsum(rng.normal(0, 2, size=n)), 0)
    vols = np.cumsum(np.maximum(0, rng.poisson(1500, size=n)).astype(float))
    times = np.arange(n).astype(float)

    sim = TradingSimulation()
    for i in range(n):
        ts = float(times[i])
        price = float(prices[i])
        volume = float(vols[i])
        force = (i == n - 1)
        sim.add(ts, price, volume, force_close=force)
    df = sim.finalize()
    print(df.tail())
# -*- coding: utf-8 -*-
"""
PPO-lite デイトレ売買シミュレーション（ティック1秒・単一銘柄・3アクション）

要求仕様:
- 外部プログラムから毎秒 `add(ts, price, volume, force_close=False)` を呼び出す
- 生ティックから特徴量を内部で生成
  * Δvolume = max(0, volume_t - volume_{t-1})
  * log1p(Δvolume) で圧縮
  * 価格ベース: MA(10), Volatility(10), RSI(14)
  * 追加: price z-score, dprice z-score
- 信用売買・呼び値=1円・売買単位=100株・ナンピン禁止（常に最大1単位の建玉）
  * 新規買い/返済買いは `price + 1`
  * 新規売り/返済売りは `price - 1`
- アクションは 3 種: [買い, 売り, ホールド]
  * ノーポジ時: 買い→新規買い / 売り→新規売り
  * 買い持ち時: 売り→返済売り / 買い→ホールド
  * 売り持ち時: 買い→返済買い / 売り→ホールド
- 報酬: 確定益 + α×含み益（α=0.1 デフォルト）
- 既存モデルのロード/保存/検証と標準出力
- `finalize()` で結果 DataFrame（Time, Price, Action, Profit）を返し、記録をリセット

依存: torch, numpy, pandas  （CPUのみでOK）
"""

# from __future__ import annotations
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


# ---------------- ユーティリティ ----------------

def rsi_from_prices(prices: np.ndarray, window: int = 14) -> float:
    if len(prices) < window + 1:
        return float('nan')
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


# ---------------- PPO-Lite ----------------

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
            gamma: float = 0.95,
            clip_eps: float = 0.2,
            vf_coef: float = 0.5,
            ent_coef: float = 0.01,
            update_epochs: int = 4,
            batch_size: int = 512,
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
        self.buffer: List[Transition] = []

    def policy(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
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

    def _compute_returns_adv(self):
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return returns, adv, acts, logps_old

    def maybe_update(self):
        if len(self.buffer) < self.batch_size:
            return
        returns, adv, acts, logps_old = self._compute_returns_adv()
        obs = torch.stack([t.obs for t in self.buffer]).to(self.device)
        acts = acts.to(self.device)
        logps_old = logps_old.to(self.device)
        returns = returns.to(self.device)
        adv = adv.to(self.device)

        ds = torch.utils.data.TensorDataset(obs, acts, logps_old, returns, adv)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        self.net.train()
        for _ in range(self.update_epochs):
            for b_obs, b_act, b_logp_old, b_ret, b_adv in loader:
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

        self.buffer.clear()


# ---------------- シミュレーション本体 ----------------

class TradingSimulation:
    """3アクション = [買い(0), 売り(1), ホールド(2)] の軽量PPOトレーナー内蔵シミュレータ"""

    def __init__(
            self,
            lot_size: int = 100,
            tick_size: float = 1.0,
            unrealized_weight: float = 0.1,
            update_every: int = 512,
            model_path: str = "models/ppo_trader_3act.pth",
            feature_window_ma: int = 10,
            feature_window_vol: int = 10,
            feature_window_rsi: int = 14,
            zscore_window: int = 200,
            seed: int = 42,
    ):
        np.random.seed(seed);
        torch.manual_seed(seed)

        self.lot = lot_size
        self.tick = tick_size
        self.unreal_w = unrealized_weight
        self.model_path = model_path
        self.update_every = update_every

        # ローリングバッファ
        self.prices = deque(maxlen=max(feature_window_ma, feature_window_vol, feature_window_rsi, zscore_window) + 2)
        self.volumes = deque(maxlen=4)

        self.window_ma = feature_window_ma
        self.window_vol = feature_window_vol
        self.window_rsi = feature_window_rsi
        self.zscore_window = zscore_window

        # 結果レコード
        self.records: List[Tuple[float, float, str, float]] = []  # Time, Price, Action(JP), Profit

        # ポジション
        self.position: Optional[str] = None  # 'long' or 'short'
        self.entry_price: Optional[float] = None

        # 観測次元: [price_z, ma_z, vol_z, rsi/100, dprice_z, log1p(dv)_z]
        self.obs_dim = 6
        self.agent = PPOLite(obs_dim=self.obs_dim, batch_size=update_every)

        self._load_or_init_model()

    # ---- モデル I/O ----
    def _load_or_init_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        created = False
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location="cpu")
                self.agent.net.load_state_dict(state["model"])  # type: ignore
                print("既存モデルを読み込みました:", self.model_path)
            except Exception as e:
                print("既存モデルが無効のため新規作成します:", e)
                created = True
        else:
            print("有効な既存モデルが見つからないため新規作成します。")
            created = True
        if created:
            torch.save({"model": self.agent.net.state_dict()}, self.model_path)
            print("No valid existing model found — created a new model.")

    def _save_model(self):
        torch.save({"model": self.agent.net.state_dict()}, self.model_path)
        # print("Model saved to", self.model_path)

    # ---- 観測生成 ----
    def _build_obs(self, price: float, volume: float) -> np.ndarray:
        self.prices.append(price)
        self.volumes.append(volume)

        # Δvolume -> log1p
        if len(self.volumes) >= 2:
            dvol = max(0.0, self.volumes[-1] - self.volumes[-2])
        else:
            dvol = 0.0
        dv_log = math.log1p(dvol)

        arr = np.array(self.prices, dtype=np.float32)
        ma = float('nan');
        vol = float('nan')
        if len(arr) >= self.window_ma:
            ma = float(arr[-self.window_ma:].mean())
        if len(arr) >= self.window_vol:
            vol = float(arr[-self.window_vol:].std(ddof=0))
        rsi = rsi_from_prices(arr, self.window_rsi)
        dprice = float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0

        # z-score（価格系列で近似スケール）
        zs = arr[-self.zscore_window:]
        m = float(zs.mean()) if len(zs) else price
        s = float(zs.std(ddof=0) + 1e-6)

        price_z = zscore(price, m, s)
        ma_z = 0.0 if math.isnan(ma) else zscore(ma, m, s)
        vol_z = 0.0 if math.isnan(vol) else vol / (s + 1e-6)
        rsi_scaled = 0.0 if math.isnan(rsi) else (rsi / 100.0)
        dprice_z = dprice / (s + 1e-6)
        dv_z = (dv_log - 1.0)  # 簡易中心化

        return np.array([price_z, ma_z, vol_z, rsi_scaled, dprice_z, dv_z], dtype=np.float32)

    # ---- PnL ----
    def _unrealized_pnl(self, price: float) -> float:
        if self.position is None or self.entry_price is None:
            return 0.0
        if self.position == 'long':
            return (price - self.entry_price) * self.lot
        else:
            return (self.entry_price - price) * self.lot

    # ---- 約定/損益反映 ----
    def _apply_action(self, ts: float, price: float, act_idx: int, force: bool = False) -> str:
        # act_idx: 0=買い, 1=売り, 2=ホールド
        action_name = "ホールド"
        realized = 0.0

        if act_idx == 0:  # 買い
            if self.position is None:
                # 新規買い（price+tick）
                self.position = 'long'
                self.entry_price = price + self.tick
                action_name = "買い"
            elif self.position == 'short':
                # 返済買い（price+tick）
                exit_price = price + self.tick
                realized = (self.entry_price - exit_price) * self.lot
                self.position = None
                self.entry_price = None
                action_name = "買い"  # 3アクション表記に統一
        elif act_idx == 1:  # 売り
            if self.position is None:
                # 新規売り（price-tick）
                self.position = 'short'
                self.entry_price = price - self.tick
                action_name = "売り"
            elif self.position == 'long':
                # 返済売り（price-tick）
                exit_price = price - self.tick
                realized = (exit_price - self.entry_price) * self.lot
                self.position = None
                self.entry_price = None
                action_name = "売り"  # 3アクション表記に統一
        else:
            action_name = "ホールド"

        self.records.append((ts, price, action_name, realized))
        return action_name

    # ---- 公開API ----
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        obs = self._build_obs(price, volume)

        # 観測が未充足ならホールド
        if np.isnan(obs).any():
            action_idx = 2
            self._apply_action(ts, price, action_idx)
            return "ホールド"

        # 強制返済なら、建玉に応じて反対アクションを選ぶ
        if force_close and self.position is not None:
            action_idx = 0 if self.position == 'short' else 1
            logits, val = self.agent.net(torch.tensor(obs).float().unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(torch.tensor([action_idx])).item()
            value = float(val.item())
            action_name = self._apply_action(ts, price, action_idx, force=True)
            realized = self.records[-1][3]
            reward = self.unreal_w * self._unrealized_pnl(price) + realized
            self.agent.push(obs, action_idx, logp, value, reward, 1.0)
            self.agent.maybe_update()
            return action_name

        # 方策からサンプル
        act_idx, logp, val = self.agent.policy(obs)
        action_name = self._apply_action(ts, price, act_idx)

        realized = self.records[-1][3]
        reward = self.unreal_w * self._unrealized_pnl(price) + realized
        self.agent.push(obs, act_idx, logp, val, reward, 0.0)
        self.agent.maybe_update()
        return action_name

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records, columns=["Time", "Price", "Action", "Profit"])
        self._save_model()
        self.records.clear()
        return df


# ---------------- 使い方（例） ----------------
"""
# 別プログラム側
sim = TradingSimulation()
for i, row in df.iterrows():
    ts = row["Time"]; price = row["Price"]; volume = row["Volume"]
    force_close = (i == len(df) - 1)
    action = sim.add(ts, price, volume, force_close=force_close)

result = sim.finalize()
print(result.tail())
"""

if __name__ == "__main__":
    # ダミーデータの簡易動作テスト
    rng = np.random.default_rng(0)
    n = 2000
    base = 5000.0
    prices = np.round(base + np.cumsum(rng.normal(0, 2, size=n)), 0)
    vols = np.cumsum(np.maximum(0, rng.poisson(1500, size=n)).astype(float))
    times = np.arange(n).astype(float)

    sim = TradingSimulation()
    for i in range(n):
        ts = float(times[i])
        price = float(prices[i])
        volume = float(vols[i])
        force = (i == n - 1)
        sim.add(ts, price, volume, force_close=force)
    df = sim.finalize()
    print(df.tail())
