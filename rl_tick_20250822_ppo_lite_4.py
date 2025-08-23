# rl_trading_simulation_detach_fix.py
# -*- coding: utf-8 -*-
"""
PPO-lite Actor-Critic（detach 修正入り）
- 1銘柄、約1秒ティック、100株単位、同時1ポジ（ナンピン不可）
- add(ts, price, volume, force_close=False) を逐次呼び出し
- finalize() で結果 DataFrame を返して内部結果バッファをリセット（モデルは継続）
- 出来高は add() 内で 差分→np.log1p→オンライン正規化
- “ゼロ損益の無駄往復”を抑える報酬シェイピング（紙コスト/近接ゼロPNLペナ/早期クローズペナ/スイッチングコスト）を実装
- detach により、保存する logp/value が計算グラフを保持しないように修正
"""

from __future__ import annotations
import os
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# ユーティリティ（オンライン正規化）
# ============================================================
class OnlineNormalizer:
    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        var = self.M2 / (self.n - 1)
        return math.sqrt(var + self.eps)

    def normalize(self, x: float) -> float:
        if self.n < 2:
            return 0.0
        return (x - self.mean) / self.std


# ============================================================
# ネットワーク（軽量）
# ============================================================
class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, 3)  # 0=hold, 1=buy/close short, 2=sell/close long
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


# メモリエントリ（学習用）
class Transition:
    __slots__ = ("obs", "action", "logp", "reward", "done", "value")
    def __init__(self, obs: np.ndarray, action: int, logp: float, reward: float, done: bool, value: float):
        self.obs = obs
        self.action = action
        self.logp = logp       # detach済みの float
        self.reward = reward
        self.done = done
        self.value = value     # detach済みの float


# ============================================================
# シミュレーション + PPO-lite
# ============================================================
class TradingSimulation:
    def __init__(self,
                 model_path: str = "models/policy.pth",
                 device: Optional[str] = None,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 lr: float = 3e-4,
                 update_interval: int = 128,
                 mini_epochs: int = 4,
                 batch_size: int = 64,
                 seed: int = 42):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 観測: price_z, ret1, ret5, ret10, log1p_dv_z, pos_long, pos_short
        self.obs_dim = 7
        self.net = PolicyValueNet(self.obs_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # オンライン正規化
        self.price_norm = OnlineNormalizer()
        self.logdv_norm = OnlineNormalizer()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_interval = update_interval
        self.mini_epochs = mini_epochs
        self.batch_size = batch_size

        # 報酬シェイピング（ゼロ往復抑制）
        self.paper_fee_per_round = 10.0     # 仮想往復コスト(円/100株)
        self.zero_close_thresh   = 50.0     # 近接ゼロとみなす確定損益の閾値
        self.zero_close_penalty  = 0.15     # 近接ゼロPNLクローズのペナルティ
        self.min_hold_secs       = 10.0     # 最短保有時間
        self.early_close_penalty = 0.10     # 最短未満かつ小PNLクローズのペナルティ
        self.switching_cost      = 0.03     # 解消/反転コスト

        self.reset_state()
        self.buffer: List[Transition] = []

        self._load_or_init_model()

    # ------------------------- 公開API -------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        action_label = "ホールド"

        # 差分出来高 → log1p → 正規化
        delta_v = max(volume - self.prev_volume if self.prev_volume is not None else 0.0, 0.0)
        self.prev_volume = volume
        logdv = math.log1p(delta_v)
        self.logdv_norm.update(logdv)
        logdv_z = self.logdv_norm.normalize(logdv)

        # 価格正規化とリターン
        self.price_norm.update(price)
        pz = self.price_norm.normalize(price)

        ret1 = 0.0 if self.prev_price is None else (price - self.prev_price) / max(self.prev_price, 1e-6)
        self.prev_price = price
        self.ret_buf.append(price)
        if len(self.ret_buf) > 10:
            self.ret_buf.pop(0)
        ret5  = 0.0 if len(self.ret_buf) < 6 else (self.ret_buf[-1] - self.ret_buf[-6]) / max(self.ret_buf[-6], 1e-6)
        ret10 = 0.0 if len(self.ret_buf) < 11 else (self.ret_buf[-1] - self.ret_buf[0]) / max(self.ret_buf[0], 1e-6)

        pos_long = 1.0 if self.position == 1 else 0.0
        pos_short = 1.0 if self.position == -1 else 0.0
        obs = np.array([pz, ret1, ret5, ret10, logdv_z, pos_long, pos_short], dtype=np.float32)

        # 方策サンプル
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value_t = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        action_t = dist.sample()
        logp_t = dist.log_prob(action_t)

        action = int(action_t.item())
        value = float(value_t.item())       # detach: float化
        logp  = float(logp_t.item())        # detach: float化

        # 取引の実行と損益
        realized = 0.0
        if self.position == 0:
            if action == 1:
                self.entry_price = price
                self.position = 1
                self.last_entry_ts = ts
                action_label = "新規買い"
            elif action == 2:
                self.entry_price = price
                self.position = -1
                self.last_entry_ts = ts
                action_label = "新規売り"
            else:
                action_label = "ホールド"
        elif self.position == 1:
            if action == 2:
                realized = (price - self.entry_price) * 100.0
                self.position = 0
                self.entry_price = None
                action_label = "返済売り"
            else:
                action_label = "ホールド"
        elif self.position == -1:
            if action == 1:
                realized = (self.entry_price - price) * 100.0
                self.position = 0
                self.entry_price = None
                action_label = "返済買い"
            else:
                action_label = "ホールド"

        # 含み損益
        unrealized = 0.0
        if self.position != 0 and self.entry_price is not None:
            unrealized = (price - self.entry_price) * 100.0 if self.position == 1 else (self.entry_price - price) * 100.0

            # 自動クローズ（利確/損切 or 強制）
            if unrealized >= 200.0 or unrealized <= -500.0 or force_close:
                if self.position == 1:
                    realized = (price - self.entry_price) * 100.0
                    action_label = "返済売り(自動)" if not force_close else "返済売り(強制)"
                else:
                    realized = (self.entry_price - price) * 100.0
                    action_label = "返済買い(自動)" if not force_close else "返済買い(強制)"
                self.position = 0
                self.entry_price = None

        # 報酬
        reward = 0.0
        # 含み損益に基づく基本報酬（スケール調整）
        reward += (unrealized / 1000.0)
        if unrealized >= 1000.0:
            reward *= 2.0
        elif unrealized >= 500.0:
            reward *= 1.5
        elif unrealized >= 200.0:
            reward *= 1.0
        if unrealized > 0.0:
            reward += 0.05
        if unrealized < 0.0:
            reward -= 0.05

        # 返済時の加点 + シェイピング
        if action_label.startswith("返済"):
            # 仮想往復コスト（学習バイアス用）
            realized_adj = realized - self.paper_fee_per_round
            reward += (realized_adj / 1000.0)

            # 近接ゼロPNLペナルティ
            if abs(realized) < self.zero_close_thresh:
                reward -= self.zero_close_penalty

            # 最短保有時間ペナルティ
            if self.last_entry_ts is not None:
                hold_secs = float(ts) - float(self.last_entry_ts)
                if hold_secs < self.min_hold_secs and abs(realized) < self.zero_close_thresh:
                    reward -= self.early_close_penalty

            # 切替/解消コスト
            reward -= self.switching_cost

        # 結果レコード
        self.results.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": action_label,
            "Profit": float(realized),
        })

        # バッファ（detach済みの float を保持）
        self.buffer.append(Transition(obs=obs, action=action, logp=logp, reward=float(reward), done=False, value=value))

        # 学習トリガ
        if len(self.buffer) >= self.update_interval:
            self._update()
            self._save_model()

        return action_label

    def finalize(self) -> pd.DataFrame:
        # エピソード終端フラグ
        if len(self.buffer) > 0:
            last = self.buffer[-1]
            self.buffer[-1] = Transition(last.obs, last.action, last.logp, last.reward, True, last.value)
            self._update()
            self._save_model()

        df = pd.DataFrame(self.results, columns=["Time", "Price", "売買アクション", "Profit"]).copy()
        self.results.clear()
        return df

    # ------------------------- 内部処理 -------------------------
    def reset_state(self):
        self.position: int = 0
        self.entry_price: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None
        self.ret_buf: List[float] = []
        self.last_entry_ts: Optional[float] = None
        self.results: List[dict] = []

    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                if isinstance(state, dict) and "model" in state:
                    self.net.load_state_dict(state["model"])  # 当実装の保存形式に合わせる
                    self.opt.load_state_dict(state.get("opt", self.opt.state_dict()))
                else:
                    self.net.load_state_dict(state)
                print("既存モデルを読み込みました。")
            except Exception as e:
                print(f"既存モデルの読み込みに失敗: {e}. 新規作成して上書きします。")
                self._save_model()
        else:
            print("有効な既存モデルが見つからないため、新規モデルを作成しました。")
            self._save_model()

    def _save_model(self):
        try:
            torch.save({"model": self.net.state_dict(), "opt": self.opt.state_dict()}, self.model_path)
        except Exception:
            # 後方互換：シンプル保存
            torch.save(self.net.state_dict(), self.model_path)

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def _update(self):
        if not self.buffer:
            return
        # ------- numpy へ一括変換（Warning回避 & 高速化） -------
        obs_np = np.stack([t.obs for t in self.buffer]).astype(np.float32)
        act_np = np.array([t.action for t in self.buffer], dtype=np.int64)
        oldlp_np = np.array([t.logp for t in self.buffer], dtype=np.float32)
        rew_np = np.array([t.reward for t in self.buffer], dtype=np.float32)
        done_np = np.array([t.done for t in self.buffer], dtype=np.bool_)

        # 現ネットで value 再評価
        with torch.no_grad():
            x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            _, v = self.net(x)
            val_np = v.detach().cpu().numpy().astype(np.float32)

        adv_np, ret_np = self._compute_gae(rew_np, val_np, done_np)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        n = len(obs_np)
        idx = np.arange(n)
        np.random.shuffle(idx)

        for _ in range(self.mini_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                bidx = idx[start:start + self.batch_size]
                bx = torch.tensor(obs_np[bidx], dtype=torch.float32, device=self.device)
                ba = torch.tensor(act_np[bidx], dtype=torch.long, device=self.device)
                boldlp = torch.tensor(oldlp_np[bidx], dtype=torch.float32, device=self.device)
                badv = torch.tensor(adv_np[bidx], dtype=torch.float32, device=self.device)
                bret = torch.tensor(ret_np[bidx], dtype=torch.float32, device=self.device)

                logits, value = self.net(bx)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(ba)
                ratio = torch.exp(new_logp - boldlp)

                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * badv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = F.mse_loss(value, bret)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.opt.step()

        # 学習後、バッファはクリア（同じグラフを再利用しない）
        self.buffer.clear()


# 簡易テスト（任意実行）
if __name__ == "__main__":
    sim = TradingSimulation()
    t = 0.0
    price = 5000.0
    volume = 0.0
    for i in range(1000):
        t += 1.0
        price += np.random.normal(0, 2)
        volume += max(0.0, np.random.poisson(50))
        action = sim.add(t, float(price), float(volume), force_close=(i == 999))
    df = sim.finalize()
    print(df.tail())
