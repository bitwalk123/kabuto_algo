# -*- coding: utf-8 -*-
"""
PPO-lite強化学習・ティック売買シミュレーション
-------------------------------------------------
- 条件:
  * 特定1銘柄、ティック(約1秒)データ [ts, price, volume]
  * 売買単位: 100株、ナンピン無し(常に最大1建玉)
  * 指値成行同等(現在価格で約定)、手数料考慮なし
  * 強制返済フラグで最終バーで建玉解消
  * モデル: 軽量PPO(Actor-Critic + クリップド比)
  * 体裁:
      - TradingSimulation.add(ts, price, volume, force_close=False) -> action(str)
      - TradingSimulation.finalize() -> pandas.DataFrame(結果)
      - モデル自動ロード/保存
  * PyTorch 2.8想定

- 収益/報酬設計:
  * 逐次報酬: 含み損益の変化分(ΔPnL)
  * 返済時ボーナス: 実現益が +300円/100株 以上で +1.0 のボーナス、
                    0円<=利益<300円 では +0.2、損失は -0.5
  * 小さなホールドペナルティ(-0.0001)で無駄な滞留を抑制

- 状態量(オンライン前処理):
  * リターン: r_t = (price_t - price_{t-1}) / price_{t-1}
  * 差分出来高: dv = max(volume_t - volume_{t-1}, 0.0)
  * ログ圧縮: v_sig = log1p(dv)
  * ポジション: pos in {-1(売建), 0(ノーポジ), +1(買建)}
  * すべて逐次ランニング標準化(平均0/分散1)でスケール調整

- アクション:
  * policyは3値 {0: SELL, 1: HOLD, 2: BUY}
  * ただし状態により無効アクションはHOLDへフォールバック
      - pos == 0: BUY/SELL/ HOLD
      - pos == +1: {CLOSE(=SELL), HOLD}
      - pos == -1: {CLOSE(=BUY), HOLD}

- 結果ログ:
  * 列: [Time, Price, 売買アクション, Profit]
  * 毎ティック追記、Profitは返済時のみ確定額(円)が入る

使い方:
  sim = TradingSimulation()
  action = sim.add(ts, price, volume, force_close=(i==last))
  df_result = sim.finalize()  # エポックごと

"""

from __future__ import annotations
import os
import math
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ==============================
# ユーティリティ: ランニング標準化
# ==============================
class RunningNorm:
    """逐次的に平均と分散を推定し、入力を標準化する。
    数値安定性のためにWelford法を使用。"""

    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float) -> float:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        return self.normalize(x)

    def normalize(self, x: float) -> float:
        var = self.variance
        std = math.sqrt(var + self.eps)
        return (x - self.mean) / std if std > 0 else 0.0

    @property
    def variance(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0


# ==============================
# ニューラルネット(Actor-Critic)
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)


# ==============================
# 軽量PPO(学習器)
# ==============================
@dataclass
class PPOConfig:
    gamma: float = 0.999  # 1秒足想定の高割引(ほぼ無割引)
    lam: float = 0.95  # GAE-lambda
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    update_epochs: int = 3
    minibatch_size: int = 1024
    update_interval: int = 4096  # これだけサンプルが溜まったら更新(または強制返済時)


class PPOLite:
    def __init__(self, state_dim: int, action_dim: int = 3, cfg: PPOConfig = PPOConfig(), device: str = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(state_dim, hidden_dim=64, action_dim=action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.clear_buffer()

    # 経験バッファ
    def clear_buffer(self):
        self.obs: List[np.ndarray] = []
        self.acts: List[int] = []
        self.logps: List[float] = []
        self.rews: List[float] = []
        self.dones: List[bool] = []
        self.vals: List[float] = []

    @torch.no_grad()
    def policy(self, state: np.ndarray) -> Tuple[int, float, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        dist = Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)
        return int(act.item()), float(logp.item()), float(value.item())

    def store(self, state: np.ndarray, act: int, logp: float, rew: float, done: bool, val: float):
        self.obs.append(state.astype(np.float32))
        self.acts.append(act)
        self.logps.append(logp)
        self.rews.append(rew)
        self.dones.append(done)
        self.vals.append(val)

    def _compute_gae(self, last_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n = len(self.rews)
        adv = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            nextnonterminal = 1.0 - float(self.dones[t])
            nextval = self.vals[t + 1] if t + 1 < n else last_val
            delta = self.rews[t] + cfg.gamma * nextval * nextnonterminal - self.vals[t]
            lastgaelam = delta + cfg.gamma * cfg.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + np.array(self.vals, dtype=np.float32)
        # 正規化
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        return adv_t, ret_t

    def maybe_update(self, last_val: float = 0.0):
        if len(self.rews) < self.cfg.update_interval:
            return
        self.update(last_val)

    def update(self, last_val: float = 0.0):
        cfg = self.cfg

        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.acts, dtype=torch.int64, device=self.device)
        old_logps = torch.tensor(self.logps, dtype=torch.float32, device=self.device)
        adv, ret = self._compute_gae(last_val)

        dataset_size = obs.size(0)
        idxs = np.arange(dataset_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, cfg.minibatch_size):
                mb_idx = idxs[start:start + cfg.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                ratio = (new_logps - mb_old_logps).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_ret)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

        self.clear_buffer()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        # cfgは既存のをそのまま使用


# ==============================
# 売買ロジック + RL結合クラス
# ==============================
class TradingSimulation:
    ACTION_SELL = 0
    ACTION_HOLD = 1
    ACTION_BUY = 2

    def __init__(self,
                 shares_per_trade: int = 100,
                 model_path: str = "models/ppo_lite_trader.pt",
                 update_interval: int | None = None,
                 seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.shares = shares_per_trade
        self.model_path = model_path

        # 状態: [ret_norm, v_sig_norm, pos]
        self.ret_norm = RunningNorm()
        self.vol_norm = RunningNorm()

        self.prev_price = None
        self.prev_volume = None
        self.position = 0  # -1: 売建, 0: ノーポジ, +1: 買建
        self.entry_price = None

        self.total_realized = 0.0
        self.last_unrealized = 0.0

        # RL本体
        self.cfg = PPOConfig()
        if update_interval is not None:
            self.cfg.update_interval = update_interval
        self.agent = PPOLite(state_dim=3, action_dim=3, cfg=self.cfg)
        if os.path.exists(self.model_path):
            try:
                self.agent.load(self.model_path)
            except Exception as e:
                print("モデル読み込み失敗。新規作成します:", e)

        # 結果ログ
        self.rows: List[dict] = []

    # ==========================
    # 内部ヘルパ
    # ==========================
    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        # 100株単位
        if self.position > 0:
            return (price - self.entry_price) * self.shares
        else:
            return (self.entry_price - price) * self.shares

    def _realize(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        pnl = self._unrealized_pnl(price)
        self.total_realized += pnl
        # ポジ解消
        self.position = 0
        self.entry_price = None
        return pnl

    def _select_action(self, state: np.ndarray) -> int:
        act, logp, val = self.agent.policy(state)
        self._last_logp = logp
        self._last_val = val
        return act

    def _adapt_action(self, raw_act: int) -> Tuple[str, int]:
        # 状態に応じて実行可能アクションへ射影
        if self.position == 0:
            if raw_act == self.ACTION_BUY:
                return "新規買い", self.ACTION_BUY
            elif raw_act == self.ACTION_SELL:
                return "新規売り", self.ACTION_SELL
            else:
                return "ホールド", self.ACTION_HOLD
        elif self.position > 0:  # ロング中 -> 売って返済 or ホールド
            if raw_act == self.ACTION_SELL:
                return "返済売り", self.ACTION_SELL
            else:
                return "ホールド", self.ACTION_HOLD
        else:  # ショート中 -> 買って返済 or ホールド
            if raw_act == self.ACTION_BUY:
                return "返済買い", self.ACTION_BUY
            else:
                return "ホールド", self.ACTION_HOLD

    def _step_env(self, price: float, action_str: str) -> Tuple[float, float]:
        """環境遷移と報酬計算。
        戻り: (reward, realized_if_any)
        """
        realized = 0.0
        reward = 0.0

        # 逐次報酬: 含み損益の増分(ΔPnL)
        unreal = self._unrealized_pnl(price)
        delta_pnl = unreal - self.last_unrealized
        reward += delta_pnl / 1000.0  # スケール緩和
        self.last_unrealized = unreal

        # 小さなホールドペナルティ
        if action_str == "ホールド":
            reward -= 0.0001

        # 返済/新規の反映
        if action_str == "返済売り":
            realized = self._realize(price)
            # 実現益ボーナス
            if realized >= 300.0:
                reward += 1.0
            elif realized >= 0.0:
                reward += 0.2
            else:
                reward -= 0.5
        elif action_str == "返済買い":
            realized = self._realize(price)
            if realized >= 300.0:
                reward += 1.0
            elif realized >= 0.0:
                reward += 0.2
            else:
                reward -= 0.5
        elif action_str == "新規買い":
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.last_unrealized = 0.0
        elif action_str == "新規売り":
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.last_unrealized = 0.0

        return reward, realized

    # ==========================
    # パブリックAPI
    # ==========================
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """1ティック分をシミュレーション。呼び出し毎にアクション文字列を返す。"""
        # 特徴量生成
        if self.prev_price is None:
            ret = 0.0
            v_sig = 0.0
        else:
            ret = (price - self.prev_price) / max(self.prev_price, 1e-8)
            dv = max(volume - (self.prev_volume or 0.0), 0.0)
            v_sig = math.log1p(dv)

        ret_n = self.ret_norm.update(ret)
        vol_n = self.vol_norm.update(v_sig)
        state = np.array([ret_n, vol_n, float(self.position)], dtype=np.float32)

        raw_act = self._select_action(state)
        action_str, exec_act = self._adapt_action(raw_act)

        # 強制返済フラグがONなら、現在の状態に応じて返済へ上書き
        if force_close and self.position != 0:
            action_str = "返済売り" if self.position > 0 else "返済買い"

        # 環境遷移&報酬
        reward, realized = self._step_env(price, action_str)

        # バッファ保存
        done = bool(force_close)
        self.agent.store(state, exec_act, self._last_logp, reward, done, self._last_val)
        # 必要に応じて更新
        last_val_bootstrap = self.agent._last_val if hasattr(self.agent, "_last_val") else 0.0
        self.agent.maybe_update(last_val=last_val_bootstrap)

        # ログ
        self.rows.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": action_str,
            "Profit": float(realized) if (action_str in ("返済買い", "返済売り")) else 0.0,
        })

        # 次準備
        self.prev_price = price
        self.prev_volume = volume

        return action_str

    def finalize(self) -> pd.DataFrame:
        """結果DataFrameを返し、内部の結果だけリセット。モデルは保存する。"""
        # エピソード終端時の学習仕上げ
        if len(self.agent.rews) > 0:
            # 末端の価値近似(ノーポジなら0、ポジションありなら少しだけブートストラップ)
            bootstrap_v = 0.0
            try:
                last_state = torch.tensor(self.agent.obs[-1], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _, v = self.agent.net(last_state)
                    bootstrap_v = float(v.item())
            except Exception:
                pass
            self.agent.update(last_val=bootstrap_v)

        # モデル保存
        self.agent.save(self.model_path)

        # DataFrame生成
        df = pd.DataFrame(self.rows, columns=["Time", "Price", "売買アクション", "Profit"])

        # 内部状態のうち、結果ログのみリセット(モデル・正規化器は継続)
        self.rows = []
        self.total_realized = 0.0
        self.last_unrealized = 0.0
        self.prev_price = None
        self.prev_volume = None
        self.position = 0
        self.entry_price = None
        # ret_norm/vol_normは日跨ぎ学習のため維持

        return df


if __name__ == "__main__":
    # 簡易自己テスト(ダミーデータで形だけ確認)
    sim = TradingSimulation()
    t = 0.0
    price = 1000.0
    vol = 0.0
    for i in range(1000):
        t += 1.0
        price = max(1.0, price + np.random.randn() * 1.0)
        vol += max(0.0, np.random.poisson(5))
        force = (i == 999)
        _ = sim.add(t, price, vol, force_close=force)
    df = sim.finalize()
    print(df.tail())
