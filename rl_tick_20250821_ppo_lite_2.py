# -*- coding: utf-8 -*-
"""
Tick 版・PPO-lite（軽量 Actor-Critic）デイトレード用シミュレーション サンプル

要件:
- Python / PyTorch 2.8 想定
- 別プログラムから 1 秒刻みの生ティック [ts, price, volume(累積)] を add() で受け取る
- 取引対象は 1 銘柄、売買単位 100 株、信用(買い/売り)可、ナンピンなし（同時に 1 建玉のみ）
- 約定手数料は考慮しない
- 学習モデルは保存/読込可能。無効なモデルは自動で作り直し
- 結果 DataFrame: [Time, Price, 売買アクション, Profit]
- finalize() で結果 DF を返し、内部の結果ログをリセット

インターフェイス:
- TradingSimulation.add(ts: float, price: float, volume: float, force_close: bool=False) -> str
    返り値は売買アクション（日本語文字列）
- TradingSimulation.finalize() -> pandas.DataFrame

備考:
- Δ出来高 = max(volume - prev_volume, 0)
- 特徴量で log1p(Δ出来高) を使用
- 価格の対数リターンや短期差分も特徴に採用
- 報酬は含み損益（スケール済）+ 保有インセンティブ + 返済時ボーナス
- クリップ付き PPO（1エポック）で小バッファ逐次学習（"PPO-lite"）

保存ファイル:
- models/ppo_lite.pt

使い方は本ファイル末尾の __main__ のコメントを参照。
"""
from __future__ import annotations
import os
import math
import json
import time
import random
from typing import List, Tuple, Optional, Deque
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ========================= ユーティリティ ========================= #

class RunningScaler:
    """オンラインで平均/分散を更新して正規化するスケーラ。
    Welford 法で数値安定。特徴量のスケール調整に使用。
    """
    def __init__(self, eps: float = 1e-6):
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
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.eps)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# ========================= モデル定義 ========================= #

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, action_dim: int = 3):
        super().__init__()
        # 共有ベース
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Actor（3 アクション: 0=ホールド, 1=買い系, 2=売り系）
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.base(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


# ========================= PPO-lite ========================= #

class PPOLite:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        train_batch: int = 256,
        mini_batch: int = 64,
        update_epochs: int = 2,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.train_batch = train_batch
        self.mini_batch = mini_batch
        self.update_epochs = update_epochs

        self.model = ActorCritic(obs_dim, hidden=128, action_dim=action_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.val_buf: List[float] = []
        self.done_buf: List[bool] = []

    def policy(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item()), float(value.item())

    def store(self, obs: np.ndarray, act: int, logp: float, rew: float, val: float, done: bool):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def _compute_gae(self, last_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """GAE(lambda) で advantage と returns を計算"""
        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf + [last_val], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            delta = rews[t] + self.gamma * vals[t+1] * (1 - dones[t]) - vals[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        # 標準化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def maybe_update(self, last_val: float = 0.0):
        if len(self.obs_buf) < self.train_batch:
            return  # まだ学習しない
        adv, ret = self._compute_gae(last_val)
        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        act = torch.tensor(self.act_buf, dtype=torch.int64, device=self.device)
        old_logp = torch.tensor(self.logp_buf, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.mini_batch):
                end = start + self.mini_batch
                b = idx[start:end]
                logits, value = self.model(obs[b])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(act[b])
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = self.val_buf_to_tensor()[b] + torch.clamp(
                    value - self.val_buf_to_tensor()[b], -self.clip_ratio, self.clip_ratio
                )
                vf_loss1 = (value - ret_t[b]).pow(2)
                vf_loss2 = (value_clipped - ret_t[b]).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
        # バッファをクリア
        self.obs_buf.clear(); self.act_buf.clear(); self.logp_buf.clear()
        self.rew_buf.clear(); self.val_buf.clear(); self.done_buf.clear()

    def val_buf_to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.val_buf, dtype=torch.float32, device=self.device)

    # 保存/読込
    def save(self, path: str, meta: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "meta": meta,
        }, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])  # type: ignore
        self.opt.load_state_dict(ckpt.get("opt", {}))
        return ckpt.get("meta", {})


# ========================= 取引ロジック & 環境 ========================= #

class TradingSimulation:
    """
    別プログラムから 1tick ずつ add() が呼ばれる前提のシミュレータ本体。
    モデル学習は内部で逐次（小バッファ単位）実行し、学習状態はファイルに永続化。
    """

    MODEL_PATH = "models/ppo_lite.pt"
    MODEL_SIGNATURE = {
        "version": 3,
        "obs_dim": 8,   # 特徴量次元を変えたら上げる
        "action_dim": 3
    }

    def __init__(self):
        # 結果ログ
        self.results: List[dict] = []

        # 取引状態
        self.position: int = 0          # +100 = 買い建 100 株, -100 = 売り建 100 株, 0 = ノーポジ
        self.entry_price: Optional[float] = None
        self.lot: int = 100

        # 直近観測
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None  # 累積
        self.prev_ts: Optional[float] = None

        # 特徴量用スケーラ
        self.scaler_price = RunningScaler()
        self.scaler_vol = RunningScaler()

        # エージェント
        obs_dim = self.MODEL_SIGNATURE["obs_dim"]
        self.agent = PPOLite(obs_dim=obs_dim)

        # モデルのロード/バリデーション
        meta = self._try_load_or_init_model()
        print(meta["_message"])  # ロード結果を標準出力

    # =================== API: 別プログラムから呼ばれる =================== #
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """1 ティック受信。行動を返す。"""
        action_str = "ホールド"  # 返すアクション名

        # 初回は観測を蓄積してホールド
        if self.prev_price is None:
            self._update_scalers(price, 0.0)
            self.prev_price = price
            self.prev_volume = volume
            self.prev_ts = ts
            # ログ
            self._log_result(ts, price, action_str, profit=0.0)
            return action_str

        # 観測量の作成
        dt = max(ts - (self.prev_ts or ts), 1.0)
        dprice = price - (self.prev_price or price)
        r1 = math.log(max(price, 1e-6)) - math.log(max(self.prev_price or price, 1e-6))  # 対数リターン
        r2 = dprice / max(self.prev_price or price, 1e-6)  # 相対リターン
        # 出来高差分
        dvol = max(volume - (self.prev_volume or volume), 0.0)
        v1 = math.log1p(dvol)

        # スケーラ更新（価格変化と出来高差分を別々に）
        self._update_scalers(dprice, dvol)
        dprice_n = self.scaler_price.normalize(dprice)
        v1_n = self.scaler_vol.normalize(v1)

        # ポジション特徴
        pos_long = 1.0 if self.position > 0 else 0.0
        pos_short = 1.0 if self.position < 0 else 0.0
        entry_px = float(self.entry_price) if self.entry_price is not None else price
        upnl = self._unrealized_pnl(price)
        upnl_scaled = upnl / 100.0  # スケール調整

        obs = np.array([
            r1, r2, dprice_n, v1_n,
            pos_long, pos_short,
            (price - entry_px) / max(entry_px, 1e-6),  # 現在の含み損益を価格比で
            upnl_scaled,
        ], dtype=np.float32)

        # 行動選択
        act, logp, val = self.agent.policy(obs)

        # 行動の解釈と約定処理
        done = False
        realized = 0.0
        if force_close and self.position != 0:
            # 大引け強制返済
            realized = self._close_position(price)
            action_str = "返済買い" if self.position < 0 else "返済売り"  # _close_position 後では 0 になるので先に参照
            done = True
        else:
            if act == 1:  # 買い系
                if self.position == 0:
                    self._open_long(price)
                    action_str = "新規買い"
                elif self.position < 0:
                    realized = self._close_position(price)
                    action_str = "返済買い"
                else:
                    action_str = "ホールド"
            elif act == 2:  # 売り系
                if self.position == 0:
                    self._open_short(price)
                    action_str = "新規売り"
                elif self.position > 0:
                    realized = self._close_position(price)
                    action_str = "返済売り"
                else:
                    action_str = "ホールド"
            else:
                action_str = "ホールド"

        # 報酬設計
        step_reward = self._compute_reward(upnl=upnl, realized=realized, position=self.position)

        # 学習用に保存
        self.agent.store(obs, act, logp, step_reward, val, done)
        # 可能なら更新
        self.agent.maybe_update(last_val=val)

        # ログ
        self._log_result(ts, price, action_str, profit=realized)

        # 次回用に更新
        self.prev_price = price
        self.prev_volume = volume
        self.prev_ts = ts

        return action_str

    def finalize(self) -> pd.DataFrame:
        """結果 DF を返し、内部ログをリセット。"""
        df = pd.DataFrame(self.results, columns=["Time", "Price", "売買アクション", "Profit"])
        # エージェント状態を保存
        meta = {
            "signature": self.MODEL_SIGNATURE,
            "saved_at": time.time(),
        }
        self.agent.save(self.MODEL_PATH, meta)
        # ログリセット
        self.results = []
        return df

    # =================== 内部関数: 取引管理 =================== #
    def _open_long(self, price: float):
        self.position = self.lot
        self.entry_price = price

    def _open_short(self, price: float):
        self.position = -self.lot
        self.entry_price = price

    def _close_position(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position > 0:
            pnl = (price - self.entry_price) * self.lot
        else:
            pnl = (self.entry_price - price) * self.lot
        self.position = 0
        self.entry_price = None
        return float(pnl)

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position > 0:
            return (price - self.entry_price) * self.lot
        else:
            return (self.entry_price - price) * self.lot

    def _compute_reward(self, upnl: float, realized: float, position: int) -> float:
        # 含み損益に比例（スケール縮小）
        r = upnl / 100.0
        # 保有インセンティブ: 含み益が大きいほど加点（300 円で打ち止め）
        if position != 0 and upnl > 0:
            r += min(upnl / 300.0, 1.0) * 0.5
        # 返済時ボーナス: 目標 300 円以上で +1.0、それ未満は小さめ
        if abs(realized) > 0:
            r += (realized / 100.0)
            if realized >= 300.0:
                r += 1.0
            elif realized > 0:
                r += 0.1
            else:
                r -= 0.1  # 損切りは軽いペナルティ（過学習回避のため過度に重くしない）
        return float(r)

    def _log_result(self, ts: float, price: float, action_str: str, profit: float):
        self.results.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": action_str,
            "Profit": float(profit),
        })

    def _update_scalers(self, dprice: float, dvol: float):
        self.scaler_price.update(float(dprice))
        self.scaler_vol.update(float(math.log1p(max(dvol, 0.0))))

    # =================== モデルのロード/初期化 =================== #
    def _try_load_or_init_model(self) -> dict:
        meta = {"_message": ""}
        path = self.MODEL_PATH
        reason = None
        if os.path.exists(path):
            try:
                loaded_meta = self.agent.load(path)
                sig_ok = isinstance(loaded_meta, dict) and loaded_meta.get("signature") == self.MODEL_SIGNATURE
                if not sig_ok:
                    reason = "シグネチャ不一致（特徴量次元 or バージョン変更）"
                else:
                    # 簡易健全性チェック: 初期推論で NaN が出ないか
                    test_obs = np.zeros(self.MODEL_SIGNATURE["obs_dim"], dtype=np.float32)
                    a, lp, v = self.agent.policy(test_obs)
                    if not (np.isfinite(lp) and np.isfinite(v)):
                        reason = "推論が不正（NaN/Inf 検出）"
                if reason:
                    # 再初期化
                    self.agent = PPOLite(obs_dim=self.MODEL_SIGNATURE["obs_dim"])
                    meta["_message"] = f"既存モデルを無効化: {reason} → 新規作成"
                else:
                    meta["_message"] = "既存モデルを読み込みました"
                    return meta
            except Exception as e:
                self.agent = PPOLite(obs_dim=self.MODEL_SIGNATURE["obs_dim"])
                meta["_message"] = f"既存モデルの読込に失敗: {e} → 新規作成"
                return meta
        else:
            self.agent = PPOLite(obs_dim=self.MODEL_SIGNATURE["obs_dim"])
            meta["_message"] = "既存モデルなし → 新規作成"
        return meta


# ========================= 使い方（参考） ========================= #
if __name__ == "__main__":
    # 下記はデモ用。実運用はユーザの別プログラムから TradingSimulation() を生成し、
    # add(ts, price, volume, force_close) を 1 行ずつ呼び出してください。
    # ここではダミーデータで動作確認のみ行います。
    rng = np.random.default_rng(0)
    base = 3000.0
    volcum = 0.0

    sim = TradingSimulation()
    for t in range(600):  # 10 分相当
        ts = float(t)
        price = base + rng.normal(0, 2)
        dvol = max(0.0, rng.normal(500, 150))
        volcum += dvol
        force_close = (t == 599)
        sim.add(ts, price, volcum, force_close)
    df = sim.finalize()
    print(df.head())
    print("総収益:", df["Profit"].sum())
