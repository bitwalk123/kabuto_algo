# trading_simulation_rl.py
# -*- coding: utf-8 -*-
"""
デイトレード向け 強化学習（PPO-lite, Actor-Critic）サンプル
- 1秒ティック [ts(float), price(float), volume(float)] を逐次 add() に渡す
- 信用売買: 返済売りは price-1, 返済買いは price+1 で約定（呼び値=1円）
- 売買単位=100株, ポジションは同時に1単位のみ（ナンピンしない）
- 特徴量は add() 内で生成（差分出来高→log1p圧縮、価格系：MA, Volatility, RSI, など）
- 報酬は含み益に応じた非線形倍率＋わずかなボーナス/ペナルティ
- 学習モデルは models/ppo_lite.pth に保存・再利用（ロード→検証→失敗時は再作成）
- finalize() で結果DataFrameを返し、内部ログのみリセット（モデルは継続学習）

依存: Python 3.10+, PyTorch 2.8+, numpy, pandas, gymnasium 1.2
"""

from __future__ import annotations
import os
import math
import json
import time
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

MODEL_DIR = "models"
MODEL_NAME = "ppo_7011_20250824.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
RESULTS_DIR = "results"


# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_rsi(prices: Deque[float], window: int = 14) -> float:
    if len(prices) < window + 1:
        return np.nan
    # last `window+1` samples
    arr = np.array(list(prices)[-window - 1:])
    delta = np.diff(arr)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


@dataclass
class TradeState:
    position: int = 0  # -1 (short), 0 (flat), +1 (long)
    entry_price: float = 0.0  # 建玉の始値（現値で建てる）
    unit: int = 100  # 株数

    def unrealized_pnl(self, price: float) -> float:
        # 含み損益（円）。
        if self.position == 0:
            return 0.0
        if self.position > 0:  # long
            return (price - self.entry_price) * self.unit
        else:  # short
            return (self.entry_price - price) * self.unit

    def close(self, price: float, tick: float = 1.0) -> float:
        # 返済時の約定価格ルールに基づいて実現損益を返す
        if self.position == 0:
            return 0.0
        if self.position > 0:
            # 返済売り: price - tick
            exit_price = price - tick
            pnl = (exit_price - self.entry_price) * self.unit
        else:
            # 返済買い: price + tick
            exit_price = price + tick
            pnl = (self.entry_price - exit_price) * self.unit
        # ポジション解消
        self.position = 0
        self.entry_price = 0.0
        return float(pnl)

    def open_long(self, price: float):
        self.position = 1
        self.entry_price = price  # 新規は現値

    def open_short(self, price: float):
        self.position = -1
        self.entry_price = price  # 新規は現値


# ------------------------------------------------------------
# PPO-lite モデル
# ------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, num_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, num_actions)  # policy logits
        self.v = nn.Linear(hidden, 1)  # state value
        # orthogonal init (軽量)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.v.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class PPOLite:
    def __init__(
            self,
            obs_dim: int,
            num_actions: int = 3,
            lr: float = 3e-4,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_ratio: float = 0.2,
            train_iters: int = 4,
            batch_size: int = 1024,
            device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(obs_dim, hidden=128, num_actions=num_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.batch_size = batch_size

        # rollout buffer
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.val_buf: List[float] = []
        self.done_buf: List[float] = []

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.net.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(o)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    def store(self, obs, act, logp, rew, val, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def _compute_gae(self, rewards, values, dones, gamma, lam):
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            nextv = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * nextv * nonterminal - values[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values
        # 標準化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self):
        if len(self.obs_buf) == 0:
            return {}
        obs = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(np.array(self.act_buf), dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)
        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv, ret = self._compute_gae(rews, vals, dones, self.gamma, self.lam)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        info = {}
        N = obs.shape[0]
        idx = np.arange(N)
        for it in range(self.train_iters):
            np.random.shuffle(idx)
            for start in range(0, N, self.batch_size):
                batch = idx[start:start + self.batch_size]
                b_obs = obs[batch]
                b_acts = acts[batch]
                b_old_logp = old_logp[batch]
                b_adv = adv[batch]
                b_ret = ret[batch]

                logits, v = self.net(b_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_acts)
                ratio = torch.exp(new_logp - b_old_logp)
                clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                loss_pi = -(torch.min(ratio * b_adv, clip_adv)).mean()
                loss_v = F.mse_loss(v, b_ret)
                ent = dist.entropy().mean()

                loss = loss_pi + 0.5 * loss_v - 0.01 * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

        # クリア
        self.obs_buf.clear();
        self.act_buf.clear();
        self.logp_buf.clear()
        self.rew_buf.clear();
        self.val_buf.clear();
        self.done_buf.clear()
        return info

    # 保存/読み込み
    def save(self, path: str = MODEL_PATH):
        ensure_dir(os.path.dirname(path))
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'meta': {
                'obs_dim': self.net.fc1.in_features,
                'timestamp': time.time(),
            }
        }, path)

    @staticmethod
    def load_or_create(path: str, obs_dim: int, num_actions: int = 3) -> Tuple['PPOLite', str]:
        ensure_dir(os.path.dirname(path))
        status = ""
        try:
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                meta = ckpt.get('meta', {})
                # obs_dim 変化などの簡易検証
                saved_obs_dim = int(meta.get('obs_dim', obs_dim))
                agent = PPOLite(obs_dim=saved_obs_dim, num_actions=num_actions)
                agent.net.load_state_dict(ckpt['model_state_dict'])
                agent.opt.load_state_dict(ckpt['opt_state_dict'])
                if saved_obs_dim != obs_dim:
                    status = f"既存モデルのobs_dim({saved_obs_dim})!=現在({obs_dim}) → 破棄して新規作成"
                    agent = PPOLite(obs_dim=obs_dim, num_actions=num_actions)
                else:
                    status = "既存モデルを読み込みました"
            else:
                status = "既存モデルなし → 新規作成"
                agent = PPOLite(obs_dim=obs_dim, num_actions=num_actions)
        except Exception as e:
            status = f"既存モデルの読み込みに失敗 → 新規作成 ({e})"
            agent = PPOLite(obs_dim=obs_dim, num_actions=num_actions)
        return agent, status


# ------------------------------------------------------------
# メイン: TradingSimulation（ストリーミングRL）
# ------------------------------------------------------------
class TradingSimulation:
    """
    外部の「別プログラム」から逐次 add(ts, price, volume, force_close=False) を呼び出す。
    戻り値はアクション（"HOLD"|"BUY"|"SELL"）。
    finalize() で結果DataFrameを返し、内部のログをリセット。
    モデルは継続学習し、適宜保存する。
    """

    def __init__(self,
                 tick_size: float = 1.0,
                 unit: int = 100,
                 rollout_min: int = 2048,
                 update_every: int = 2048,
                 feature_windows: Tuple[int, int] = (10, 14),  # MA/Vol=10, RSI=14
                 device: Optional[str] = None):
        self.tick = tick_size
        self.state = TradeState(position=0, entry_price=0.0, unit=unit)

        # 特徴量用バッファ
        self.prices: Deque[float] = deque(maxlen=max(feature_windows) + 2)
        self.volumes: Deque[float] = deque(maxlen=3)  # 累積出来高（直近2点で差分）

        # 結果ログ
        self._records: List[Tuple[float, float, str, float]] = []  # (Time, Price, Action, Profit)

        # 観測ベクトル: [price_norm, ret1, zscore10, vol10, rsi14/100, log1p(dv), pos]
        self.obs_dim = 7
        self.num_actions = 3  # 0=HOLD, 1=BUY/CloseShort, 2=SELL/CloseLong

        # エージェント
        self.agent, status = PPOLite.load_or_create(MODEL_PATH, obs_dim=self.obs_dim, num_actions=self.num_actions)
        print(status)
        self.device = self.agent.device

        # ロールアウト管理
        self.update_every = update_every
        self.rollout_min = rollout_min
        self.t_step = 0

    # ------------------- 特徴量生成 -------------------
    def _features(self, price: float, volume: float) -> np.ndarray:
        self.prices.append(price)
        self.volumes.append(volume)

        # 差分出来高
        delta_v = 0.0
        if len(self.volumes) >= 2:
            dv = self.volumes[-1] - self.volumes[-2]
            # 取引所の巻き戻りなどでマイナスになることがあるためクリップ
            delta_v = max(0.0, float(dv))
        log_dv = math.log1p(delta_v)

        # 移動平均/ボラ（window=10）
        ma = np.nan
        vol = np.nan
        if len(self.prices) >= 10:
            arr10 = np.array(list(self.prices)[-10:])
            ma = float(arr10.mean())
            vol = float(arr10.std(ddof=0))
        # RSI(window=14)
        rsi = compute_rsi(self.prices, window=14)

        # 1秒リターン
        ret1 = 0.0
        if len(self.prices) >= 2:
            prev = self.prices[-2]
            if prev > 0:
                ret1 = (price - prev) / prev

        # 標準化・スケーリング
        price_norm = 0.0 if np.isnan(ma) or vol is None or (vol is not None and vol == 0.0) else (price - ma) / (
                vol + 1e-6)
        zscore10 = price_norm
        vol10 = 0.0 if (vol is None or np.isnan(vol)) else vol
        rsi01 = 0.0 if np.isnan(rsi) else (rsi / 100.0)
        pos = float(self.state.position)

        obs = np.array([
            price_norm if not np.isnan(price_norm) else 0.0,
            ret1,
            zscore10 if not np.isnan(zscore10) else 0.0,
            vol10 if not np.isnan(vol10) else 0.0,
            rsi01,
            log_dv,
            pos,
        ], dtype=np.float32)
        return obs

    # ------------------- 報酬関数 -------------------
    def _reward(self, prev_unreal: float, curr_unreal: float) -> float:
        # 基本は含み損益の増分（差分）
        base = curr_unreal - prev_unreal
        # 非線形倍率（含み益が閾値超えで強化）
        multiplier = 1.0
        if curr_unreal >= 1000.0:
            multiplier = 2.0
        elif curr_unreal >= 500.0:
            multiplier = 1.5
        elif curr_unreal >= 200.0:
            multiplier = 1.0
        elif curr_unreal > 0.0:
            multiplier = 1.0  # わずかなボーナスは下で加点
        elif curr_unreal < 0.0:
            multiplier = 1.0  # ペナルティは下で減点

        reward = base * multiplier
        # わずかなボーナス/ペナルティ（スケールを小さく）
        if curr_unreal > 0:
            reward += 0.1
        elif curr_unreal < 0:
            reward -= 0.1

        return float(reward)

    # ------------------- 行動適用 -------------------
    def _apply_action(self, action: int, price: float) -> Tuple[str, float]:
        """ポリシーの 0/1/2 を売買規則にマップして適用。返り値: (アクション名, realized_profit)"""
        realized = 0.0
        label = "HOLD"

        if self.state.position == 0:
            if action == 1:
                self.state.open_long(price)
                label = "BUY"  # 新規買い
            elif action == 2:
                self.state.open_short(price)
                label = "SELL"  # 新規売り（空売り）
        elif self.state.position > 0:  # ロング中
            if action == 2:
                realized = self.state.close(price, tick=self.tick)
                label = "CLOSE_LONG"
        elif self.state.position < 0:  # ショート中
            if action == 1:
                realized = self.state.close(price, tick=self.tick)
                label = "CLOSE_SHORT"
        return label, float(realized)

    # ------------------- 公開API -------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        1件のティックを受け取り、行動を決定して適用。行動ラベルを返す。
        """
        # 観測作成
        obs = self._features(price, volume)
        prev_unreal = self.state.unrealized_pnl(price)

        # 強制クローズ対応（終端）: まず適用
        if force_close and self.state.position != 0:
            realized = self.state.close(price, tick=self.tick)
            self._records.append((ts, price, "FORCE_CLOSE", realized))
            # 終了遷移: done=1 でストア
            self.agent.store(obs, 0, 0.0, realized, prev_unreal, 1.0)
            # バッファが溜まっていれば更新
            if len(self.agent.obs_buf) >= self.rollout_min:
                self.agent.update()
                self.agent.save(MODEL_PATH)
            return "FORCE_CLOSE"

        # 行動サンプル
        act, logp, val = self.agent.act(obs)
        label, realized = self._apply_action(act, price)

        # 報酬: 含み損益の差分＋スケーリング
        curr_unreal = self.state.unrealized_pnl(price)
        reward = self._reward(prev_unreal, curr_unreal)
        done = 0.0

        # 返済が起きた場合は実現損益を報酬に加える（大きめの重み）
        if label in ("CLOSE_LONG", "CLOSE_SHORT"):
            reward += realized

        # バッファへ保存
        self.agent.store(obs, act, logp, reward, val, done)

        # 更新タイミング
        self.t_step += 1
        if len(self.agent.obs_buf) >= self.update_every:
            self.agent.update()
            self.agent.save(MODEL_PATH)

        # ログ
        self._records.append((ts, price, label, realized))
        return label

    def finalize(self) -> pd.DataFrame:
        """結果DataFrameを返し、内部ログをリセット。必要なら学習も実施し保存。"""
        if len(self.agent.obs_buf) >= self.rollout_min:
            self.agent.update()
            self.agent.save(MODEL_PATH)

        df = pd.DataFrame(self._records, columns=["Time", "Price", "Action", "Profit"])
        # 内部ログをリセット
        self._records.clear()
        return df


# ------------------------------------------------------------
# スモークテスト（任意）
# ------------------------------------------------------------
if __name__ == "__main__":
    # 簡易ランダムデータで動作確認（必要に応じてコメントアウト）
    np.random.seed(0)
    sim = TradingSimulation()
    base = 3000.0
    volume = 10000.0
    t0 = 1_755_000_000.0
    for i in range(5000):
        price = base + np.random.randn() * 5.0
        volume += max(0.0, np.random.exponential(scale=50.0))
        force = (i == 4999)
        _ = sim.add(t0 + i, float(price), float(volume), force_close=force)
    df_res = sim.finalize()
    ensure_dir(RESULTS_DIR)
    df_res.to_csv(os.path.join(RESULTS_DIR, "smoke_results.csv"), index=False)
    print(df_res.tail())
    print("total profit:", df_res["Profit"].sum())
