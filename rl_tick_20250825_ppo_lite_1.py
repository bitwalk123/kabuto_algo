import os
import math
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ==========================================================
# PPO-lite Actor-Critic for 1銘柄デイトレ tick シミュレーション
# ----------------------------------------------------------
# ・約1秒間隔のティック [ts, price, volume(累積)] を逐次 add() に流し込む
# ・ポジションは同時に1つ（ナンピン/分割なし）
# ・呼び値=1円、売買単位=100株、信用（売り建て可）
# ・新規売り/返済売りは「現在価格 - 1」、新規買い/返済買いは「現在価格 + 1」で約定
# ・報酬: 確定益(全額) + 含み益の一部（重み alpha）
# ・特徴量: Price, MA(10), Volatility(10), RSI(14), log1p(ΔVolume)
# ・PPO(超軽量): 直近のロールアウトを小バッチで数エポック学習
# ・モデル永続化: models/policy.pth（存在すれば読み込み、ダメなら作り直し）
# ・外部の「別プログラム」から finalize() で結果DFを取得&リセット
# ==========================================================


# -----------------------------
# Utilities
# -----------------------------

def _safe_std(x: List[float]) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _compute_rsi_from_deltas(deltas: List[float]) -> float:
    # 教示の単純移動平均版に合わせる（rolling mean）。ストリームでは直近14本を使用
    if len(deltas) < 14:
        return 50.0  # ウォームアップ中の中立値
    gains = [max(d, 0.0) for d in deltas[-14:]]
    losses = [-min(d, 0.0) for d in deltas[-14:]]
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


# -----------------------------
# Actor-Critic Model
# -----------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, act_dim: int = 3):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


# -----------------------------
# PPO-lite core
# -----------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE λ
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 5e-4
    train_iters: int = 4  # 1ロールアウトあたりの学習反復
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_len: int = 512  # ロールアウト蓄積長


class PPOLite:
    def __init__(self, obs_dim: int, act_dim: int, device: Optional[str] = None, model_path: str = "models/policy.pth"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = ActorCritic(obs_dim, hidden=128, act_dim=act_dim).to(self.device)
        self.pi_opt = optim.Adam(self.model.pi.parameters(), lr=PPOConfig.pi_lr)
        self.vf_opt = optim.Adam(self.model.v.parameters(), lr=PPOConfig.vf_lr)

        self.reset_rollout()

        # モデルの読み込み
        loaded = False
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])  # type: ignore
                if "pi_opt" in ckpt and "vf_opt" in ckpt:
                    self.pi_opt.load_state_dict(ckpt["pi_opt"])  # type: ignore
                    self.vf_opt.load_state_dict(ckpt["vf_opt"])  # type: ignore
                loaded = True
                print("既存モデルを読み込みました:", self.model_path)
            except Exception as e:
                print("既存モデルの読み込みに失敗しました。新規に作成します。理由:", str(e))
        if not loaded:
            print("新規モデルを作成しました。")
            self.save()

    def reset_rollout(self):
        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.val_buf: List[float] = []
        self.done_buf: List[float] = []

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "pi_opt": self.pi_opt.state_dict(),
            "vf_opt": self.vf_opt.state_dict(),
        }, self.model_path)
        # print("モデルを保存しました:", self.model_path)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(value.item())

    def store(self, obs, act, logp, rew, val, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def maybe_update(self, force: bool = False):
        if len(self.obs_buf) >= PPOConfig.rollout_len or (force and len(self.obs_buf) > 0):
            self.update()
            self.reset_rollout()
            self.save()

    def update(self):
        # GAE-Lambda advantage
        vals = np.array(self.val_buf + [0.0], dtype=np.float32)
        rews = np.array(self.rew_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + PPOConfig.gamma * vals[t + 1] * nonterminal - vals[t]
            lastgaelam = delta + PPOConfig.gamma * PPOConfig.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals[:-1]

        # to tensors
        obs = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.array(self.act_buf), dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        # normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs.size(0)
        idx = np.arange(n)

        for _ in range(PPOConfig.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, PPOConfig.minibatch_size):
                mb = idx[start:start + PPOConfig.minibatch_size]
                mb_obs = obs[mb]
                mb_act = act[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                logits, v = self.model(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                ratio = (new_logp - mb_old_logp).exp()

                # policy loss
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - PPOConfig.clip_ratio, 1.0 + PPOConfig.clip_ratio) * mb_adv
                pi_loss = -torch.min(unclipped, clipped).mean()
                entropy = dist.entropy().mean()

                # value loss
                v_loss = nn.functional.mse_loss(v.squeeze(-1), mb_ret)

                loss = pi_loss + PPOConfig.value_coef * v_loss - PPOConfig.entropy_coef * entropy

                self.pi_opt.zero_grad()
                self.vf_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), PPOConfig.max_grad_norm)
                self.pi_opt.step()
                self.vf_opt.step()


# -----------------------------
# Trading Simulation
# -----------------------------

class TradingSimulation:
    """
    別プログラムから add(ts, price, volume, force_close=False) を逐次呼び、
    アクション文字列を返す。finalize() で結果DataFrameを返し、内部ログをリセット。

    アクション空間（ポリシーの出力カテゴリ）
      0: HOLD (ホールド)
      1: BUY  → 未建なら新規買い、売り建なら返済買い
      2: SELL → 未建なら新規売り、買い建なら返済売り
    同方向への追撃指示はルール上不可のため HOLD に丸める。
    """

    def __init__(self,
                 model_path: str = "models/ppo_7011_20250825.pth",
                 alpha_unrealized: float = 0.1,  # 含み益重み
                 reward_scale: float = 1.0,  # 報酬スケール（100株分は自動で掛かる）
                 seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 特徴量用バッファ
        self.prices: deque = deque(maxlen=64)  # 余裕を持って確保
        self.deltas: deque = deque(maxlen=64)
        self.prev_volume: Optional[float] = None

        # 売買ルール関連
        self.tick_size = 1.0
        self.lot = 100  # 株
        self.position: str = "FLAT"  # "LONG" / "SHORT" / "FLAT"
        self.entry_price: Optional[float] = None

        # ログ
        self.rows: List[dict] = []

        # PPO-lite
        self.obs_dim = 5  # [norm_price, ma10_z, vol10, rsi, log1p(deltaVol)]
        self.act_dim = 3
        self.agent = PPOLite(self.obs_dim, self.act_dim, model_path=model_path)

        # 報酬パラメータ
        self.alpha_unreal = alpha_unrealized
        self.rscale = reward_scale

        # 前ステップの含み損益（差分報酬用）
        self.prev_unrealized_for_diff: float = 0.0

    # -------------------------
    # Feature engineering per tick
    # -------------------------
    def _features(self, price: float, volume: float) -> np.ndarray:
        self.prices.append(price)
        if len(self.prices) >= 2:
            self.deltas.append(self.prices[-1] - self.prices[-2])
        # MA(10) & Volatility(10)
        ma10 = float(np.mean(list(self.prices)[-10:])) if len(self.prices) >= 10 else price
        vol10 = _safe_std(list(self.prices)[-10:]) if len(self.prices) >= 10 else 0.0
        # z-score of price vs MA/Volatility（分母0は回避）
        if vol10 <= 1e-6:
            ma10_z = 0.0
        else:
            ma10_z = (price - ma10) / (vol10 + 1e-6)
        # RSI(14)
        rsi = _compute_rsi_from_deltas(list(self.deltas))
        rsi_scaled = (rsi - 50.0) / 50.0  # おおよそ [-1, +1]
        # ΔVolume → log1p
        if self.prev_volume is None:
            delta_vol = 0.0
        else:
            delta_vol = max(volume - self.prev_volume, 0.0)
        self.prev_volume = volume
        lv = math.log1p(delta_vol)
        # 価格の正規化（相対変化）: 直近MAで割る
        norm_price = (price / (ma10 + 1e-6)) - 1.0  # おおよそ数%レンジ

        obs = np.array([norm_price, ma10_z, vol10, rsi_scaled, lv], dtype=np.float32)
        return obs

    # -------------------------
    # PnL helpers
    # -------------------------
    def _exec_new_buy(self, price: float) -> float:
        return price + self.tick_size

    def _exec_new_sell(self, price: float) -> float:
        return price - self.tick_size

    def _exec_close_buy(self, price: float) -> float:
        return price + self.tick_size

    def _exec_close_sell(self, price: float) -> float:
        return price - self.tick_size

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == "LONG" and self.entry_price is not None:
            return (price - self.entry_price) * self.lot
        if self.position == "SHORT" and self.entry_price is not None:
            return (self.entry_price - price) * self.lot
        return 0.0

    # -------------------------
    # Public API
    # -------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        1ティック分のデータを受け取り、アクション文字列を返す。
        アクション文字列は "ホールド" / "新規買い" / "返済買い" / "新規売り" / "返済売り"
        のいずれか。
        """
        obs = self._features(price, volume)
        act_idx, logp, val = self.agent.act(obs)

        action_str = "ホールド"
        realized_profit = 0.0

        # マッピング & ルール整合
        if act_idx == 1:  # BUY系
            if self.position == "FLAT":
                # 新規買い
                px = self._exec_new_buy(price)
                self.position = "LONG"
                self.entry_price = px
                action_str = "新規買い"
            elif self.position == "SHORT":
                # 返済買い（ショート解消）
                px = self._exec_close_buy(price)
                realized_profit = (self.entry_price - px) * self.lot  # entry(売) - close(買)
                self.position = "FLAT"
                self.entry_price = None
                action_str = "返済買い"
            else:
                action_str = "ホールド"  # LONG中にBUY指示→無効
        elif act_idx == 2:  # SELL系
            if self.position == "FLAT":
                # 新規売り
                px = self._exec_new_sell(price)
                self.position = "SHORT"
                self.entry_price = px
                action_str = "新規売り"
            elif self.position == "LONG":
                # 返済売り（ロング解消）
                px = self._exec_close_sell(price)
                realized_profit = (px - self.entry_price) * self.lot  # close(売) - entry(買)
                self.position = "FLAT"
                self.entry_price = None
                action_str = "返済売り"
            else:
                action_str = "ホールド"  # SHORT中にSELL指示→無効

        # 強制返済
        if force_close and self.position != "FLAT":
            if self.position == "LONG":
                px = self._exec_close_sell(price)
                realized_profit += (px - self.entry_price) * self.lot
                action_str = "返済売り(強制)"
            elif self.position == "SHORT":
                px = self._exec_close_buy(price)
                realized_profit += (self.entry_price - px) * self.lot
                action_str = "返済買い(強制)"
            self.position = "FLAT"
            self.entry_price = None

        # 報酬（含み益差分 + 確定益）
        unreal = self._unrealized_pnl(price)
        unreal_diff = unreal - self.prev_unrealized_for_diff
        reward = self.alpha_unreal * unreal_diff + realized_profit
        self.prev_unrealized_for_diff = unreal if self.position != "FLAT" else 0.0

        # ロールアウトへ格納
        done_flag = 1.0 if force_close else 0.0
        self.agent.store(obs, act_idx, logp, reward * self.rscale, val, done_flag)
        self.agent.maybe_update(force=False)

        # ログ行
        self.rows.append({
            "Time": ts,
            "Price": price,
            "Action": action_str,
            "Profit": realized_profit,
        })

        return action_str

    def finalize(self) -> pd.DataFrame:
        """結果DataFrameを返して内部状態をリセット。学習を1回まとめて実行。"""
        # 残バッファで最終学習
        self.agent.maybe_update(force=True)

        df = pd.DataFrame(self.rows, columns=["Time", "Price", "Action", "Profit"])

        # エピソード終了ごとに内部のログだけリセット（モデルは維持）
        self.rows.clear()
        self.prices.clear()
        self.deltas.clear()
        self.prev_volume = None
        self.position = "FLAT"
        self.entry_price = None
        self.prev_unrealized_for_diff = 0.0

        return df


# -----------------------------
# 簡易スモークテスト
# -----------------------------
if __name__ == "__main__":
    # ダミーデータで軽く動作確認
    sim = TradingSimulation()
    ts0 = 1_755_000_000
    price = 5000.0
    vol = 1_000_000.0
    for i in range(1200):  # 約20分
        ts = ts0 + i
        price += np.random.randn() * 3.0
        vol += max(0.0, np.random.exponential(2000))
        force = (i == 1199)
        sim.add(ts, float(price), float(vol), force_close=force)
    dfres = sim.finalize()
    print(dfres.head())
    print("総収益:", dfres["Profit"].sum())
