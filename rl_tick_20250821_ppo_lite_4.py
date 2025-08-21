# -*- coding: utf-8 -*-
"""
PPO-lite 強化学習デイトレ・シミュレータ（単一銘柄／約1秒ティック）

要件対応ポイント:
- インターフェイス: add(ts, price, volume, force_close=False) → 売買アクションを返す
- finalize() で Time/Price/売買アクション/Profit(確定損益) を DataFrame で返す → 返却後に内部結果をリセット
- 1銘柄固定・100株単位・信用売買(現在値で指値約定想定)・ナンピンなし(建玉1つのみ)
- 約定手数料は無視
- 入力は生のティック: ts(float), price(float), volume(float: 累積出来高)
  → add 内で Δvolume= max(0, volume - last_volume) → log1p(Δvolume) で圧縮
  → 価格変化率等も add 内で生成（リターン/モメンタム特徴）
- 学習: PyTorch 2.8 想定の軽量 PPO(Actor-Critic)。モデルは保存/再利用。無効/不一致時は新規作成し上書き。
- 報酬設計:
  * ステップ毎: 含み損益の増分(ΔPnL) を主成分 (100株)
  * 建玉保持ボーナス: 含み益が出ているときに保持すると 0→600秒で最大化、その後は逓減
  * 無効アクション(例: 建玉保有中に新規建て等)は小ペナルティ
  * 返済時: 確定利益を加点。確定利益が +1000円 以上ならボーナス
- 学習更新タイミング: 内部で N ステップごと、ならびに force_close/end で PPO 更新
- 返すアクション(日本語):
  * ホールド
  * 新規買い (Flat→Long)
  * 新規売り (Flat→Short)
  * 返済売り (Long→Flat)
  * 返済買い (Short→Flat)

使い方(質問の別プログラム例に対応):
    sim = TradingSimulation()
    action = sim.add(ts, price, volume, force_close=...)  # 1行ごと
    df_result = sim.finalize()  # 学習1回分の終端で呼び出し

依存:
    pip install torch pandas numpy

注意:
    ・本サンプルは最小構成です。実運用には乱数シード管理、学習停止条件、
      より堅牢なスケーリング/特徴量、異常値処理や板/約定乖離などの現実的制約対応が必要です。
"""
from __future__ import annotations
import math
import os
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================ ユーティリティ: ランニング統計 ============================
class RunningNorm:
    """オンラインで平均/分散を推定して正規化する簡易スケーラ(Welford法)。"""

    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.eps)

    def normalize(self, x: float) -> float:
        if self.n < 10:  # 学習初期はスケールが不安定なので軽く押さえる
            return (x - self.mean) / (self.std + 1.0)
        return (x - self.mean) / (self.std + self.eps)


# ============================ モデル本体 ============================
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(x)
        v = self.value(x).squeeze(-1)
        return logits, v


# ============================ PPO-lite エージェント ============================
@dataclass
class PPOConfig:
    gamma: float = 0.997  # ティック毎の割引(1秒ステップ前提でかなり高め)
    lam: float = 0.95  # GAE lambda
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 1024
    rollout_length: int = 2048  # これだけ溜まったら更新(相場長短に応じて要調整)


class PPOLite:
    def __init__(self, state_dim: int, action_dim: int, device: str = None, config: PPOConfig = PPOConfig()):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = config
        self.net = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        # 直近のロールアウトバッファ
        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf: List[List[float]] = []
        self.act_buf: List[int] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.val_buf: List[float] = []
        self.done_buf: List[bool] = []

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.net.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(x)
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

    def _compute_gae(self, last_val: float = 0.0):
        adv = np.zeros_like(self.rew_buf, dtype=np.float32)
        ret = np.zeros_like(self.rew_buf, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(self.rew_buf))):
            nonterminal = 1.0 - float(self.done_buf[t])
            next_val = last_val if t == len(self.rew_buf) - 1 else self.val_buf[t + 1]
            delta = self.rew_buf[t] + self.cfg.gamma * next_val * nonterminal - self.val_buf[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * gae
            adv[t] = gae
            ret[t] = adv[t] + self.val_buf[t]
        # 標準化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        return adv, ret

    def update(self):
        if len(self.obs_buf) < max(64, self.cfg.minibatch_size):
            return  # データ不足
        adv, ret = self._compute_gae()
        obs = torch.tensor(self.obs_buf, dtype=torch.float32, device=self.device)
        act = torch.tensor(self.act_buf, dtype=torch.long, device=self.device)
        logp_old = torch.tensor(self.logp_buf, dtype=torch.float32, device=self.device)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.tensor(ret, dtype=torch.float32, device=self.device)

        n = obs.size(0)
        idxs = np.arange(n)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idxs[start:start + self.cfg.minibatch_size]
                mb = torch.tensor(mb, dtype=torch.long, device=self.device)
                logits, v = self.net(obs[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((v - ret[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
        self.reset_buffer()

    # --------------- モデル保存/読込 ---------------
    def save(self, path: str, meta: dict):
        payload = {
            "state_dict": self.net.state_dict(),
            "meta": meta,
            "cfg": self.cfg.__dict__,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str, expected_state_dim: int, expected_action_dim: int) -> bool:
        if not os.path.exists(path):
            return False
        try:
            payload = torch.load(path, map_location=self.device)
            meta = payload.get("meta", {})
            cfg = payload.get("cfg", None)
            # 次元不一致などの場合は無効扱い
            if meta.get("state_dim") != expected_state_dim or meta.get("action_dim") != expected_action_dim:
                return False
            self.net.load_state_dict(payload["state_dict"])
            # 学習率等を再設定(保存時と異なる場合も現行 cfg を優先)
            self.opt = optim.Adam(self.net.parameters(), lr=self.cfg.lr)
            return True
        except Exception:
            return False


# ============================ トレード・シミュレーション ============================
class TradingSimulation:
    """
    add(ts, price, volume, force_close=False) を繰り返し呼び出す。
    finalize() で結果 DataFrame を返す(返済後リセット)。
    """
    ACTIONS = ["ホールド", "新規買い", "新規売り", "返済売り", "返済買い"]
    ACT_HOLD, ACT_BUY_OPEN, ACT_SELL_OPEN, ACT_CLOSE_LONG, ACT_CLOSE_SHORT = range(5)

    def __init__(self,
                 # model_path: str = "models/ppo_lite/single_symbol.pt",
                 model_path: str = "models/ppo_7011_20250821.pt",
                 shares_per_trade: int = 100,
                 target_bonus_threshold: float = 1000.0,  # 円
                 hold_peak_sec: int = 600,
                 rollout_length: int = 2048,
                 seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_path = model_path
        self.shares = shares_per_trade
        self.target_bonus = target_bonus_threshold
        self.hold_peak_sec = hold_peak_sec

        # 特徴量: [price_norm, return_1s, vol_log1p_norm, pos_flag_long, pos_flag_short, time_in_pos_norm]
        self.state_dim = 6
        self.action_dim = 5

        cfg = PPOConfig(rollout_length=rollout_length)
        self.agent = PPOLite(self.state_dim, self.action_dim, config=cfg)

        # モデルのロード/新規作成の判定
        loaded = self.agent.load(self.model_path, self.state_dim, self.action_dim)
        if loaded:
            print("[model] 既存モデルを読み込みました → 使用")
        else:
            print("[model] 既存モデルなし/不一致 → 新規作成して保存予定")

        # ランニング正規化器
        self.norm_price = RunningNorm()
        self.norm_ret = RunningNorm()
        self.norm_vol = RunningNorm()

        # 取引状態
        self.reset_episode_state()

        # 出力結果用バッファ
        self.result_rows: List[dict] = []

    # ---------------- 内部状態リセット ----------------
    def reset_episode_state(self):
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None
        self.position: int = 0  # 1: long, -1: short, 0: flat
        self.entry_price: float = 0.0
        self.time_in_pos: int = 0  # 秒
        self.unrealized_pnl: float = 0.0
        self.prev_unrealized_pnl: float = 0.0
        self.last_obs: Optional[np.ndarray] = None
        self.step_counter: int = 0

    # ---------------- 特徴量作成 ----------------
    def _make_features(self, price: float, volume: float) -> np.ndarray:
        # 価格
        self.norm_price.update(price)
        price_norm = self.norm_price.normalize(price)

        # 1秒リターン(初回は0)
        if self.prev_price is None:
            ret = 0.0
        else:
            ret = (price / max(1e-6, self.prev_price)) - 1.0
        self.norm_ret.update(ret)
        ret_norm = self.norm_ret.normalize(ret)

        # Δ出来高 → log1p で圧縮
        if self.prev_volume is None:
            dvol = 0.0
        else:
            dvol = max(0.0, volume - self.prev_volume)
        vol_feat = math.log1p(dvol)
        self.norm_vol.update(vol_feat)
        vol_norm = self.norm_vol.normalize(vol_feat)

        # ポジションフラグ
        pos_long = 1.0 if self.position == 1 else 0.0
        pos_short = 1.0 if self.position == -1 else 0.0
        tnorm = min(self.time_in_pos / float(self.hold_peak_sec), 1.0)

        obs = np.array([price_norm, ret_norm, vol_norm, pos_long, pos_short, tnorm], dtype=np.float32)
        return obs

    # ---------------- 報酬設計 ----------------
    def _step_reward(self, price: float, action: int, realized_on_close: float, done: bool) -> float:
        # 含み損益の変化分(ΔPnL)をベース: 100株固定
        self.unrealized_pnl = (price - self.entry_price) * self.position * self.shares if self.position != 0 else 0.0
        delta_unreal = self.unrealized_pnl - self.prev_unrealized_pnl
        self.prev_unrealized_pnl = self.unrealized_pnl

        r = 0.001 * (delta_unreal)  # スケール調整(相場や価格帯に応じて要調整)

        # 保持ボーナス: 含み益がプラスの時のみ、0→peakで増加、以降は逓減
        if self.position != 0 and self.unrealized_pnl > 0:
            x = min(self.time_in_pos, self.hold_peak_sec)
            hold_bonus = 0.0005 * (x / self.hold_peak_sec) * (self.unrealized_pnl / 1000.0)
            r += hold_bonus
            if self.time_in_pos > self.hold_peak_sec:
                # 長すぎる保持は小さく減点(逓減)
                over = self.time_in_pos - self.hold_peak_sec
                r -= 0.0002 * (over / self.hold_peak_sec)

        # 無効アクションの軽いペナルティ
        invalid = False
        if self.position == 0 and action in (self.ACT_CLOSE_LONG, self.ACT_CLOSE_SHORT):
            invalid = True
        if self.position == 1 and action == self.ACT_BUY_OPEN:
            invalid = True
        if self.position == -1 and action == self.ACT_SELL_OPEN:
            invalid = True
        if invalid:
            r -= 0.001

        # 返済時の確定損益を加点
        if realized_on_close != 0.0:
            r += 0.002 * realized_on_close
            if realized_on_close >= self.target_bonus:
                r += 1.0  # 目標達成ボーナス

        # エピソード終端で強制返済された場合は、終端でやや強い信号
        if done and self.position == 0:
            r += 0.0  # 必要なら微調整
        return float(r)

    # ---------------- アクション適用 ----------------
    def _apply_action(self, price: float, action: int) -> Tuple[str, float]:
        action_str = "ホールド"
        realized = 0.0

        if action == self.ACT_BUY_OPEN:
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.time_in_pos = 0
                self.prev_unrealized_pnl = 0.0
                action_str = "新規買い"
            else:
                action_str = "ホールド"  # 無効 → 何もしない

        elif action == self.ACT_SELL_OPEN:
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.time_in_pos = 0
                self.prev_unrealized_pnl = 0.0
                action_str = "新規売り"
            else:
                action_str = "ホールド"

        elif action == self.ACT_CLOSE_LONG:
            if self.position == 1:
                realized = (price - self.entry_price) * self.shares
                self.position = 0
                self.entry_price = 0.0
                self.time_in_pos = 0
                self.unrealized_pnl = 0.0
                self.prev_unrealized_pnl = 0.0
                action_str = "返済売り"
            else:
                action_str = "ホールド"

        elif action == self.ACT_CLOSE_SHORT:
            if self.position == -1:
                realized = (self.entry_price - price) * self.shares
                self.position = 0
                self.entry_price = 0.0
                self.time_in_pos = 0
                self.unrealized_pnl = 0.0
                self.prev_unrealized_pnl = 0.0
                action_str = "返済買い"
            else:
                action_str = "ホールド"

        else:
            action_str = "ホールド"

        return action_str, realized

    # ---------------- 公開 API: add ----------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        # 観測生成
        obs = self._make_features(price, volume)

        # ポリシーから行動サンプル
        act, logp, val = self.agent.act(obs)

        # 建玉状況に応じて CLOSE アクションのマスク代替(無効時の確率を自然に抑える)
        # ここでは無効でも実行はせず報酬側で軽く減点する設計

        # 強制返済フラグが True なら、建玉があれば返済を最優先
        if force_close and self.position != 0:
            act = self.ACT_CLOSE_LONG if self.position == 1 else self.ACT_CLOSE_SHORT

        # 行動を適用
        action_str, realized = self._apply_action(price, act)

        # 報酬計算
        done = bool(force_close)
        reward = self._step_reward(price, act, realized_on_close=realized, done=done)

        # 次状態(学習用)
        next_obs = self._make_features(price, volume)

        # バッファへ格納
        self.agent.store(obs.tolist(), act, logp, reward, val, done)

        # 出力行を追加
        self.result_rows.append({
            "Time": ts,
            "Price": price,
            "売買アクション": action_str,
            "Profit": realized  # 返済時のみ±値、それ以外は0
        })

        # 経過時間更新
        if self.position != 0:
            self.time_in_pos += 1  # 1秒ステップ仮定

        # 前回値更新
        self.prev_price = price
        self.prev_volume = volume
        self.step_counter += 1

        # ロールアウト長に達したら更新
        if len(self.agent.obs_buf) >= self.agent.cfg.rollout_length or done:
            try:
                self.agent.update()
            except Exception as e:
                print("[warn] PPO更新に失敗:", repr(e))

        return action_str

    # ---------------- 公開 API: finalize ----------------
    def finalize(self) -> pd.DataFrame:
        # 念のため建玉が残っていればクローズ(理論上は force_close で処理済み)
        if self.position != 0 and self.prev_price is not None:
            act = self.ACT_CLOSE_LONG if self.position == 1 else self.ACT_CLOSE_SHORT
            action_str, realized = self._apply_action(self.prev_price, act)
            reward = self._step_reward(self.prev_price, act, realized_on_close=realized, done=True)
            self.agent.store((self.last_obs if self.last_obs is not None else [0] * self.state_dim),
                             act, 0.0, reward, 0.0, True)
            self.result_rows.append({
                "Time": (self.result_rows[-1]["Time"] if self.result_rows else 0.0),
                "Price": self.prev_price,
                "売買アクション": action_str,
                "Profit": realized
            })
            try:
                self.agent.update()
            except Exception as e:
                print("[warn] PPO更新に失敗:", repr(e))

        # モデル保存
        meta = {"state_dim": self.state_dim, "action_dim": self.action_dim, "ts": time.time()}
        try:
            self.agent.save(self.model_path, meta)
            # print("[model] モデルを保存しました:", self.model_path)
        except Exception as e:
            print("[warn] モデル保存に失敗:", repr(e))

        # 結果を DataFrame 化
        df = pd.DataFrame(self.result_rows, columns=["Time", "Price", "売買アクション", "Profit"])

        # 次エポックに向けて内部状態をリセット(モデルは保持)
        self.reset_episode_state()
        self.result_rows = []
        # 学習バッファは流動的に保持(エポック跨ぎで継続学習を許容)。

        return df


# --------------- スモークテスト ---------------
if __name__ == "__main__":
    # ダミーデータで軽く動作テスト(本番は質問者の別プログラムを使用)
    sim = TradingSimulation()
    t = 0
    price = 3000.0
    volume_cum = 0.0
    for i in range(3000):  # 約50分想定
        t += 1
        # ランダムウォーク
        price *= (1.0 + np.random.randn() * 0.0005)
        volume_cum += max(0.0, np.random.poisson(50))
        force = (i == 2999)
        _ = sim.add(float(t), float(price), float(volume_cum), force_close=force)
    df_res = sim.finalize()
    print(df_res.head())
    print("総収益:", df_res["Profit"].sum())
