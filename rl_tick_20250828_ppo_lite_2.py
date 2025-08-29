"""
PPO-lite 強化学習ティック売買シミュレーション（単一銘柄・1秒バー想定）
Python 3.13.7 / gymnasium==1.2.0 / numpy==2.2.6 / pandas==2.3.2 / torch==2.8.0

インターフェイス：
  - TradingSimulation.add(ts: float, price: float, volume: float, force_close: bool=False) -> str
      生のティックを1件ずつ追加し、当該時点の売買アクションを返します。
      返り値（アクション文字列・日本語）:
        0: "ホールド"
        1: "買い"
        2: "売り"
        3: "返済"
      ※ 内部では整数アクション {0,1,2,3} を使用しますが、外部返却は日本語文字列です。
      ※ 取引ルールにより不可能なアクションは自動で "ホールド" に置換します。
  - TradingSimulation.finalize() -> pandas.DataFrame
      実行結果（Time, Price, Action, Profit）を返し、内部の記録はリセットされます。

売買ルール（要点）
  - 売買単位 = 100 株、呼び値=1円、スリッページは呼び値1倍。
  - 同時に保持できる建玉は 1 のみ（ロングまたはショート）。ナンピンなし。
  - 約定価格：
      新規買い       : entry = price + 1
      新規売り(空売り): entry = price - 1
      返済（ロング解消）: exit  = price - 1
      返済（ショート解消）: exit  = price + 1
      いずれも金額は株数100を掛けて損益へ反映。
  - 取引手数料は考慮しない。
  - 強制返済フラグ(force_close=True)受領時、建玉があれば即時返済。

特徴量（add 内で自動計算／n=60）
  - 差分出来高 Δvolume = max(volume - last_volume, 0)
  - log1p(Δvolume)
  - 移動平均 MA(60)
  - ボラティリティ std(60)
  - RSI(60)
  - 価格正規化: price_z = (price - running_mean) / sqrt(running_var + eps)
  - 欠損やウォームアップ不足時は学習せずホールド。

報酬設計（PPO-lite）
  - 返済時の確定益（円）をそのまま報酬Rに加算。
  - 含み益の微小な shaping を各ティックで付与（scale_small * unrealized_pnl）。
    scale_small の既定は 0.001（1000円含み益で +1 相当）。

モデル永続化
  - models/policy.pth, models/optimizer.pth
  - 既存モデルがあれば読み込み、簡易検証（ダミー入力での forward）に失敗すれば再生成し上書き。

学習タイミング
  - 1エポック（別プログラムがファイル全件を流し終え finalize() を呼ぶ）ごとに、
    そのエポックの全トランジションで1〜K回の PPO 更新を実施（Kは config.update_epochs）。

注意
  - 本コードは「PPO-lite」の最小実装です。安定化や高速化の余地は多々あります。
"""
from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass, asdict
from typing import Deque, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------- ユーティリティ -------------------------

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# 数値安定用 EPS
EPS = 1e-8


# ------------------------- PPO-lite モデル -------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 2048
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    scale_unrealized: float = 1e-3  # 含み益の shaping 係数


# ------------------------- 特徴量バッファ（オンライン計算） -------------------------

class FeatureState:
    def __init__(self, n: int = 60):
        self.n = n
        self.prices: Deque[float] = deque(maxlen=n)
        self.price_diffs: Deque[float] = deque(maxlen=n)
        self.volumes: Deque[float] = deque(maxlen=2)  # 直近とその前（Δ算出用）
        # 逐次正規化用（価格）
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # 二乗和（Welford）

    def update_price_stats(self, price: float):
        self.count += 1
        delta = price - self.mean
        self.mean += delta / self.count
        delta2 = price - self.mean
        self.m2 += delta * delta2

    def running_std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def compute(self, price: float, volume: float) -> Optional[np.ndarray]:
        """十分なウォームアップが無い間は None を返す。"""
        # Δvolume とその圧縮
        delta_vol = 0.0
        if len(self.volumes) > 0:
            delta_vol = max(volume - self.volumes[-1], 0.0)
        self.volumes.append(volume)
        log1p_dv = math.log1p(delta_vol)

        # 価格系列更新
        if len(self.prices) > 0:
            self.price_diffs.append(price - self.prices[-1])
        self.prices.append(price)
        self.update_price_stats(price)

        # MA, Volatility, RSI（window = n）
        if len(self.prices) < self.n:
            return None  # ウォームアップ不十分

        arr = np.fromiter(self.prices, dtype=np.float64)
        ma = float(np.mean(arr))
        vol = float(np.std(arr, ddof=0))

        # RSI: 直近 n-1 個の price_diffs から算出
        diffs = np.fromiter(self.price_diffs, dtype=np.float64)
        gains = np.clip(diffs, 0, None)
        losses = -np.clip(diffs, None, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # 価格をランニング標準化
        std = self.running_std()
        price_z = (price - self.mean) / (std + 1e-6)  # 標準化

        arr_feature = np.array(
            [price, price_z, log1p_dv, ma, vol, rsi, ],
            dtype=np.float32
        )
        return arr_feature


# ------------------------- 体験バッファ -------------------------

class Trajectory:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.act: List[int] = []
        self.logp: List[float] = []
        self.rew: List[float] = []
        self.val: List[float] = []
        self.done: List[bool] = []

    def clear(self):
        self.__init__()


# ------------------------- 売買シミュレータ -------------------------

class TradingSimulation:
    def __init__(
            self,
            n_feature_window: int = 60,
            unit: int = 100,  # 売買単位 = 100 株
            tick: int = 1,  # 呼び値=1円
            model_dir: str = "models",
            results_dir: str = "results",
            ppo: PPOConfig = PPOConfig(),
            seed: int = 42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        _safe_mkdir(model_dir)
        _safe_mkdir(results_dir)

        self.unit = unit
        self.tick = tick  # 呼び値=1円
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.config = ppo

        # 特徴量状態
        self.fe = FeatureState(n=n_feature_window)

        # 取引状態
        self.position: Optional[str] = None  # "long" | "short" | None
        self.entry_price: float = 0.0  # 約定価格（1株あたり）

        # 記録用 DF
        self.rows: List[Tuple[float, float, str, float]] = []  # Time, Price, Action(str), Profit

        # PPO モデル
        self.obs_dim = 6 + 4  # 6 features + 4 mask bits (行動可否マスク)
        self.act_dim = 4  # 0:Hold, 1:Buy, 2:Sell, 3:Close
        self.device = torch.device("cpu")
        self.net = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.config.lr)

        # 体験
        self.traj = Trajectory()

        # モデルのロード or 新規作成
        self._load_or_init_model()

        # 最後の時刻（Δ計算で使う場合がある）
        self.last_ts: Optional[float] = None

    # ------------------ モデルロード/検証 ------------------
    def _policy_path(self):
        model_name = "policy_7011_20250828.pth"
        return os.path.join(self.model_dir, model_name)

    def _optim_path(self):
        optim_name = "optimizer_7011_20250828.pth"
        return os.path.join(self.model_dir, optim_name)

    def _load_or_init_model(self):
        policy_path = self._policy_path()
        optim_path = self._optim_path()
        created_new = False

        if os.path.exists(policy_path):
            try:
                self.net.load_state_dict(torch.load(policy_path, map_location=self.device))
                if os.path.exists(optim_path):
                    self.optim.load_state_dict(torch.load(optim_path, map_location=self.device))
                # 簡易検証
                self.net.eval()
                dummy = torch.zeros(1, self.obs_dim)
                with torch.no_grad():
                    _ = self.net(dummy)
                print("既存モデルを読み込みました。")
                self.net.train()
                return
            except Exception as e:
                print("既存モデルの読み込みに失敗。新規に作成します:", e)
                created_new = True
        else:
            created_new = True

        if created_new:
            # すでに __init__ で初期化済みの self.net をそのまま使用
            torch.save(self.net.state_dict(), policy_path)
            torch.save(self.optim.state_dict(), optim_path)
            print("有効な既存モデルが無いため、新規モデルを作成しました。")

    def _save_model(self):
        torch.save(self.net.state_dict(), self._policy_path())
        torch.save(self.optim.state_dict(), self._optim_path())
        # print("モデルを保存しました。")

    # ------------------ 売買ロジック ------------------
    def _action_mask(self) -> np.ndarray:
        """行動可否マスク（1:可, 0:不可）。順序は [Hold, Buy, Sell, Close]"""
        can_buy = self.position is None
        can_sell = self.position is None
        can_close = self.position is not None
        mask = np.array([1, 1 if can_buy else 0, 1 if can_sell else 0, 1 if can_close else 0], dtype=np.float32)
        return mask

    def _apply_action(self, action: int, ts: float, price: float) -> Tuple[str, float]:
        """アクション適用し、(action_str, realized_profit) を返す。"""
        realized = 0.0
        action_str = "ホールド"

        if action == 1:  # Buy
            if self.position is None:
                self.position = "long"
                self.entry_price = price + self.tick
                action_str = "買い"
        elif action == 2:  # Sell (short)
            if self.position is None:
                self.position = "short"
                self.entry_price = price - self.tick
                action_str = "売り"
        elif action == 3:  # Close
            if self.position == "long":
                exit_price = price - self.tick
                realized = (exit_price - self.entry_price) * self.unit
                self.position = None
                action_str = "返済"
            elif self.position == "short":
                exit_price = price + self.tick
                realized = (self.entry_price - exit_price) * self.unit
                self.position = None
                action_str = "返済"
        # Hold は何もしない
        return action_str, realized

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == "long":
            return (price - self.entry_price) * self.unit
        elif self.position == "short":
            return (self.entry_price - price) * self.unit
        else:
            return 0.0

    # ------------------ PPO 事後更新 ------------------
    def _ppo_update(self):
        if len(self.traj.obs) == 0:
            return
        cfg = self.config
        obs = torch.tensor(np.array(self.traj.obs), dtype=torch.float32, device=self.device)
        act = torch.tensor(self.traj.act, dtype=torch.long, device=self.device)
        old_logp = torch.tensor(self.traj.logp, dtype=torch.float32, device=self.device)
        rew = torch.tensor(self.traj.rew, dtype=torch.float32, device=self.device)
        val = torch.tensor(self.traj.val, dtype=torch.float32, device=self.device)
        done = torch.tensor(self.traj.done, dtype=torch.float32, device=self.device)

        # GAE-Lambda で advantage & returns
        T = len(rew)
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        lastgaelam = 0.0
        # ブートストラップ値: 最終状態の value を 0 と仮定（エピソード終端）
        for t in reversed(range(T)):
            nextnonterminal = 1.0 - done[t]
            nextvalue = 0.0 if t == T - 1 else val[t + 1]
            delta = rew[t] + cfg.gamma * nextvalue * nextnonterminal - val[t]
            lastgaelam = delta + cfg.gamma * cfg.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 反復更新
        N = len(obs)
        idx = np.arange(N)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, N)
                mb = idx[start:end]
                mb_obs = obs[mb]
                mb_act = act[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_ret = ret[mb]

                logits, value = self.net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_old_logp)

                # clipped surrogate
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_ret - value).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optim.step()

        # 体験をクリア
        self.traj.clear()
        # モデル保存
        self._save_model()

    # ------------------ API: add / finalize ------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """ティックを入力しアクション文字列を返す。"""
        # 特徴量計算
        feats = self.fe.compute(price=float(price), volume=float(volume))

        # 行動マスク
        mask = self._action_mask()

        # ウォームアップ不足 or 強制返済優先
        if force_close and self.position is not None:
            action = 3  # Close
        elif feats is None:
            action = 0  # Hold
        else:
            # 観測ベクトル = [features, mask]
            obs = np.concatenate([feats, mask], axis=0).astype(np.float32)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits, value = self.net(obs_t)
            # マスクを logits に反映（不可アクションは -inf）
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            masked_logits = logits + torch.log(mask_t + EPS)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor([action], device=self.device)).item())
            value_item = float(value.squeeze(0).item())

        # 実行不可アクションの置換
        if action == 1 and mask[1] == 0:
            action = 0
        if action == 2 and mask[2] == 0:
            action = 0
        if action == 3 and mask[3] == 0:
            action = 0

        # アクション適用
        action_str, realized = self._apply_action(action, ts, price)

        # 報酬（確定益 + 含み益 shaping）
        unreal = self._unrealized_pnl(price)
        reward = realized + self.config.scale_unrealized * unreal

        # 経験を貯める（ウォームアップ中は学習しない）
        if feats is not None:
            obs_for_store = np.concatenate([feats, mask], axis=0).astype(np.float32)
            with torch.no_grad():
                logits, value = self.net(torch.tensor(obs_for_store, dtype=torch.float32).unsqueeze(0))
                mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                masked_logits = logits + torch.log(mask_t + EPS)
                dist = torch.distributions.Categorical(logits=masked_logits)
                logp = float(dist.log_prob(torch.tensor([action])).item())
                value_item = float(value.squeeze(0).item())
            self.traj.obs.append(obs_for_store)
            self.traj.act.append(int(action))
            self.traj.logp.append(float(logp))
            self.traj.rew.append(float(reward))
            self.traj.val.append(float(value_item))
            self.traj.done.append(False)  # エピソード終端は finalize() 時に付与

        # 記録
        self.rows.append((float(ts), float(price), action_str, float(realized)))
        self.last_ts = ts

        return action_str

    def finalize(self) -> pd.DataFrame:
        """現在のエピソードを閉じ、結果DFを返して内部状態をリセットする。"""
        # 終端フラグを最後のステップに付与
        if len(self.traj.done) > 0:
            self.traj.done[-1] = True

        # PPO 更新
        self._ppo_update()

        # 結果 DF を作成
        df = pd.DataFrame(self.rows, columns=["Time", "Price", "Action", "Profit"])

        # 内部ログをリセット（取引状態のみ継続: モデルは継続学習する想定）
        self.rows.clear()
        self.traj.clear()
        # 特徴量バッファや position は継続学習のため保持したままにするが、
        # 1エポックごとに完全リセットしたい場合は下記を有効化：
        self.fe = FeatureState(n=self.fe.n)
        self.position = None
        self.entry_price = 0.0

        return df


# -------------- 簡易スモークテスト（直接実行時のみ） --------------
if __name__ == "__main__":
    # 疑似データでワンパス回す（学習確認用）
    sim = TradingSimulation()
    t0 = 1_755_000_000.0
    price = 5000.0
    vol = 1_000_000.0
    rng = np.random.default_rng(0)
    for i in range(400):
        ts = t0 + i
        price += rng.normal(0, 2)
        vol += abs(rng.normal(0, 2000))
        force_close = (i == 399)
        sim.add(ts, price, vol, force_close=force_close)
    out = sim.finalize()
    print(out.tail())
