# trading_simulation_ppolite.py
# Python 3.13.7 / numpy 2.2.6 / pandas 2.3.2 / torch 2.8.0
import os
import math
import json
import time
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

# ====== 定数（売買条件）======
LOT_SIZE = 100                # 売買単位（株）
TICK_SIZE = 1.0               # 呼び値（円）
SLIPPAGE = 2.0                # スリッページ幅（円）= 呼び値の2倍
MAX_POSITION = 1              # 同時に1建玉のみ（ロング=+1 / ショート=-1 / ノーポジ=0）

# ====== PPO-lite ハイパー ======
GAMMA = 0.99
LAMBDA = 0.95                 # GAE
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
BATCH_SIZE = 512              # これ以上経験が溜まったら更新
PPO_EPOCHS = 4                # 1バッチあたりの反復回数
MINI_BATCHES = 4              # ミニバッチ分割数
UNREALIZED_SHAPING_BETA = 0.02  # 含み益の形状化比率（調整推奨）

RESULTS_COLUMNS = ["Time", "Price", "Action", "Profit"]  # Profitは確定益（建玉返済時のみ値）

# ====== ユーティリティ ======
def safe_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ====== 価格・出来高の特徴量（60秒窓）======
class FeatureBuilder:
    def __init__(self, window: int = 60):
        self.win = window
        self.prices: Deque[float] = deque(maxlen=window)
        self.prev_volume: Optional[float] = None

    def _rsi(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 50.0
        diffs = np.diff(prices)
        gains = np.clip(diffs, 0, None)
        losses = np.clip(-diffs, 0, None)
        # 単純移動平均（ご指定と同等のロジック）
        if len(diffs) < self.win:
            w = len(diffs)
        else:
            w = self.win - 1
        if w <= 0:
            return 50.0
        avg_gain = gains[-w:].mean() if w > 0 else 0.0
        avg_loss = losses[-w:].mean() if w > 0 else 0.0
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    def update_and_build(self, price: float, volume: float) -> Tuple[np.ndarray, dict]:
        """生Price/Volumeから観測ベクトルとデバッグ用辞書を生成"""
        self.prices.append(price)

        # Δvolume → log1p で圧縮
        if self.prev_volume is None or volume < self.prev_volume:
            delta_vol = 0.0  # 寄り付きやリセット時は0扱い
        else:
            delta_vol = max(volume - self.prev_volume, 0.0)
        self.prev_volume = volume
        log_dv = math.log1p(delta_vol)

        # MA / Volatility / RSI（窓が満たない間はフォールバック）
        if len(self.prices) >= 2:
            ret1 = (self.prices[-1] - self.prices[-2]) / max(self.prices[-2], 1e-6)
        else:
            ret1 = 0.0

        if len(self.prices) >= self.win:
            arr = np.array(self.prices, dtype=float)
            ma = float(arr.mean())
            vol = float(arr.std(ddof=0))
        else:
            arr = np.array(self.prices, dtype=float)
            ma = float(arr.mean())
            vol = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0

        rsi = self._rsi(list(self.prices))
        # 乖離（%）
        ma_dev = (price - ma) / max(ma, 1e-6)

        # スケーリング：rsiは0..1へ、volは価格で正規化
        rsi01 = rsi / 100.0
        vol_norm = vol / max(price, 1e-6)

        feats = np.array([
            ret1,          # 1秒リターン
            ma_dev,        # MA乖離率
            vol_norm,      # ボラの相対値
            rsi01,         # RSI(0..1)
            log_dv,        # log1p(Δvolume)
        ], dtype=np.float32)

        info = dict(MA=ma, Volatility=vol, RSI=rsi, LogDeltaVol=log_dv)
        return feats, info

# ====== PPO-lite ネットワーク ======
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 128
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v  = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z)

# ====== メモリ（オンポリシー）======
@dataclass
class TrajBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    acts: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    rews: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)

    def clear(self):
        self.obs.clear(); self.acts.clear(); self.logps.clear()
        self.rews.clear(); self.dones.clear(); self.vals.clear()

# ====== メイン：TradingSimulation ======
class TradingSimulation:
    """
    add(ts, price, volume, force_close=False) -> action_str
    finalize() -> pd.DataFrame (Time, Price, Action, Profit)
    """
    def __init__(self, model_dir: str = "models", results_dir: str = "results", seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        safe_dir(model_dir); safe_dir(results_dir)
        self.model_dir = model_dir
        self.results_dir = results_dir

        # 状態
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fb = FeatureBuilder(window=60)
        self.results = []  # list of dict rows for DataFrame
        self.position = 0  # -1: short, 0: flat, +1: long
        self.entry_price_exec = None  # 取引の約定価格（スリッページ込み）
        self.buffer = TrajBuffer()
        self.last_value = 0.0  # critic の最後の値（GAE終端計算用）

        # 観測: 価格特徴5 + ポジション埋め込み(3) + 含み益相対 + 時刻推定は省略
        self.obs_dim = 5 + 3 + 1
        self.act_dim = 4  # 0:ホールド,1:買い,2:売り,3:返済

        # モデルのロード or 新規作成
        self.model = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self._load_or_init_model()

    # ---------- モデル保存/読み込み ----------
    def _ckpt_path(self):
        return os.path.join(self.model_dir, "policy.pth")

    def _load_or_init_model(self):
        path = self._ckpt_path()
        meta_ok = False
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                meta = ckpt.get("meta", {})
                obs_dim_ok = (meta.get("obs_dim") == self.obs_dim)
                act_dim_ok = (meta.get("act_dim") == self.act_dim)
                if obs_dim_ok and act_dim_ok:
                    self.model.load_state_dict(ckpt["model"])
                    self.optimizer.load_state_dict(ckpt["opt"])
                    print("既存モデルを使用: models/policy.pth")
                    meta_ok = True
                else:
                    print("既存モデルが無効（入出力次元の不一致）。新規作成して上書きします。")
            except Exception as e:
                print(f"既存モデルが読み込めませんでした（{e}）。新規作成して上書きします。")

        if not meta_ok:
            # すでに self.model は初期化済み。すぐ保存して「有効モデル」を作る
            self._save_model(newly_created=True)

    def _save_model(self, newly_created: bool = False):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "meta": {"obs_dim": self.obs_dim, "act_dim": self.act_dim, "time": time.time()},
        }
        torch.save(ckpt, self._ckpt_path())
        if newly_created:
            print("新規モデルを作成: models/policy.pth")
        else:
            print("モデルを保存しました: models/policy.pth")

    # ---------- 観測の構築 ----------
    def _build_obs(self, price: float, volume: float) -> Tuple[np.ndarray, dict]:
        feats, info = self.fb.update_and_build(price, volume)
        # ポジションのワンホット
        pos_embed = {
            -1: np.array([1,0,0], dtype=np.float32),
             0: np.array([0,1,0], dtype=np.float32),
             1: np.array([0,0,1], dtype=np.float32),
        }[self.position]

        # 含み益（1株あたり）を相対化して1特徴として追加（ノーポジ=0）
        unreal = 0.0
        if self.position != 0 and self.entry_price_exec is not None:
            if self.position > 0:
                # ロングの含み益（まだ未約定なのでスリッページ無しの現在値ベース）
                unreal = (price - self.entry_price_exec)
            else:
                unreal = (self.entry_price_exec - price)
        unreal_rel = unreal / max(price, 1e-6)

        obs = np.concatenate([feats, pos_embed, np.array([unreal_rel], dtype=np.float32)], axis=0)
        return obs, info

    # ---------- 約定ヘルパ ----------
    def _exec_price_for_open(self, side: int, price: float) -> float:
        # side: +1=新規買い, -1=新規売り
        if side > 0:
            return price + SLIPPAGE
        else:
            return price - SLIPPAGE

    def _exec_price_for_close(self, side: int, price: float) -> float:
        # sideは現在の建玉 (+1:ロング, -1:ショート)
        # ロングを返済→成行売り（価格-スリッページ）、ショート返済→成行買い（価格+スリッページ）
        if side > 0:
            return price - SLIPPAGE
        else:
            return price + SLIPPAGE

    # ---------- アクション選択 ----------
    def _select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.model.eval()
        x = to_tensor(obs[None, :], self.device)
        with torch.no_grad():
            logits, val = self.model(x)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()  # ストキャスティックにサンプル（探索を内包）
        logp = dist.log_prob(act)
        return int(act.item()), float(logp.item()), float(val.item())

    # ---------- ステップ実行（報酬計算含む） ----------
    def _step_env(self, ts: float, price: float, action: int, force_close: bool) -> Tuple[float, str, Optional[float], bool]:
        """
        戻り値: (reward, action_str, realized_profit, done)
        realized_profit は建玉返済時のみ実数、それ以外None
        """
        realized = None
        action_str = "ホールド"
        done = False

        # 強制返済が指定されていれば、返済を優先
        if force_close and self.position != 0:
            action = 3  # 返済
            action_str = "返済"

        # 無効アクションはホールド化
        if self.position == 0 and action == 3:
            action = 0  # ノーポジで返済→無効
        if self.position != 0 and action in (1,2):
            action = 0  # 建玉中の新規→禁止（ナンピン不可）

        # 実行
        if action == 1 and self.position == 0:  # 買いで新規ロング
            self.position = +1
            self.entry_price_exec = self._exec_price_for_open(+1, price)
            action_str = "買い"
        elif action == 2 and self.position == 0:  # 売りで新規ショート
            self.position = -1
            self.entry_price_exec = self._exec_price_for_open(-1, price)
            action_str = "売り"
        elif action == 3 and self.position != 0:  # 返済（強制含む）
            exit_exec = self._exec_price_for_close(self.position, price)
            if self.position > 0:
                # ロングの確定益（100株）
                realized = (exit_exec - self.entry_price_exec) * LOT_SIZE
                action_str = "返済売り" if not force_close else "返済売り（強制）"
            else:
                # ショートの確定益（100株）
                realized = (self.entry_price_exec - exit_exec) * LOT_SIZE
                action_str = "返済買い" if not force_close else "返済買い（強制）"
            # ノーポジへ
            self.position = 0
            self.entry_price_exec = None
        else:
            action_str = "ホールド"

        # 報酬：確定益 + β×含み益（形状化）。単位は円ベース→適度にスケール。
        reward = 0.0
        if realized is not None:
            reward += realized

        # 含み益（1株あたり）→100株に合わせつつβ係数で弱める
        if self.position != 0 and self.entry_price_exec is not None:
            if self.position > 0:
                unreal = (price - self.entry_price_exec) * LOT_SIZE
            else:
                unreal = (self.entry_price_exec - price) * LOT_SIZE
            reward += UNREALIZED_SHAPING_BETA * unreal

        # エピソード終端は「強制返済が行われてノーポジになった瞬間」とする
        if force_close and self.position == 0:
            done = True

        return reward, action_str, realized, done

    # ---------- 1ティック処理 ----------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        別プログラムから1秒ごとに呼び出す。
        返り値：'買い' '売り' '返済' 'ホールド' のいずれか（強制返済時の表示は結果DFに残ります）
        """
        # 観測の構築
        obs, _ = self._build_obs(price, volume)

        # アクション選択（PPOポリシー）
        act, logp, val = self._select_action(obs)

        # 環境反映（約定・報酬）
        reward, action_str, realized, done = self._step_env(ts, price, act, force_close)

        # バッファに蓄積
        self.buffer.obs.append(obs)
        self.buffer.acts.append(
            { "ホールド":0, "買い":1, "売り":2, "返済":3 }[action_str if action_str in ("ホールド","買い","売り","返済") else "返済"]
        )
        self.buffer.logps.append(logp)
        self.buffer.rews.append(reward)
        self.buffer.dones.append(done)
        self.buffer.vals.append(val)

        # 結果行を記録（Profitは確定時のみ、それ以外は0）
        self.results.append({
            "Time": float(ts),
            "Price": float(price),
            "Action": action_str if action_str in ("返済買い","返済売り","返済買い（強制）","返済売り（強制）") else action_str,
            "Profit": float(realized) if realized is not None else 0.0
        })

        # 学習トリガ（十分溜まるか、エピソード終端）
        if len(self.buffer.obs) >= BATCH_SIZE or done:
            self._ppo_update(last_done=done)
            self._save_model(newly_created=False)

        return "返済" if action_str.startswith("返済") else action_str

    # ---------- PPO 更新 ----------
    def _ppo_update(self, last_done: bool):
        if not self.buffer.obs:
            return

        # 末端の価値（bootstrap用）
        with torch.no_grad():
            if last_done:
                next_val = 0.0
            else:
                # 現在の最後の観測から近似（最後のobsを再利用）
                x = to_tensor(self.buffer.obs[-1][None, :], self.device)
                _, v = self.model(x)
                next_val = float(v.item())

        # GAEでアドバンテージ計算
        rews = np.array(self.buffer.rews, dtype=np.float32)
        vals = np.array(self.buffer.vals, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.bool_)

        advs = np.zeros_like(rews, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rews))):
            next_v = next_val if t == len(rews) - 1 else vals[t+1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rews[t] + GAMMA * next_v * nonterminal - vals[t]
            last_gae = delta + GAMMA * LAMBDA * nonterminal * last_gae
            advs[t] = last_gae
        returns = advs + vals

        # テンソル化
        obs_t = to_tensor(np.array(self.buffer.obs, dtype=np.float32), self.device)
        acts_t = torch.as_tensor(np.array(self.buffer.acts), dtype=torch.int64, device=self.device)
        old_logp_t = to_tensor(np.array(self.buffer.logps, dtype=np.float32), self.device)
        adv_t = to_tensor(advs, self.device)
        ret_t = to_tensor(returns, self.device)

        # 標準化（安定化）
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        # ミニバッチ学習
        N = obs_t.size(0)
        idxs = np.arange(N)
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idxs)
            for mb in np.array_split(idxs, MINI_BATCHES):
                mb = np.asarray(mb)
                logits, v = self.model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts_t[mb])
                ratio = torch.exp(logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((v.squeeze(-1) - ret_t[mb]) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        # クリア
        self.buffer.clear()

    # ---------- 終了処理 ----------
    def finalize(self) -> pd.DataFrame:
        """
        学習/シミュレーションを一旦区切る。
        戻り値: 結果DataFrame（Time, Price, Action, Profit）
        返した後、内部結果をリセット。
        """
        df = pd.DataFrame(self.results, columns=RESULTS_COLUMNS)
        # 内部状態は継続（モデル/建玉/特徴量）は通常継続しますが、
        # 別日データでの学習を想定する場合は、必要に応じてFeatureBuilderや建玉をリセットしてください。
        self.results = []
        return df

# ====== 単体テスト用（任意）======
if __name__ == "__main__":
    # ダミーデータで簡易動作確認
    sim = TradingSimulation()
    ts0 = 1_700_000_000
    price = 5000.0
    vol = 0.0
    for i in range(180):  # 3分ぶん
        ts = ts0 + i
        price += np.random.randn() * 2.0
        vol += abs(np.random.randn()) * 1000.0
        force = (i == 179)
        action = sim.add(ts, float(price), float(vol), force_close=force)
        # print(ts, price, action)
    out = sim.finalize()
    print(out.tail())
