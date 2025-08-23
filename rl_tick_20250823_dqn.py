#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingSimulation: ティックデータ用の軽量強化学習サンプル（表形式Q学習）

特徴:
- "別プログラム" から 1 秒間隔想定の生ティック [ts, price, volume(cum)] を add() に渡すだけ
- 累積出来高 volume は add() 内で差分化して np.log1p 変換
- ポジションは 100 株単位、ナンピン不可（建玉を解消するまで次の建玉は持たない）
- 現在株価を指値にして常に約定 (信用取引前提)
- 報酬は「含み益の増分 + 実現益」を基本に設計（含み益を報酬へ反映）
- 学習は表形式の ε-greedy Q 学習（超軽量）で、日々のティックで継続学習
- 既存モデルがあればロード、仕様不一致や破損時は自動で新規作成して上書き
- 結果は DataFrame に蓄積し、finalize() で返却（呼び出し毎にリセット）

依存:
- Python 3.9+
- numpy, pandas
- (任意) gymnasium==1.2.* があれば observation_space / action_space を定義

使い方: 下の if __name__ == "__main__": のデモ or ユーザ提示の別プログラムから呼び出し
"""
from __future__ import annotations
import os
import math
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:
    _HAS_GYM = False
    spaces = None  # type: ignore


# ============================
# 設定
# ============================
@dataclass
class SimConfig:
    lot_size: int = 100  # 売買単位（株）
    # 価格・出来高のオンライン標準化のための窓（Welford 法なので明示窓は持たないが、スケール護身用のクリップ値を用意）
    price_delta_clip: float = 10.0  # 価格変化のクリップ（円）
    vol_logdiff_clip: float = 12.0  # log1p(出来高差分) のクリップ

    # 状態離散化（ビン分割数）
    n_bins_price_delta: int = 21  # [-clip,+clip] を等分
    n_bins_vol_logdiff: int = 13  # [0, +clip] を等分
    # ポジション状態: 3 値 { -1:ショート, 0:ノーポジ, +1:ロング }

    # 学習ハイパーパラメータ
    alpha: float = 0.15      # 学習率
    gamma: float = 0.98      # 割引率
    epsilon_start: float = 0.10  # ε-greedy 初期値
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995  # add() 呼び出し毎に減衰

    # モデル保存
    model_path: str = "models/q_table_v1.npz"

    # 報酬設計
    reward_unrealized_coeff: float = 1.0   # 含み益の増分係数
    reward_realized_coeff: float = 1.0     # 実現益の係数（約定で確定した分）


# ============================
# ユーティリティ
# ============================
class OnlineStd:
    """Welford 法によるオンライン平均・分散。ここではスケール確認のみで使う。"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

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
        return math.sqrt(max(self.var, 1e-12))


# ============================
# TradingSimulation 本体
# ============================
class TradingSimulation:
    """
    売買アクションの定義（add() の返り値はこのラベルを返す）:
        0: "ホールド"           : 何もしない
        1: "新規買い"           : ノーポジ → ロング建て (100株)
        2: "新規売り"           : ノーポジ → ショート建て (100株)
        3: "返済売り"           : ロング → クローズ (100株 決済)
        4: "返済買い"           : ショート → クローズ (100株 決済)
    無効アクションは自動でマスク（例: ロング保有中に "新規買い" は選ばれない）
    """

    ACTION_LABELS = {
        0: "ホールド",
        1: "新規買い",
        2: "新規売り",
        3: "返済売り",
        4: "返済買い",
    }

    def __init__(self, config: Optional[SimConfig] = None):
        self.cfg = config or SimConfig()

        # Q テーブルの形状: [pos_state(3), price_bin, vol_bin, action(5)]
        self.n_pos = 3
        self.n_price = self.cfg.n_bins_price_delta
        self.n_vol = self.cfg.n_bins_vol_logdiff
        self.n_actions = 5

        # 内部状態
        self.reset_runtime_state()

        # Q テーブル読み込み or 初期化
        loaded = self._load_or_init_model()
        print(loaded)

        # gymnasium(optional)
        if _HAS_GYM:
            self.observation_space = spaces.MultiDiscrete([self.n_pos, self.n_price, self.n_vol])
            # 有効アクションは状態依存でマスクするため、ここは上限 5 とする
            self.action_space = spaces.Discrete(self.n_actions)

    # ----------------------------
    # ランタイム状態の初期化（エポック毎に維持、finalize() で結果はリセット）
    # ----------------------------
    def reset_runtime_state(self):
        self.prev_price: Optional[float] = None
        self.prev_vol: Optional[float] = None  # 累積出来高

        self.position: int = 0  # -1, 0, +1
        self.entry_price: Optional[float] = None

        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0

        self.price_std = OnlineStd()
        self.results: List[Dict] = []  # DataFrame 用バッファ

        # 学習用
        self.epsilon = self.cfg.epsilon_start
        self.prev_state_idx: Optional[Tuple[int, int, int]] = None
        self.prev_action: Optional[int] = None

    # ----------------------------
    # モデルのロード or 新規初期化
    # ----------------------------
    def _load_or_init_model(self) -> str:
        shape = (self.n_pos, self.n_price, self.n_vol, self.n_actions)
        os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
        if os.path.exists(self.cfg.model_path):
            try:
                data = np.load(self.cfg.model_path, allow_pickle=True)
                q = data["q_table"]
                meta = json.loads(data["meta"].item())  # type: ignore
                if q.shape != shape:
                    self.Q = np.zeros(shape, dtype=np.float32)
                    np.savez_compressed(self.cfg.model_path, q_table=self.Q, meta=json.dumps(asdict(self.cfg)))
                    return "既存モデルの形状が不一致 — 新規モデルを作成し上書きしました。"
                else:
                    self.Q = q.astype(np.float32, copy=False)
                    return "既存モデルを読み込みました。"
            except Exception as e:
                self.Q = np.zeros(shape, dtype=np.float32)
                np.savez_compressed(self.cfg.model_path, q_table=self.Q, meta=json.dumps(asdict(self.cfg)))
                return f"既存モデルが無効 — 新規モデルを作成しました。（理由: {e}）"
        else:
            self.Q = np.zeros(shape, dtype=np.float32)
            np.savez_compressed(self.cfg.model_path, q_table=self.Q, meta=json.dumps(asdict(self.cfg)))
            return "有効な既存モデルが見つからない — 新規モデルを作成しました。"

    # ----------------------------
    # 低レベル: 状態構築と離散化
    # ----------------------------
    def _compute_features(self, price: float, vol_cum: float) -> Tuple[float, float]:
        # 価格差分（円）
        if self.prev_price is None:
            price_delta = 0.0
        else:
            price_delta = float(price - self.prev_price)
        price_delta = float(np.clip(price_delta, -self.cfg.price_delta_clip, self.cfg.price_delta_clip))
        self.price_std.update(price_delta)

        # 出来高: 累積→差分→log1p
        if self.prev_vol is None:
            vol_diff = 0.0
        else:
            vol_diff = max(0.0, float(vol_cum - self.prev_vol))
        vol_logdiff = float(np.clip(np.log1p(vol_diff), 0.0, self.cfg.vol_logdiff_clip))
        return price_delta, vol_logdiff

    def _discretize(self, price_delta: float, vol_logdiff: float) -> Tuple[int, int, int]:
        # price_delta ∈ [-clip, +clip] → n_bins 等分
        pd_clip = self.cfg.price_delta_clip
        pv = (price_delta + pd_clip) / (2 * pd_clip)  # [0,1]
        price_bin = int(np.clip(int(pv * self.n_price), 0, self.n_price - 1))

        # vol_logdiff ∈ [0, clip] → n_bins 等分
        vv = vol_logdiff / self.cfg.vol_logdiff_clip  # [0,1]
        vol_bin = int(np.clip(int(vv * self.n_vol), 0, self.n_vol - 1))

        pos_idx = { -1: 0, 0: 1, +1: 2 }[self.position]
        return pos_idx, price_bin, vol_bin

    # ----------------------------
    # 有効アクションのマスク
    # ----------------------------
    def _valid_actions(self, force_close: bool) -> List[int]:
        if self.position == 0:
            acts = [0, 1, 2]  # ホールド/新規買い/新規売り
        elif self.position == +1:
            acts = [0, 3]     # ホールド/返済売り
        else:  # -1
            acts = [0, 4]     # ホールド/返済買い

        if force_close:
            if self.position == +1:
                return [3]
            elif self.position == -1:
                return [4]
            else:
                return [0]
        return acts

    # ----------------------------
    # 行動選択（ε-greedy, 無効アクションは -inf 扱い）
    # ----------------------------
    def _select_action(self, state_idx: Tuple[int, int, int], valid: List[int]) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))
        qvals = self.Q[state_idx]
        mask = np.full(self.n_actions, -np.inf, dtype=np.float32)
        mask[valid] = qvals[valid]
        return int(np.nanargmax(mask))

    # ----------------------------
    # Q 学習アップデート
    # ----------------------------
    def _q_update(self, s0: Tuple[int,int,int], a0: int, r: float, s1: Tuple[int,int,int], valid_next: List[int]):
        q = self.Q
        best_next = np.max(q[s1][valid_next]) if len(valid_next) else 0.0
        td_target = r + self.cfg.gamma * best_next
        td_error = td_target - q[s0][a0]
        q[s0][a0] += self.cfg.alpha * td_error

    # ----------------------------
    # 収益・報酬計算
    # ----------------------------
    def _calc_unrealized(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position == +1:
            return (price - self.entry_price) * self.cfg.lot_size
        else:  # -1
            return (self.entry_price - price) * self.cfg.lot_size

    def _realize_if_close(self, action: int, price: float) -> float:
        realized = 0.0
        if action == 3 and self.position == +1 and self.entry_price is not None:
            realized = (price - self.entry_price) * self.cfg.lot_size
            self.position = 0
            self.entry_price = None
        elif action == 4 and self.position == -1 and self.entry_price is not None:
            realized = (self.entry_price - price) * self.cfg.lot_size
            self.position = 0
            self.entry_price = None
        return realized

    def _open_if_entry(self, action: int, price: float):
        if action == 1 and self.position == 0:
            self.position = +1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price

    # ----------------------------
    # 公開 API: add()
    # ----------------------------
    def add(self, ts: float, price: float, volume: float, *, force_close: bool=False) -> str:
        """
        別プログラムから 1 ティック毎に呼び出す。返り値は売買アクションの日本語ラベル。
        パラメータ:
            ts: タイムスタンプ（float, 秒）
            price: 株価（円）
            volume: 累積出来高（寄り付きからの合計, float で可）
            force_close: True の場合は建玉を強制決済（最終行想定）
        """
        # 特徴量
        price_delta, vol_logdiff = self._compute_features(price, volume)
        state_idx = self._discretize(price_delta, vol_logdiff)

        # 有効アクションと行動選択
        valid = self._valid_actions(force_close)
        action = self._select_action(state_idx, valid)

        # 約定・損益処理（指値=現在値で常に成立）
        prev_unreal = self.unrealized_pnl
        realized = self._realize_if_close(action, price)
        self._open_if_entry(action, price)
        self.unrealized_pnl = self._calc_unrealized(price)

        # 報酬 = 含み益の「増分」 + 実現益
        unrealized_gain = (self.unrealized_pnl - prev_unreal)
        reward = (
            self.cfg.reward_unrealized_coeff * unrealized_gain
            + self.cfg.reward_realized_coeff * realized
        )
        self.realized_pnl += realized

        # 次状態（オンポリシー風に: 即時に次状態を構築）
        next_state_idx = state_idx  # ここでは 1 ティック内で状態は同一とみなす
        next_valid = self._valid_actions(False)

        # 学習更新
        if self.prev_state_idx is not None and self.prev_action is not None:
            self._q_update(self.prev_state_idx, self.prev_action, reward, next_state_idx, next_valid)
        self.prev_state_idx = state_idx
        self.prev_action = action

        # ε 減衰
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

        # ログ: DataFrame バッファへ
        self.results.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": self.ACTION_LABELS[action],
            "Reward": float(reward),
            "Profit": float(self.realized_pnl),
        })

        # 次ティックへ
        self.prev_price = price
        self.prev_vol = volume

        # モデル保存（適度に間引くならここを条件付きに）
        self._save_model()

        return self.ACTION_LABELS[action]

    # ----------------------------
    # finalize(): 結果取得 & バッファリセット
    # ----------------------------
    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results, columns=["Time", "Price", "売買アクション", "Reward", "Profit"])
        # ランタイムの結果だけリセット。学習済み Q は保持
        self.results = []
        self.prev_state_idx = None
        self.prev_action = None
        return df

    # ----------------------------
    # モデル保存
    # ----------------------------
    def _save_model(self):
        np.savez_compressed(self.cfg.model_path, q_table=self.Q, meta=json.dumps(asdict(self.cfg)))


# ============================
# デモ（任意）: ランダム擬似ティック
# ============================
if __name__ == "__main__":
    # 本ファイル単体でも簡易デモが動くようにしています（本番は提示の別プログラムを使用）。
    rng = np.random.default_rng(42)
    base_price = 3500.0
    n_ticks = 300

    sim = TradingSimulation()

    price = base_price
    vol_cum = 0.0
    for i in range(n_ticks):
        # ランダムウォーク + 小さなジャンプ
        price += rng.normal(0, 2.0) + (rng.random() < 0.02) * rng.normal(0, 10.0)
        price = float(max(1.0, price))
        # 出来高は右肩上がり
        vol_cum += float(abs(rng.normal(8000, 3000)))

        force_close = (i == n_ticks - 1)
        action = sim.add(ts=1.0 * i, price=price, volume=vol_cum, force_close=force_close)
        if (i + 1) % 50 == 0:
            print(f"tick {i+1}: action={action}")

    df_res = sim.finalize()
    print(df_res.tail())
    out = "results/demo_results.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_res.to_csv(out, index=False)
    print("結果を保存:", out)
