# trading_simulation.py
# -*- coding: utf-8 -*-
"""
デイトレード用 強化学習シミュレータ（PPO-lite / Actor-Critic）
- 1銘柄固定、約1秒ティック、100株単位、信用売買可、同時保有は1ポジのみ（ナンピン禁止）
- add(ts, price, volume, force_close=False) を逐次呼び出すストリーミング型インターフェイス
- finalize() で結果 DataFrame を返して内部状態をリセット
- 価格は生値を受け取り、差分出来高→log1p圧縮を add 内で自動処理
- モデルは models/policy.pth に保存・再利用。無効なら自動上書き再生成
- 報酬設計：
    * 含み益が+200円以上で利確（返済）
    * 含み損が-500円以下で損切り（返済）
    * 非線形報酬ブースト：+1000円以上×2.0、+500円以上×1.5、+200円以上×1.0、含み益>0で微ボーナス、含み損<0で微ペナルティ
"""

from __future__ import annotations
import os
import math
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# ユーティリティ
# ============================================================
class OnlineNormalizer:
    """ランニング平均・分散で正規化（Welford法）"""
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
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.eps)

    def normalize(self, x: float) -> float:
        if self.n < 2:
            return 0.0
        return (x - self.mean) / self.std


# ============================================================
# 方策・価値ネットワーク（PPO-lite）
# ============================================================
class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, 3)   # 行動：0=ホールド/維持, 1=新規買い or 返済買い, 2=新規売り or 返済売り
        self.v  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    logp: float
    reward: float
    done: bool
    value: float


# ============================================================
# シミュレーションクラス
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
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # オブザベーション: [price_norm, ret_1s, ret_5s, ret_10s, log1p_dv_norm, pos_long, pos_short]
        self.obs_dim = 7
        self.net = PolicyValueNet(self.obs_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # ランニング正規化器
        self.price_norm = OnlineNormalizer()
        self.logdv_norm = OnlineNormalizer()

        # 内部状態
        self.reset_state()

        # PPOハイパラ
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_interval = update_interval
        self.mini_epochs = mini_epochs
        self.batch_size = batch_size

        self.buffer: List[Transition] = []

        # 既存モデルの読み込み or 新規作成
        self._load_or_init_model()

    # --------------------------------------------------------
    # 公開API
    # --------------------------------------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """1ティック受信 -> 行動決定 -> 取引実行 -> 報酬蓄積 -> 必要に応じ学習"""
        action_label = "ホールド"

        # 差分出来高
        delta_v = max(volume - self.prev_volume if self.prev_volume is not None else 0.0, 0.0)
        self.prev_volume = volume
        logdv = math.log1p(delta_v)

        # 価格正規化 & 収益指標
        self.price_norm.update(price)
        self.logdv_norm.update(logdv)
        pnorm = self.price_norm.normalize(price)

        # リターン算出（1s/5s/10s）
        ret_1s = 0.0 if self.prev_price is None else (price - self.prev_price) / max(self.prev_price, 1e-6)
        self.ret_buf.append(price)
        if len(self.ret_buf) > 10:
            self.ret_buf.pop(0)
        ret_5s = 0.0 if len(self.ret_buf) < 6 else (self.ret_buf[-1] - self.ret_buf[-6]) / max(self.ret_buf[-6], 1e-6)
        ret_10s= 0.0 if len(self.ret_buf) < 11 else (self.ret_buf[-1] - self.ret_buf[0]) / max(self.ret_buf[0], 1e-6)
        self.prev_price = price

        logdv_norm = self.logdv_norm.normalize(logdv)

        pos_long = 1.0 if self.position == 1 else 0.0
        pos_short = 1.0 if self.position == -1 else 0.0

        obs = np.array([pnorm, ret_1s, ret_5s, ret_10s, logdv_norm, pos_long, pos_short], dtype=np.float32)

        # 方策から行動サンプル
        logits, value = self._forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()  # 0: hold/maintain, 1: buy(新規 or 返済買い), 2: sell(新規 or 返済売り)
        logp = dist.log_prob(torch.tensor(action, device=self.device)).item()
        value = value.item()

        # ポジション制約に基づいて実際のトレードを解釈
        realized = 0.0
        if self.position == 0:
            if action == 1:
                # 新規買い
                self.entry_price = price
                self.position = 1
                action_label = "新規買い"
            elif action == 2:
                # 新規売り（空売り）
                self.entry_price = price
                self.position = -1
                action_label = "新規売り"
            else:
                action_label = "ホールド"
        elif self.position == 1:
            # ロング保有中 → 0=ホールド, 2=返済売り に限定（1=新規買いは無効→ホールド扱い）
            if action == 2:
                realized = (price - self.entry_price) * 100.0
                self.position = 0
                self.entry_price = None
                action_label = "返済売り"
            else:
                action_label = "ホールド"
        elif self.position == -1:
            # ショート保有中 → 0=ホールド, 1=返済買い に限定（2=新規売りは無効→ホールド扱い）
            if action == 1:
                realized = (self.entry_price - price) * 100.0
                self.position = 0
                self.entry_price = None
                action_label = "返済買い"
            else:
                action_label = "ホールド"

        # 強制返済 or しきい値判定（自動クローズ）
        unrealized = 0.0
        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:
                unrealized = (price - self.entry_price) * 100.0
            else:
                unrealized = (self.entry_price - price) * 100.0

            # 自動クローズ（日内利確/損切り）
            if unrealized >= 200.0 or unrealized <= -500.0 or force_close:
                # 返済執行
                if self.position == 1:
                    realized = (price - self.entry_price) * 100.0
                    action_label = "返済売り(自動)" if not force_close else "返済売り(強制)"
                else:
                    realized = (self.entry_price - price) * 100.0
                    action_label = "返済買い(自動)" if not force_close else "返済買い(強制)"
                self.position = 0
                self.entry_price = None

        # 報酬の設計
        reward = 0.0
        # 直近の含み損益をベース（デルタではなくレベルでもOK。過度なスケール回避のため/1000）
        reward += (unrealized / 1000.0)
        # ブースト/ペナルティ
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
        # 返済時は確定損益を追加（スケール/1000）
        if realized != 0.0:
            reward += (realized / 1000.0)

        # バッファに保存
        self.buffer.append(Transition(obs=obs, action=action, logp=logp, reward=reward, done=False, value=value))

        # 結果テーブル
        self.results.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": action_label,
            "Profit": float(realized)  # 返済時のみ非0
        })

        # 学習（一定間隔）
        if len(self.buffer) >= self.update_interval:
            self._update()
            self._save_model()

        return action_label

    def finalize(self) -> pd.DataFrame:
        """結果をDataFrameで返して内部の結果バッファをクリア。学習も締める。"""
        # エピソード終端としてGAE計算を閉じる
        if len(self.buffer) > 0:
            # 末尾は done=True として扱う
            self.buffer[-1] = Transition(
                obs=self.buffer[-1].obs,
                action=self.buffer[-1].action,
                logp=self.buffer[-1].logp,
                reward=self.buffer[-1].reward,
                done=True,
                value=self.buffer[-1].value,
            )
            self._update()
            self._save_model()

        df = pd.DataFrame(self.results, columns=["Time", "Price", "売買アクション", "Profit"]).copy()
        # 状態のクリア（ただしモデルと正規化統計は維持して継続学習可）
        self.results.clear()
        self.buffer.clear()
        return df

    # --------------------------------------------------------
    # 内部ヘルパ
    # --------------------------------------------------------
    def reset_state(self):
        self.position: int = 0          # 0=ノーポジ, 1=ロング, -1=ショート
        self.entry_price: Optional[float] = None
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None
        self.ret_buf: List[float] = []
        self.results: List[dict] = []

    def _forward(self, obs_np: np.ndarray):
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        return logits.squeeze(0), value.squeeze(0)

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [a + v for a, v in zip(advantages, values)]
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def _update(self):
        if len(self.buffer) == 0:
            return
        obs = np.stack([t.obs for t in self.buffer])
        actions = np.array([t.action for t in self.buffer], dtype=np.int64)
        old_logp = np.array([t.logp for t in self.buffer], dtype=np.float32)
        rewards = np.array([t.reward for t in self.buffer], dtype=np.float32)
        dones = np.array([t.done for t in self.buffer], dtype=np.bool_)

        # 現在ネットで value を再評価（値関数 bootstrap 用に旧値を使う実装でも可）
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits, values = self.net(x)
            values = values.detach().cpu().numpy()

        adv, ret = self._compute_gae(rewards, values, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset = self._make_dataset(obs, actions, old_logp, adv, ret)

        for _ in range(self.mini_epochs):
            for batch in dataset:
                loss = self._ppo_loss(*batch)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.opt.step()

        # バッファを空に
        self.buffer.clear()

    def _make_dataset(self, obs, actions, old_logp, adv, ret):
        n = len(obs)
        idx = np.arange(n)
        self.rng.shuffle(idx)
        for i in range(0, n, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            x = torch.tensor(obs[batch_idx], dtype=torch.float32, device=self.device)
            a = torch.tensor(actions[batch_idx], dtype=torch.long, device=self.device)
            olp = torch.tensor(old_logp[batch_idx], dtype=torch.float32, device=self.device)
            ad = torch.tensor(adv[batch_idx], dtype=torch.float32, device=self.device)
            rt = torch.tensor(ret[batch_idx], dtype=torch.float32, device=self.device)
            yield x, a, olp, ad, rt

    def _ppo_loss(self, x, a, old_logp, adv, ret):
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        ratio = torch.exp(logp - old_logp)

        # クリップドサロゲート
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        # 価値関数損失
        value_loss = F.mse_loss(value, ret)

        # エントロピー（探索促進）
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        return loss

    def _load_or_init_model(self):
        status = None
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.net.load_state_dict(state["model"])  # 期待：{"model": state_dict}
                self.opt.load_state_dict(state.get("opt", self.opt.state_dict()))
                status = "既存モデルを読み込みました。"
            except Exception as e:
                print(f"既存モデルの読み込みに失敗: {e}. 新規に作成して上書きします。")
                self._save_model()
                status = "既存モデルが無効だったため、新規モデルを作成し上書きしました。"
        else:
            status = "有効な既存モデルが見つからないため、新規モデルを作成しました。"
            self._save_model()
        print(status)

    def _save_model(self):
        state = {
            "model": self.net.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(state, self.model_path)


# 参考：スタンドアロンでの簡易テスト（任意）
if __name__ == "__main__":
    # ダミーデータでの疎通確認
    sim = TradingSimulation()
    t = 0
    price = 5000.0
    volume = 0.0
    for i in range(1000):
        t += 1
        price += np.random.normal(0, 2)
        volume += max(0.0, np.random.poisson(50))
        force_close = (i == 999)
        sim.add(t, float(price), float(volume), force_close)
    df = sim.finalize()
    print(df.tail())
