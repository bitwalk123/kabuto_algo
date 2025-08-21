import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =============================
# Utility: Running Normalizer
# =============================
class RunningNorm:
    """Online feature standardizer using Welford's algorithm."""

    def __init__(self, eps: float = 1e-6):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var) + self.eps

    def normalize(self, x: float) -> float:
        if self.count < 10:
            # warmup: avoid extreme z-scores in the early steps
            return 0.0
        return (x - self.mean) / self.std


# =====================================
# PPO-lite (Actor-Critic without clip)
# =====================================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, n_actions: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, n_actions)  # logits for categorical policy
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.mu(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class TradeState:
    position: int = 0  # -1 short, 0 flat, +1 long
    entry_price: float = 0.0
    entry_ts: float = 0.0


class TradingSimulation:
    """
    Streaming tick-by-tick trading simulator with a lightweight Actor-Critic (PPO-lite) learner.

    Interface:
      - add(ts: float, price: float, volume: float, force_close: bool=False) -> str
      - finalize() -> pd.DataFrame  # returns and resets results buffer

    Key constraints enforced:
      * Single instrument, tick data input (no OHLC conversion)
      * Order size fixed to 100 shares, no averaging down/up (one position at a time)
      * Short selling allowed; fill price assumed to be current price (mid/last)
      * No commission considered
    """

    def __init__(
        self,
        model_path: str = "models/ppo_lite_trader.pt",
        device: Optional[str] = None,
        gamma: float = 0.995,
        lr: float = 3e-4,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        grad_clip: float = 1.0,
        target_hold_s: float = 300.0,  # reward peaks around 300s in position
        lots: int = 100,
        min_close_bonus_yen: float = 500.0,
    ):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Feature engineering state
        self.prev_price: Optional[float] = None
        self.prev_volume: Optional[float] = None  # cumulative volume
        self.start_ts: Optional[float] = None
        self.elapsed_s: float = 0.0

        # Online normalizers for key features
        self.norm_ret = RunningNorm()
        self.norm_dv = RunningNorm()

        # Trading state
        self.state = TradeState()
        self.lots = lots

        # RL params
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.target_hold_s = target_hold_s
        self.min_close_bonus_yen = min_close_bonus_yen

        # Action space: 0=HOLD, 1=OPEN_LONG (only if flat), 2=OPEN_SHORT (only if flat), 3=CLOSE (only if in position)
        self.n_actions = 4
        self.obs_dim = 8

        # Build/load model
        self.net = PolicyNet(self.obs_dim, hidden=128, n_actions=self.n_actions).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self._load_or_init_model()

        # Last transition for online A2C update
        self.last_obs: Optional[torch.Tensor] = None
        self.last_value: Optional[torch.Tensor] = None
        self.last_action: Optional[int] = None
        self.last_logp: Optional[torch.Tensor] = None

        # Results buffer for finalize()
        self.rows: List[dict] = []

        # For reproducibility in a long session
        torch.set_num_threads(max(1, torch.get_num_threads()))

    # ---------------------------
    # Model persistence utilities
    # ---------------------------
    def _load_or_init_model(self):
        reason = ""
        if os.path.exists(self.model_path):
            try:
                payload = torch.load(self.model_path, map_location=self.device)
                state_dict = payload.get("state_dict")
                meta = payload.get("meta", {})
                if state_dict is not None and meta.get("obs_dim") == self.obs_dim and meta.get("n_actions") == self.n_actions:
                    self.net.load_state_dict(state_dict)
                    reason = "既存モデルを読み込みました"  # loaded
                else:
                    reason = "既存モデルが不整合のため再生成します"
                    self._reset_model()
            except Exception:
                reason = "既存モデルが破損しているため再生成します"
                self._reset_model()
        else:
            reason = "学習モデルが存在しないため新規生成します"
            self._reset_model()
        print(f"[Model] {reason}")

    def _reset_model(self):
        self.net = PolicyNet(self.obs_dim, hidden=128, n_actions=self.n_actions).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.opt.param_groups[0]['lr'] if len(self.opt.param_groups) > 0 else 3e-4)
        self._save_model(tag="init")

    def _save_model(self, tag: str = "auto"):
        payload = {
            "state_dict": self.net.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "timestamp": time.time(),
                "tag": tag,
            },
        }
        torch.save(payload, self.model_path)

    # ---------------------------
    # Feature engineering (done inside add())
    # ---------------------------
    def _features(self, ts: float, price: float, volume_cum: float) -> np.ndarray:
        if self.start_ts is None:
            self.start_ts = ts
        self.elapsed_s = max(0.0, ts - self.start_ts)

        # Return (price change ratio) vs previous tick
        if self.prev_price is None:
            ret = 0.0
        else:
            # Safe return using relative change
            ret = (price - self.prev_price) / max(1e-6, self.prev_price)
        self.norm_ret.update(ret)
        zret = self.norm_ret.normalize(ret)

        # Delta volume in this second, guard for resets
        if self.prev_volume is None or volume_cum < self.prev_volume:
            dv = 0.0
        else:
            dv = float(volume_cum - self.prev_volume)
        dv_log = math.log1p(max(0.0, dv))  # compress magnitude
        self.norm_dv.update(dv_log)
        zdv = self.norm_dv.normalize(dv_log)

        # Time-of-day cyclical features from elapsed seconds
        # Map elapsed within a day; if ts already seconds-of-day this still works
        day_sec = 24 * 3600.0
        theta = 2 * math.pi * ((self.elapsed_s % day_sec) / day_sec)
        sin_t, cos_t = math.sin(theta), math.cos(theta)

        # Position features
        pos = self.state.position
        holding_s = (self.elapsed_s - self.state.entry_ts) if pos != 0 else 0.0
        hold_feat = math.tanh(holding_s / 300.0)  # saturates ~1 after ~5min

        # Unrealized PnL per 100 shares (thousand-yen scaled)
        if pos == 0:
            upnl = 0.0
        else:
            upnl = (price - self.state.entry_price) * self.lots * (1 if pos > 0 else -1)
        upnl_scaled = upnl / 1000.0

        obs = np.array([
            ret, zret,
            dv_log, zdv,
            float(pos), hold_feat,
            upnl_scaled,
            sin_t + cos_t,  # compact time feature (2 -> 1 dim)
        ], dtype=np.float32)

        # Update prev trackers AFTER computing features
        self.prev_price = price
        self.prev_volume = volume_cum
        return obs

    # ---------------------------
    # Reward shaping
    # ---------------------------
    def _holding_shaping(self, holding_s: float) -> float:
        # Peak around target_hold_s, then decay. Use Gaussian bump.
        if holding_s <= 0:
            return 0.0
        sigma = self.target_hold_s / 2.0
        return math.exp(-((holding_s - self.target_hold_s) ** 2) / (2 * sigma * sigma))

    def _compute_reward(
        self,
        price: float,
        next_price: Optional[float],
        action: int,
        force_close: bool,
    ) -> float:
        pos = self.state.position
        holding_s = (self.elapsed_s - self.state.entry_ts) if pos != 0 else 0.0

        # Unrealized PnL delta as dense reward while holding
        r_upnl = 0.0
        if pos != 0 and self.prev_price is not None and next_price is not None:
            # change since last step
            dprice = (next_price - self.prev_price)
            r_upnl = dprice * self.lots * (1 if pos > 0 else -1)

        # Holding-time shaping reward (encourages ~300s holds when profitable)
        r_hold = self._holding_shaping(holding_s) * 0.5  # scale

        # Realized reward on CLOSE
        r_realized = 0.0
        close_bonus = 0.0
        if (action == 3 or force_close) and pos != 0:
            realized = (price - self.state.entry_price) * self.lots * (1 if pos > 0 else -1)
            r_realized = realized
            if realized >= self.min_close_bonus_yen:  # encourage >= +500 JPY profits
                close_bonus = 50.0  # extra bonus (tunable)
            elif abs(realized) < 10.0:
                # discourage churning around zero
                close_bonus = -10.0

        reward = r_upnl + r_hold + r_realized + close_bonus
        return float(reward)

    # ---------------------------
    # Action selection with masking
    # ---------------------------
    def _masked_action(self, obs_t: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.net(obs_t)
        # Mask unavailable actions depending on position state
        # flat: allow [HOLD, OPEN_LONG, OPEN_SHORT], disallow CLOSE
        # in position: allow [HOLD, CLOSE], disallow opening actions
        pos = self.state.position
        mask = torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
        if pos == 0:
            mask[3] = False
        else:
            mask[1] = False
            mask[2] = False
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        probs = torch.distributions.Categorical(logits=masked_logits)
        action = int(probs.sample().item())
        logp = probs.log_prob(torch.tensor(action, device=self.device))
        return action, logp, value, logits

    # ---------------------------
    # Public API: add() and finalize()
    # ---------------------------
    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        """
        Accept raw tick (ts, price, cumulative volume). Return action label in Japanese.
        """
        obs_np = self._features(ts, price, volume)
        obs_t = torch.from_numpy(obs_np).to(self.device)

        # Choose action (respect force_close)
        if force_close and self.state.position != 0:
            action = 3  # CLOSE
            with torch.no_grad():
                logits, value = self.net(obs_t)
            logp = torch.tensor(0.0, device=self.device)  # dummy (no policy grad for forced close)
        else:
            action, logp, value, _ = self._masked_action(obs_t)

        # Compute immediate reward using next_price approximation (we'll reuse current price as next for streaming)
        reward = self._compute_reward(price=price, next_price=price, action=action, force_close=force_close)

        # Apply environment transition (update position based on action)
        profit_realized = 0.0
        action_label = "様子見"
        if action == 1 and self.state.position == 0:  # OPEN_LONG
            self.state.position = +1
            self.state.entry_price = price
            self.state.entry_ts = self.elapsed_s
            action_label = "新規買い"
        elif action == 2 and self.state.position == 0:  # OPEN_SHORT
            self.state.position = -1
            self.state.entry_price = price
            self.state.entry_ts = self.elapsed_s
            action_label = "新規売り"
        elif action == 3 and self.state.position != 0:  # CLOSE
            realized = (price - self.state.entry_price) * self.lots * (1 if self.state.position > 0 else -1)
            profit_realized = float(realized)
            action_label = "返済（売）" if self.state.position > 0 else "返済（買）"
            # reset position
            self.state = TradeState()
        else:
            action_label = "様子見"

        # Online A2C update (PPO-lite): one-step TD target
        # Skip policy gradient on forced close
        if not (force_close and action == 3):
            self._train_step(obs_t, action, logp, value, reward)

        # Log row
        self.rows.append({
            "Time": float(ts),
            "Price": float(price),
            "売買アクション": action_label,
            "Profit": float(profit_realized),  # non-zero only when position closes
        })

        # Periodically autosave model
        if len(self.rows) % 1000 == 0:
            self._save_model(tag="autosave")

        return action_label

    def _train_step(self, obs_t: torch.Tensor, action: int, logp: torch.Tensor, value: torch.Tensor, reward: float):
        self.net.train()
        # bootstrap value of next state approximated as current state's value (streaming simplification)
        with torch.no_grad():
            _, next_value = self.net(obs_t)
        advantage = torch.tensor(reward, device=self.device) + self.gamma * next_value.detach() - value.detach()

        # Recompute log prob for the chosen action (for stability)
        logits, _ = self.net(obs_t)
        pos = self.state.position
        mask = torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
        if pos == 0:
            mask[3] = False
        else:
            mask[1] = False
            mask[2] = False
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        logp_new = dist.log_prob(torch.tensor(action, device=self.device))
        entropy = dist.entropy()

        policy_loss = -(logp_new * advantage.detach())
        value_loss = (value - (advantage + value.detach()))**2  # (V - target)^2
        loss = policy_loss + self.value_coef * value_loss.mean() - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

    def finalize(self) -> pd.DataFrame:
        """Return results DataFrame and reset only the results buffer (not the model)."""
        df = pd.DataFrame(self.rows, columns=["Time", "Price", "売買アクション", "Profit"])
        # Reset rows only; keep the trained model and trading/running state for next epoch/day
        self.rows = []
        # Save model on finalize to persist learning between days/epochs
        self._save_model(tag="finalize")
        return df


# ===== Optional: quick self-test =====
if __name__ == "__main__":
    # Minimal smoke test with synthetic ticks
    sim = TradingSimulation()
    ts0 = time.time()
    v = 0.0
    price = 1000.0
    rng = np.random.default_rng(0)

    for i in range(500):
        ts = ts0 + i
        # random walk price
        price += rng.normal(0, 0.5)
        v += rng.integers(0, 50)
        force_close = (i == 499)
        sim.add(ts, float(price), float(v), force_close=force_close)
    out = sim.finalize()
    print(out.tail())
