"""
Intraday RL trading example (single-file, runnable in JupyterLab).
Fixed to use state_dict save/load for PyTorch 2.6+ compatibility.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class TradingEnv:
    def __init__(self, prices, window=50, lot=100):
        self.prices = np.array(prices, dtype=np.float32)
        self.n = len(self.prices)
        self.window = window
        self.lot = lot
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0
        self.entry_price = 0.0
        self.done = False
        return self._get_obs()

    def step(self, action):
        reward = 0.0
        info = {}
        price = float(self.prices[self.t])
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            info['exec'] = 'enter_long'
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
            info['exec'] = 'enter_short'
        elif action == 3 and self.position != 0:
            pnl = self.position * (price - self.entry_price) * self.lot
            reward = pnl
            info['exec'] = 'close'
            info['pnl'] = pnl
            self.position = 0
            self.entry_price = 0.0
        self.t += 1
        if self.t >= self.n:
            self.done = True
            if self.position != 0:
                last_price = price
                pnl = self.position * (last_price - self.entry_price) * self.lot
                reward += pnl
                info['exec'] = info.get('exec', '') + '|forced_close'
                info['pnl_forced'] = pnl
                self.position = 0
                self.entry_price = 0.0
            return self._get_obs(), reward, True, info
        return self._get_obs(), reward, False, info

    def _get_obs(self):
        start = max(0, self.t - self.window + 1)
        w = self.prices[start:self.t + 1]
        if len(w) < self.window:
            pad = np.ones(self.window - len(w), dtype=np.float32) * w[0]
            w = np.concatenate([pad, w])
        logp = np.log(w + 1e-8)
        ret = np.diff(logp)
        if np.std(ret) > 0:
            ret = (ret - np.mean(ret)) / (np.std(ret) + 1e-8)
        else:
            ret = ret - np.mean(ret)
        pos_flag = np.array([self.position], dtype=np.float32)
        return np.concatenate([ret.astype(np.float32), pos_flag])


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)


def collect_trajectory(env, policy, device):
    obs = env.reset()
    done = False
    traj = []
    while not done:
        obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = policy(obs_v)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logp = dist.log_prob(torch.tensor(action, device=device)).item()
        next_obs, reward, done, info = env.step(action)
        traj.append((obs, action, reward, logp, value.item()))
        obs = next_obs
    return traj


def compute_returns(traj, gamma=0.99):
    rewards = [t[2] for t in traj]
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def ppo_update(policy, optimizer, traj, returns, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, batch_size=64,
               device='cpu'):
    obs = torch.tensor([t[0] for t in traj], dtype=torch.float32, device=device)
    actions = torch.tensor([t[1] for t in traj], dtype=torch.long, device=device)
    old_logps = torch.tensor([t[3] for t in traj], dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    for _ in range(epochs):
        idxs = np.arange(len(traj))
        np.random.shuffle(idxs)
        for start in range(0, len(traj), batch_size):
            batch_idx = idxs[start:start + batch_size]
            b_obs = obs[batch_idx]
            b_actions = actions[batch_idx]
            b_old_logps = old_logps[batch_idx]
            b_returns = returns_t[batch_idx]
            logits, values = policy(b_obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logps = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            adv = b_returns - values.detach()
            ratio = torch.exp(logps - b_old_logps)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((b_returns - values) ** 2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def train_on_day(csv_path, model_path='models/actor_critic.pt', epochs=20, window=50, device='cpu'):
    df = pd.read_csv(csv_path)
    prices = df['Price'].values
    env = TradingEnv(prices, window=window)
    obs0 = env.reset()
    input_dim = len(obs0)
    policy = ActorCritic(input_dim=input_dim)
    if os.path.exists(model_path):
        print(f'Loading existing weights from {model_path}')
        policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    daily_total_rewards = []
    trade_counts = []
    for ep in range(epochs):
        traj = collect_trajectory(env, policy, device)
        returns = compute_returns(traj)
        trades = sum(1 for t in traj if t[2] != 0)
        total_reward = sum([t[2] for t in traj])
        ppo_update(policy, optimizer, traj, returns)
        daily_total_rewards.append(total_reward)
        trade_counts.append(trades)
        print(f"Epoch {ep + 1}/{epochs}: Reward={total_reward:.0f} JPY, Trades={trades}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"Saved weights to {model_path}")
    total_pnl, n_trades, trades_detail = evaluate_policy(
        csv_path,
        model_path,
        window=window,
        device=device,
        input_dim=input_dim,
    )
    """
    plt.figure(figsize=(8, 4))
    plt.plot(daily_total_rewards, marker='o')
    plt.title('Learning curve (per-epoch total reward on this day)')
    plt.xlabel('Epoch')
    plt.ylabel('Total reward (JPY)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """

    return daily_total_rewards, trade_counts, total_pnl, n_trades, trades_detail


def evaluate_policy(csv_path, model_path, window=50, device='cpu', input_dim=None):
    df = pd.read_csv(csv_path)
    prices = df['Price'].values
    env = TradingEnv(prices, window=window)
    policy = ActorCritic(input_dim=input_dim if input_dim else len(env.reset()))
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    obs = env.reset()
    done = False
    total_pnl = 0.0
    trades = 0
    trades_detail = []
    while not done:
        obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = policy(obs_v)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        action = int(np.argmax(probs))
        obs, reward, done, info = env.step(action)
        if reward != 0:
            total_pnl += reward
            trades += 1
            trades_detail.append({'index': env.t, 'pnl': reward, 'info': info})
    return total_pnl, trades, trades_detail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--model', default='models/actor_critic.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    rewards, trade_counts, pnl, ntrades, trades_detail = train_on_day(
        args.csv,
        model_path=args.model,
        epochs=args.epochs,
        window=args.window,
        device=args.device,
    )
    print('\n=== DAILY SUMMARY ===')
    print(f'Total P&L (evaluation): {pnl:.0f} JPY')
    print(f'Number of trades (evaluation): {ntrades}')
    print('Per-epoch rewards (training):', rewards)
    if len(trades_detail) > 0:
        outdf = pd.DataFrame(trades_detail)
        out_csv = os.path.splitext(args.csv)[0] + '_trades.csv'
        outdf.to_csv(out_csv, index=False)
        print(f'Saved trades detail to {out_csv}')
