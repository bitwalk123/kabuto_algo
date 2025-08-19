# tick_rl_trader.py
"""
Tick-level DQN trader (PyTorch).
Usage:
    python tick_rl_trader.py --data_paths path/tick_20250808_7011.csv,path/tick_20250731_7011.csv
"""

import argparse
import random
import os
from collections import deque, namedtuple
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Hyperparameters
# -------------------------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 200_000
TARGET_UPDATE_FREQ = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200_000  # steps
K_HISTORY = 50  # number of prior ticks to include in state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# -------------------------
# Environment
# -------------------------
class TickEnv:
    """
    Single-day tick environment.
    CSV must have columns: Time (seconds), Price (yen)
    Business rules enforced here:
      - trade unit = 100 shares
      - one position at a time, no pyramiding
      - profit-taking (intraday close) allowed only if profit_per_share >= 3 yen
      - forced liquidation at day end
    """

    def __init__(self, df, k_history=K_HISTORY, profit_ticks=3):
        self.df = df.reset_index(drop=True)
        self.prices = self.df['Price'].values.astype(float)
        self.t = 0
        self.k = k_history
        self.profit_ticks = profit_ticks
        self.n = len(self.prices)
        # position: None or dict{'entry_price': float, 'size': int}
        self.position = None
        self.trade_count = 0
        self.total_reward = 0.0

    def reset(self):
        self.t = 0
        self.position = None
        self.trade_count = 0
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self):
        # Build history window of length k: use log returns normalized
        start = max(0, self.t - self.k + 1)
        window = self.prices[start:self.t + 1]
        if len(window) < self.k:
            # pad with first price
            pad = np.full(self.k - len(window), window[0] if len(window) > 0 else 0.0)
            window = np.concatenate([pad, window])
        # convert to returns
        returns = np.diff(window)  # length k-1
        # normalize by recent std to avoid scale issues
        std = (np.std(returns) + 1e-6)
        norm_returns = returns / std
        # position flag
        pos_flag = 1.0 if self.position is not None else 0.0
        # time feature (progress through day)
        time_feat = np.array([self.t / max(1, self.n - 1)])
        state = np.concatenate([norm_returns, [self.prices[self.t]], [pos_flag], time_feat]).astype(np.float32)
        return state  # dim = (k-1)+1+1+1 = k+2

    def step(self, action):
        """
        action: 0 hold, 1 open_long, 2 close_long
        returns: next_state, reward, done, info
        """
        price = self.prices[self.t]
        reward = 0.0
        info = {'executed': False, 'price': price}

        # open_long
        if action == 1:
            if self.position is None:
                # open at current price, 100 shares
                self.position = {'entry_price': price, 'size': 100}
                self.trade_count += 1
                info['executed'] = True
        # close_long
        elif action == 2:
            if self.position is not None:
                entry = self.position['entry_price']
                profit_per_share = price - entry
                # check intraday profit-taking rule: allow only if profit >= profit_ticks (yen)
                if profit_per_share >= self.profit_ticks:
                    # close
                    pnl = profit_per_share * self.position['size']
                    reward = pnl
                    self.total_reward += reward
                    self.position = None
                    self.trade_count += 1
                    info['executed'] = True
                else:
                    # disallow close (treated as hold)
                    info['executed'] = False
        # hold -> nothing

        # step time forward
        done = False
        self.t += 1
        if self.t >= self.n:
            # end of day: forced liquidation if position exists (liquidate at last observed price)
            done = True
            # last price used is previous index (self.n-1)
            last_price = self.prices[-1]
            if self.position is not None:
                entry = self.position['entry_price']
                profit_per_share = last_price - entry
                pnl = profit_per_share * self.position['size']
                reward += pnl
                self.total_reward += pnl
                self.position = None
                self.trade_count += 1
                info['forced_liquidation'] = True
            next_state = np.zeros_like(self._get_state())  # terminal state (not used)
        else:
            next_state = self._get_state()

        return next_state, reward, done, info


# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# -------------------------
# DQN Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Agent (DQN)
# -------------------------
class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=LR, device=DEVICE):
        self.device = device
        self.policy_net = DQN(state_dim, n_actions).to(device)
        self.target_net = DQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
        self.n_actions = n_actions

    def select_action(self, state, eps_threshold):
        # state: numpy array
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qvals = self.policy_net(t)
                return int(qvals.argmax(dim=1).item())

    def optimize(self, batch):
        # batch is Transition of tuples
        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        non_final_mask = torch.tensor([not d for d in batch.done], device=self.device)
        non_final_next_states = torch.tensor(
            [s for s, d in zip(batch.next_state, batch.done) if not d],
            dtype=torch.float32, device=self.device
        ) if any(not d for d in batch.done) else torch.empty((0, states.shape[1]), device=self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = torch.zeros((len(batch.state), 1), device=self.device)

        if non_final_next_states.shape[0] > 0:
            next_q = self.target_net(non_final_next_states).max(1)[0].detach().unsqueeze(1)
            next_q_values[non_final_mask] = next_q

        expected_q = rewards + (GAMMA * next_q_values)
        loss = nn.functional.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        # Save state_dict (recommended method for PyTorch >=2.6)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data['policy_state_dict'])
        self.target_net.load_state_dict(data['policy_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.steps_done = data.get('steps_done', 0)


# -------------------------
# Utilities
# -------------------------
def load_csv(path):
    df = pd.read_csv(path)
    assert 'Price' in df.columns, "CSV must contain Price column"
    return df[['Time', 'Price']]


def linear_eps(step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step / EPS_DECAY)


# -------------------------
# Training Loop
# -------------------------
def train_loop(data_paths, n_epochs=3, save_every=1):
    # initialize agent with dummy env to get dims
    sample_df = load_csv(data_paths[0])
    env = TickEnv(sample_df)
    state_dim = env._get_state().shape[0]
    n_actions = 3
    agent = DQNAgent(state_dim, n_actions)
    replay = ReplayBuffer()

    global_step = 0
    stats = []

    for epoch in range(n_epochs):
        for file_idx, path in enumerate(data_paths):
            df = load_csv(path)
            env = TickEnv(df)
            state = env.reset()
            done = False
            day_reward = 0.0
            day_trades = 0
            step_in_episode = 0

            while not done:
                eps = linear_eps(agent.steps_done)
                action = agent.select_action(state, eps)
                next_state, reward, done, info = env.step(action)

                # push transition
                replay.push(state, action, reward, next_state, done)
                state = next_state

                # learning step
                if len(replay) >= BATCH_SIZE:
                    batch = replay.sample(BATCH_SIZE)
                    agent.optimize(batch)

                # target update
                if agent.steps_done % TARGET_UPDATE_FREQ == 0:
                    agent.update_target()

                if info.get('executed', False) or info.get('forced_liquidation', False):
                    # count trades and accumulate immediate reward
                    day_trades = env.trade_count
                day_reward = env.total_reward

                step_in_episode += 1
                global_step += 1

            # end of day
            print(
                f"Epoch {epoch + 1}/{n_epochs}  File {file_idx + 1}/{len(data_paths)}  Day: {os.path.basename(path)}  "
                f"TotalP&L={day_reward:.0f} yen  Trades={day_trades}")
            stats.append({'epoch': epoch + 1, 'file': path, 'pnl': day_reward, 'trades': day_trades})

        # save model after epoch
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(SAVE_DIR, f"dqn_state_epoch{epoch + 1}.pt")
            agent.save(save_path)
            print("Saved model to", save_path)

    return agent, stats


# -------------------------
# Evaluation (run once with a saved model)
# -------------------------
def evaluate_one_file(agent, path):
    df = load_csv(path)
    env = TickEnv(df)
    s = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            t = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            action = int(agent.policy_net(t).argmax(dim=1).item())
        s, r, done, info = env.step(action)
    return env.total_reward, env.trade_count


# -------------------------
# Main CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, required=True,
                        help='Comma separated list of CSV paths (path/tick_YYYYMMDD_CODE.csv)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_paths = args.data_paths.split(',')
    agent, stats = train_loop(data_paths, n_epochs=args.epochs, save_every=args.save_every)
    # print summary
    print("Training finished. Summary:")
    for s in stats:
        print(s)

    # save final model
    final_path = os.path.join(SAVE_DIR, "dqn_final.pt")
    agent.save(final_path)
    print("Saved final model to", final_path)


if __name__ == "__main__":
    main()
