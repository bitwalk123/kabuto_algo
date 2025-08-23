import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TradingSimulation:
    def __init__(self, model_path='model.pth'):
        self.model_path = model_path
        self.input_dim = 5
        self.output_dim = 4  # hold:0, buy:1, sell:2, close:3
        self.actor = ActorNet(self.input_dim, self.output_dim)
        self.critic = CriticNet(self.input_dim)
        loaded = False
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path)
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                print("Using existing model.")
                loaded = True
            except Exception as e:
                print("Existing model invalid, creating new.")
        if not loaded:
            print("Creating new model.")
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=1e-3
        )
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.position = 0  # 0: none, 1: long, -1: short
        self.entry_price = 0.0
        self.results = []
        self.prev_volume = 0.0
        self.prev_price = 0.0
        self.first_tick = True

    def compute_reward(self, pnl):
        if pnl >= 1000:
            return pnl * 2.0
        elif pnl >= 500:
            return pnl * 1.5
        elif pnl >= 200:
            return pnl * 1.0
        elif pnl > 0:
            return pnl + 10.0
        else:
            return pnl - 10.0

    def add(self, ts, price, volume, force_close=False):
        if self.first_tick:
            self.prev_volume = volume
            self.prev_price = price
            self.first_tick = False
            delta_volume = 0.0
            price_change = 0.0
        else:
            delta_volume = volume - self.prev_volume
            price_change = price - self.prev_price
        self.prev_volume = volume
        self.prev_price = price
        log_delta = np.log1p(max(0, delta_volume))
        unreal_pnl = 0.0
        if self.position != 0:
            unreal_pnl = (price - self.entry_price) * self.position * 100
        state = np.array([
            price / 10000.0,
            price_change / 100.0,
            log_delta / 20.0,
            float(self.position),
            unreal_pnl / 10000.0
        ], dtype=np.float32)
        state_t = torch.from_numpy(state)
        with torch.no_grad():
            logits = self.actor(state_t)
            value = self.critic(state_t)
            # Mask invalid actions
            if self.position == 0:
                logits[3] = float('-inf')  # cannot close
            else:
                logits[1] = float('-inf')  # cannot buy
                logits[2] = float('-inf')  # cannot sell
                mask_close = (0 <= unreal_pnl < 200) and not force_close
                if mask_close:
                    logits[3] = float('-inf')  # force hold if small positive pnl
            if force_close and self.position != 0:
                logits[0] = float('-inf')  # force close, cannot hold
            probs = torch.softmax(logits, dim=0)
            if torch.all(torch.isinf(probs) | torch.isnan(probs)):
                action = 0  # fallback to hold
                logprob = torch.tensor(0.0)
            else:
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                logprob = action_dist.log_prob(torch.tensor(action))
            self.states.append(state_t)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.values.append(value)

        # Execute action
        action_str = ['hold', 'buy', 'sell', 'close'][action]
        profit = 0.0
        reward = 0.0
        if force_close and self.position != 0:
            profit = (price - self.entry_price) * self.position * 100
            reward = self.compute_reward(profit)
            self.position = 0
            action_str = 'force_close'
        elif action == 1 and self.position == 0:  # buy
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # sell
            self.position = -1
            self.entry_price = price
        elif action == 3 and self.position != 0:  # close
            profit = (price - self.entry_price) * self.position * 100
            reward = self.compute_reward(profit)
            self.position = 0
        # For hold or after entry
        if profit == 0 and self.position != 0:
            small_factor = 0.01
            if unreal_pnl > 0:
                reward = small_factor * unreal_pnl
            else:
                reward = -small_factor * abs(unreal_pnl)
        self.rewards.append(reward)
        self.results.append({
            'Time': ts,
            'Price': price,
            'Action': action_str,
            'Profit': profit
        })
        return action_str

    def finalize(self):
        if self.rewards:
            # Compute returns
            returns = []
            R = 0.0
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.cat(self.values).squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            states_t = torch.stack(self.states)
            actions_t = torch.tensor(self.actions, dtype=torch.long)
            old_logprobs_t = torch.tensor(self.logprobs, dtype=torch.float32)
            # PPO update
            for _ in range(self.K_epochs):
                logits = self.actor(states_t)
                probs = torch.softmax(logits, dim=1)
                new_logprobs = torch.log(probs.gather(1, actions_t.unsqueeze(1))).squeeze()
                ratios = torch.exp(new_logprobs - old_logprobs_t)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                new_values = self.critic(states_t).squeeze()
                critic_loss = F.mse_loss(new_values, returns)
                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # Save model
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, self.model_path)
        # Create DataFrame
        df_result = pd.DataFrame(self.results)
        # Reset
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.position = 0
        self.entry_price = 0.0
        self.results = []
        self.first_tick = True
        return df_result