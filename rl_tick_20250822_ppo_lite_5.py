import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class TradingSimulation:
    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64, lr=3e-4):
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

        self.reset()

    def reset(self):
        self.memory = []
        self.total_profit = 0
        self.position = 0
        self.entry_price = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def step(self, state, action, price):
        reward = 0.0

        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
        elif action == 1:  # Sell
            if self.position == 1:
                profit = price - self.entry_price
                self.total_profit += profit
                # 報酬設計：±500円未満は無視、それ以上のみ反映
                if abs(profit) < 500:
                    reward = 0.0
                else:
                    reward = profit / 1000.0
                self.position = 0
        elif action == 2:  # Hold
            reward = 0.0

        return reward

    def store(self, state, action, reward, log_prob):
        self.memory.append((state, action, reward, log_prob))

    def update(self):
        states, actions, rewards, log_probs = zip(*self.memory)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.stack(log_probs)

        values = self.value_net(states).squeeze()
        advantages = rewards - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, rewards)

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        self.reset()
