import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
from collections import deque

# Model save/load configuration
MODEL_PATH = "trading_agent_ppo_lite.pt"


class PPOLiteAgent(nn.Module):
    """
    PPO-lite (lightweight Actor-Critic) agent.

    This agent uses a simplified PPO algorithm with Generalized Advantage Estimation (GAE).
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        super(PPOLiteAgent, self).__init__()

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        # Networks for Actor and Critic
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # On-policy data buffer (for a single episode)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # Learning step counter
        self.learning_step = 0

    def get_action(self, state):
        """
        Choose an action and its log probability based on the state.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        """
        Get the state value from the critic network.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.critic(state_tensor).item()

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """
        Store a transition in the on-policy data buffer.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def update(self):
        """
        Perform a single learning step using the collected data.
        """
        if not self.states:
            return

        # Convert collected data to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(self.dones)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.FloatTensor(self.values)

        # Calculate GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)
            td_errors = rewards + self.gamma * next_values * (1 - dones) - old_values

            advantages = torch.zeros_like(td_errors)
            last_advantage = 0
            for t in reversed(range(len(td_errors))):
                advantages[t] = td_errors[t] + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
                last_advantage = advantages[t]

        # PPO Learning Loop (can be run for multiple epochs)
        for _ in range(1):
            # Actor loss (PPO loss)
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (MSE loss)
            returns = advantages + old_values
            critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.learning_step += 1

        # Clear the buffer after one learning step
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path):
        """
        Save the model parameters.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model parameters.
        """
        self.load_state_dict(torch.load(path))


class TradingSimulation:
    """
    Main simulation class for executing trades and managing the reinforcement learning loop.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.state_dim = 3  # (price_norm, profit_norm, log_delta_volume)
        self.action_dim = 3  # 0: Buy, 1: Sell, 2: Hold
        self.agent = PPOLiteAgent(self.state_dim, self.action_dim)

        # Check if a model exists and load it
        self.is_model_loaded = False
        if os.path.exists(self.model_path):
            try:
                self.agent.load(self.model_path)
                self.is_model_loaded = True
                print("既存の学習モデルを読み込みました。")
            except Exception as e:
                print(f"既存モデルの読み込みに失敗しました: {e}")
                print("新しいモデルを生成します。")
                self.agent = PPOLiteAgent(self.state_dim, self.action_dim)
        else:
            print("既存の学習モデルが見つかりません。新しいモデルを生成します。")

        self.last_volume = 0
        self.current_position = 0  # 1: Long, -1: Short, 0: None
        self.entry_price = 0
        self.entry_ts = 0
        self.results_df = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])

    def _normalize_state(self, price, current_profit, delta_volume):
        """
        Normalize the state values.
        """
        price_norm = price / 10000.0  # Normalize price to a reasonable range
        profit_norm = current_profit / 1000.0  # Normalize profit
        log_delta_volume = np.log1p(delta_volume)

        return np.array([price_norm, profit_norm, log_delta_volume])

    def _calculate_reward(self, profit):
        """
        Calculate the reward based on the profit.
        """
        if profit >= 1000:
            return 2.0  # Big bonus for large profits
        elif profit >= 500:
            return 1.5  # Medium bonus
        elif profit >= 200:
            return 1.0  # Small bonus
        elif profit > 0:
            return 0.1  # Minor bonus for positive profit
        elif profit < 0:
            return -0.1  # Minor penalty for loss
        else:
            return 0

    def add(self, ts, price, volume, force_close=False):
        """
        Add new tick data and decide a trading action.
        """
        # Calculate delta volume and update last volume
        delta_volume = volume - self.last_volume
        self.last_volume = volume

        # Calculate current profit if there is a position
        current_profit = 0
        if self.current_position != 0:
            current_profit = self.current_position * (price - self.entry_price) * 100

        # Get the state and action
        state = self._normalize_state(price, current_profit, delta_volume)
        action, log_prob = self.agent.get_action(state)

        # --- Trading Logic ---
        executed_action = "Hold"
        profit = np.nan

        # If there is a position, check for closing conditions
        if self.current_position != 0:
            # Condition 1: Profit target reached
            if current_profit >= 200:
                executed_action = "Close"
                profit = current_profit
                reward = self._calculate_reward(profit)

                # Store the transition
                next_state = self._normalize_state(price, 0, delta_volume)
                self.agent.store_transition(state, action, reward, next_state, True, log_prob,
                                            self.agent.get_value(state))

                # Reset position
                self.current_position = 0
                self.entry_price = 0
                self.entry_ts = 0

            # Condition 2: Forced close at the end of the day
            elif force_close:
                executed_action = "Forced Close"
                profit = current_profit
                reward = self._calculate_reward(profit)

                # Store the transition
                next_state = self._normalize_state(price, 0, delta_volume)
                self.agent.store_transition(state, action, reward, next_state, True, log_prob,
                                            self.agent.get_value(state))

                # Reset position
                self.current_position = 0
                self.entry_price = 0
                self.entry_ts = 0

            else:
                # No action, just store the state
                executed_action = "Hold"
                reward = 0
                next_state = self._normalize_state(price, current_profit, delta_volume)
                self.agent.store_transition(state, action, reward, next_state, False, log_prob,
                                            self.agent.get_value(state))

        # If there is no position, decide whether to buy or sell
        else:
            # Buy Action
            if action == 0:
                executed_action = "Buy"
                self.current_position = 1
                self.entry_price = price
                self.entry_ts = ts

            # Sell Action
            elif action == 1:
                executed_action = "Sell"
                self.current_position = -1
                self.entry_price = price
                self.entry_ts = ts

            # Hold Action
            else:
                executed_action = "Hold"

            # Store the transition
            reward = 0
            next_state = self._normalize_state(price, 0, delta_volume)
            self.agent.store_transition(state, action, reward, next_state, False, log_prob, self.agent.get_value(state))

        # Log the result
        self.results_df.loc[len(self.results_df)] = [ts, price, executed_action, profit]

        return executed_action

    def finalize(self):
        """
        Finalize the simulation for the day, perform learning, save the model, and reset the results.
        """
        # Learn from the collected experience buffer
        self.agent.update()

        # Save the model
        self.agent.save(self.model_path)
        print("学習モデルを保存しました。")

        # Get the results for the day
        results = self.results_df.copy()

        # Reset the state for the next epoch
        self.current_position = 0
        self.entry_price = 0
        self.entry_ts = 0
        self.results_df = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])
        self.last_volume = 0

        return results


# --- Main simulation loop provided by the user ---
if __name__ == "__main__":
    # 学習対象のティックデータファイル: Time, Price, Volume の 3 列
    excel_file = "data/tick_20250819_7011.xlsx"

    # for learning curve
    df_lc = pd.DataFrame({"Epoch": list(), "Profit": list()})

    # 学習回数
    epochs = 100

    # ティックデータを読み込む
    # Sample data for demonstration purposes
    # Replace this with your actual data loading
    if not os.path.exists(excel_file):
        print(f"'{excel_file}' が見つかりません。デモ用データを作成します。")
        ts_range = range(100)
        prices = [1000 + np.sin(t / 10) * 50 + np.random.randn() * 5 for t in ts_range]
        volumes = [1000 + t * 10 + np.random.randn() * 10 for t in ts_range]
        df = pd.DataFrame({"Time": ts_range, "Price": prices, "Volume": volumes})
    else:
        df = pd.read_excel(excel_file)  # "Time", "Price", "Volume" 列がある想定

    print(df.head())

    # シミュレータ・インスタンス
    sim = TradingSimulation()

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    print("ティックファイル:", excel_file)

    # 繰り返し学習
    for epoch in range(epochs):
        # 1行ずつシミュレーションに流す
        for i, row in df.iterrows():
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]

            # 最後の行だけ強制返済フラグを立てる
            force_close = (i == len(df) - 1)
            action = sim.add(ts, price, volume, force_close=force_close)

        # 結果（総収益）を保存し、学習を実行
        df_result = sim.finalize()
        profit = df_result["Profit"].sum()
        print(f"Epoch {epoch}: 総収益 {profit:.2f}")
        df_result.to_csv(f"results/trade_results_{epoch:02}.csv")

        # for plot of learning curve
        df_lc.loc[epoch] = [epoch, profit]

    df_lc.to_csv(f"results/learning_curve.csv")
    print("学習曲線のデータを保存しました。")
    print("シミュレーションが完了しました。")