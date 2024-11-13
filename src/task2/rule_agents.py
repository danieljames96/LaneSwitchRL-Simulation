import numpy as np
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

class RuleBasedAgent:
    def __init__(self, env, initial_distance=4000, num_lanes=5, strategy='fastest_adjacent'):
        """
        Initializes the RuleBasedAgent with a specified strategy for navigating lanes.

        Parameters:
        - env: The environment instance where the agent will act.
        - initial_distance (int): The initial distance to the destination.
        - num_lanes (int): The number of lanes in the environment.
        - strategy (str): Strategy for lane selection ('fastest_adjacent' or 'stay').

        Attributes:
        - Env: Environment object where the agent interacts.
        - strategy: Strategy for decision-making.
        - initial_distance: The starting distance to the target.
        - num_lanes: Total number of lanes in the environment.
        - seed_value: Seed value for reproducibility.
        - current_lane: Current lane of the agent.
        - clearance_rates: Clearance rates per lane at the current time step.
        """
        self.Env = env
        self.strategy = strategy
        self.initial_distance = initial_distance
        self.num_lanes = num_lanes
        
        self.seed_value = self.Env.seed_value
        if self.seed_value is not None:
            self.set_seed(self.seed_value)
        
        self.current_lane = None
        self.clearance_rates = None
    
    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Parameters:
        - seed (int): Seed value for random number generation.
        """
        random.seed(seed)
        np.random.seed(seed)

    def choose_action(self, state):
        """
        Selects an action based on the current state and the agent's strategy.

        Parameters:
        - state (np.array): The current state as a flattened array.

        Returns:
        - action (int): Action to take (0: move left, 1: stay, 2: move right).
        """
        self.current_lane = int(state[1])  # Current lane at time step t
        self.clearance_rates = state[3:]  # Clearance rates at time step t

        if self.strategy == 'fastest_adjacent':
            return self._choose_action_fastest_adjacent()
        elif self.strategy == 'stay':
            return 1  # Always stay in the current lane
        else:
            raise ValueError("Invalid strategy. Choose 'fastest_adjacent' or 'stay'.")

    def _choose_action_fastest_adjacent(self):
        """
        Chooses the adjacent lane with the highest clearance rate, if better than the current lane.

        Returns:
        - action (int): Action to take (0: move left, 1: stay, 2: move right).
        """
        
        # Get the clearance rate for the current lane
        current_lane_rate = self.clearance_rates[self.current_lane - 1]
        
        # Check adjacent lanes' clearance rates
        left_lane_rate = self.clearance_rates[self.current_lane - 2] if self.current_lane > 1 else float('-inf')
        right_lane_rate = self.clearance_rates[self.current_lane] if self.current_lane < self.num_lanes else float('-inf')
        
        # Move to the adjacent lane with the highest clearance rate if it's better than the current lane
        if left_lane_rate > current_lane_rate and left_lane_rate >= right_lane_rate:
            return 0  # Move left
        elif right_lane_rate > current_lane_rate and right_lane_rate > left_lane_rate:
            return 2  # Move right
        else:
            return 1  # Stay in the current lane if neither adjacent lane is faster
    
    def format_state(self, state):
        """
        Formats the state array, converting specific indices to integers and rounding others.

        Parameters:
        - state (list or np.array): Raw state data from the environment.

        Returns:
        - formatted_state (list): Processed state with specified values as integers or rounded.
        """
        formatted_state = []
        for i, value in enumerate(state):
            if i == 1:  # Indices to be converted to integers
                formatted_state.append(int(value))
            elif i == 2:
                continue  # Skip these indices
            else:
                formatted_state.append(round(value, 1))  # Round to one decimal place
        return formatted_state
    
    def evaluate_agent(self, num_episodes=10, starting_lane = 1, output_file=f"./logs/task2/rule_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"):
        """
        Evaluates the agent over a specified number of episodes, recording rewards and steps.

        Parameters:
        - num_episodes (int): Number of episodes to run.
        - starting_lane (int): Lane from which the agent starts each episode.
        - output_file (str, optional): Path to the JSON file for saving episode logs. If None, a timestamped file is used.

        Returns:
        - all_episode_rewards (list): Cumulative reward for each episode.
        - all_timesteps (list): Total steps per episode.
        - all_episode_reward_types (dict): Average rewards for different types.
        - output_file (str): Path to the output file with evaluation details.
        """
        all_episode_rewards = []
        all_timesteps = []
        truncated_count = 0
        all_episode_reward_types = []

        with open(output_file, 'w') as f:
            for episode in tqdm(range(num_episodes)):
                episode_rewards = []
                
                options = {
                    'starting_lane': starting_lane
                }
                state, info = self.Env.reset(seed=self.seed_value+episode, options=options)
                terminated = False
                truncated = False
                cumulative_reward = 0
                episode_steps = 0
                reward_type_values = info
                
                self.set_seed(self.seed_value+episode)
                
                episode_details = {
                        "Episode": episode + 1,
                        "Initial State": self.format_state(state.tolist() if hasattr(state, 'tolist') else state),
                        "Timesteps": []
                    }

                while not terminated and not truncated:
                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.Env.step(action)
                    cumulative_reward += reward
                    state = next_state
                    episode_steps += 1
                    
                    action = int(action)
                    mapped_action = action - 1
                    
                    for key in info.keys():
                        if key not in reward_type_values:
                            reward_type_values[key] = 0
                        reward_type_values[key] += round(info[key],1)
                    
                    # Log details of each timestep including reward
                    timestep_details = {
                        "Timestep": episode_steps,
                        "State": self.format_state(next_state.tolist() if hasattr(next_state, 'tolist') else next_state),
                        "Action": mapped_action,
                        "Reward": reward
                    }
                    
                    episode_details["Timesteps"].append(timestep_details)
                    
                    # Store rewards at each timestep for this episode
                    episode_rewards.append(cumulative_reward)

                    if truncated:
                        truncated_count += 1
                        break
                    
                # Add total reward and timestep count to episode details
                episode_details["Total Reward"] = episode_rewards
                episode_details["Total Timesteps"] = episode_steps
                all_episode_reward_types.append(reward_type_values)

                # Write the full episode details as a JSON object to the file
                f.write(json.dumps(episode_details) + "\n")

                # Append results for each episode
                all_episode_rewards.append(cumulative_reward)
                if not truncated:
                    all_timesteps.append(episode_steps)
        
        df = pd.DataFrame(all_episode_reward_types)
        all_episode_reward_types = df.mean().to_dict()


        print(f"Truncated episodes: {truncated_count}")
        
        return all_episode_rewards, all_timesteps, all_episode_reward_types, output_file
    
    def plot_metrics(self, rewards, steps, window_size=50):
        """
        Plots episode rewards and steps with a rolling mean for smoothing.

        Parameters:
        - rewards (list): List of cumulative rewards per episode.
        - steps (list): List of steps taken per episode.
        - window_size (int): Window size for rolling mean.
        
        Returns:
        - fig (matplotlib.figure.Figure): The figure containing the plots.
        """
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Create dataframes for easier plotting
        df_rewards = pd.DataFrame({
            'Episode': range(len(rewards)),
            'Reward': rewards,
            'Rolling Mean': pd.Series(rewards).rolling(window=window_size).mean()
        })
        
        df_steps = pd.DataFrame({
            'Episode': range(len(steps)),
            'Steps': steps,
            'Rolling Mean': pd.Series(steps).rolling(window=window_size).mean()
        })
        
        # Plot rewards
        sns.scatterplot(data=df_rewards, x='Episode', y='Reward', 
                    alpha=0.3, color='blue', ax=ax1, label='Reward')
        sns.lineplot(data=df_rewards, x='Episode', y='Rolling Mean',
                    color='red', ax=ax1, linewidth=2, label=f'Rolling Mean (window={window_size})')
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot steps
        sns.scatterplot(data=df_steps, x='Episode', y='Steps',
                    alpha=0.3, color='blue', ax=ax2, label='Steps')
        sns.lineplot(data=df_steps, x='Episode', y='Rolling Mean',
                    color='red', ax=ax2, linewidth=2, label=f'Rolling Mean (window={window_size})')
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Number of Steps')
        
        # Adjust layout and display
        plt.tight_layout()
        return fig