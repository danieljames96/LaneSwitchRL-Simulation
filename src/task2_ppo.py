import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class PolicyAgents:
    def __init__(self, env, oiv = 0, alpha=0.1, epsilon=0.1, lambd=0.9, gamma=0.9, epsilon_decay=0.999, epsilon_min=0.1):
        """
        Args:
        - lanes (int): Number of lanes (default is 5).
        - alpha (float): learning rate, between 0 and 1
        - epsilon (float): exploration rate between, 0 and 1
        - lambd (float): contribution of past rewards, between 0 and 1
        - gamma (float): discount factor, between 0 and 1
        - oiv (int/float): optimistic initial value
        """
        super().__init__()
        self.Env = env
        self.seed_value = self.Env.seed_value
        if self.seed_value is not None:
            self.set_seed(self.seed_value)
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambd = lambd
        self.gamma = gamma
        self.oiv = oiv
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Action space: 3 actions (0: move left, 1: stay, 2: move right)
        self.action_space = 3

        # use nested dictionaries for V, Q, E.
        # {state: [s_a1, s_a2, s_a3]}
        self.Q = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        self.E = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        
        # store 'best model'
        self.best_Q = None
        self.best_reward = float('-inf')
    
    def format_state(self, state):
        """
        Formats the state array so that specific indices are integers, 
        and others are rounded to one decimal place.
        """
        formatted_state = []
        for i, value in enumerate(state):
            if i in [1, 9, 17]:  # Indices to be converted to integers
                formatted_state.append(int(value))
            elif i in [2, 10, 18]:
                continue  # Skip these indices
            else:
                formatted_state.append(round(value, 1))  # Round to one decimal place
        return formatted_state
    
    def evaluate(self, model, env, num_episodes, output_file=None):
        """
        Evaluates the model over a specified number of episodes, records rewards for each episode,
        and plots the rewards.

        Args:
        - model: Trained model to be tested.
        - env (gym.Env): The environment to evaluate the model on.
        - num_episodes (int): Total number of episodes to run the evaluation.
        - output_file (str, optional): The path to the JSON file where episode details will be saved. If not provided,
        a timestamped default filename will be used.

        Returns:
        - episode_rewards (list): List of total rewards for each episode.
        """
        
        # Use a timestamped default filename if output_file is not provided
        if output_file is None:
            output_file = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Initialize lists for rewards and timesteps
        rewards = []
        timesteps = []

        # Create a JSON file for writing
        with open(output_file, 'w') as f:
            # Run the model for the specified number of episodes
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_timestep = 0
                episode_details = {
                    "Episode": episode + 1,
                    "Initial State": self.format_state(obs.tolist() if hasattr(obs, 'tolist') else obs),
                    "Timesteps": []
                }

                while True:  # Run until the episode ends
                    action, _states = model.predict(obs, deterministic=True)
                    action = int(action)
                    mapped_action = action - 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_timestep += 1

                    # Log details of each timestep including reward
                    timestep_details = {
                        "Timestep": episode_timestep,
                        "State": self.format_state(obs.tolist() if hasattr(obs, 'tolist') else obs),
                        "Action": mapped_action,
                        "Reward": reward
                    }
                    episode_details["Timesteps"].append(timestep_details)

                    # Check if the episode is done
                    if terminated or truncated:
                        episode_reward = round(episode_reward)
                        rewards.append(episode_reward)
                        timesteps.append(episode_timestep)

                        # Add total reward and timestep count to episode details
                        episode_details["Total Reward"] = episode_reward
                        episode_details["Total Timesteps"] = episode_timestep

                        # Write the full episode details as a JSON object to the file
                        f.write(json.dumps(episode_details) + "\n")
                        break  

        # Calculate average reward
        reward_ave = round(sum(rewards) / num_episodes)
        print(f"Average reward of {num_episodes} episodes is {reward_ave}.")

        # Calculate average timestep
        timestep_ave = round(sum(timesteps) / num_episodes)
        print(f"Average timesteps of {num_episodes} episodes is {timestep_ave}.")
        
        return rewards, timesteps
    
    def plot_test_results(rewards, timesteps, interval=10):
        """
        Plots individual episode rewards and timesteps as dots with transparency, 
        along with average rewards and timesteps per specified interval (e.g., 10 episodes) as lines.

        Args:
        - rewards (list): List of total rewards for each episode.
        - timesteps (list): List of total timesteps for each episode.
        - interval (int): Number of episodes over which to average values for the line plot.
        """
        
        # Calculate average rewards and timesteps per interval episodes
        avg_rewards = [
            sum(rewards[i:i+interval]) / interval for i in range(0, len(rewards), interval)
        ]
        avg_timesteps = [
            sum(timesteps[i:i+interval]) / interval for i in range(0, len(timesteps), interval)
        ]
        
        # x-axis for average values positioned at the midpoint of each interval
        avg_episodes = [i + interval // 2 for i in range(0, len(rewards), interval)]

        # Set up a figure with two subplots: one for rewards and one for timesteps (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot rewards
        ax1.scatter(range(1, len(rewards) + 1), rewards, color='blue', alpha=0.3, label='Individual Episode Rewards')
        ax1.plot(avg_episodes, avg_rewards, color='red', marker='o', linestyle='-', linewidth=2, label=f'Average Reward per {interval} Episodes')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward')
        ax1.set_title(f'Rewards')
        ax1.legend()

        # Plot timesteps
        ax2.scatter(range(1, len(timesteps) + 1), timesteps, color='green', alpha=0.3, label='Individual Episode Timesteps')
        ax2.plot(avg_episodes, avg_timesteps, color='purple', marker='o', linestyle='-', linewidth=2, label=f'Average Timesteps per {interval} Episodes')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Timesteps')
        ax2.set_title(f'Timesteps')
        ax2.legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(self, hyperparameter_space, lambd=0, episodes=10000, on_policy=True, n_trials=50):
        """
        Perform hyperparameter tuning using Optuna.

        Parameters:
            hyperparameter_space (dict): Dictionary defining the ranges of hyperparameters to tune.
            episodes (int): Number of training episodes for each trial.
            n_trials (int): Number of trials to run for the Optuna study.

        Returns:
            tuple: The best agent instance and the best hyperparameters found.
        """

        def objective(trial):
            # Define the hyperparameter space using Optuna's suggest functions
            self.alpha = trial.suggest_float('alpha', *hyperparameter_space['alpha'], log=True)
            self.gamma = trial.suggest_float('gamma', *hyperparameter_space['gamma'])
            self.epsilon = 1.0
            # self.epsilon_decay = trial.suggest_float('epsilon_decay', *hyperparameter_space['epsilon_decay'])
            self.epsilon_min = trial.suggest_float('epsilon_min', *hyperparameter_space['epsilon_min'])
            # self.lambd = trial.suggest_float('lambd', *hyperparameter_space['lambd'])

            # Train the agent with the current hyperparameters
            rewards, _ = self.train(num_episodes=episodes, on_policy = on_policy)

            # Calculate the average reward over the last 1000 episodes
            average_reward = np.mean(rewards[-500:])
            print(f"Trial {trial.number}: Average Reward = {average_reward:.2f}")

            return average_reward

        # Create a study and optimize the objective function
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Retrieve the best hyperparameters and best average reward
        best_params = study.best_params
        best_average_reward = study.best_value  # negate since we minimized -reward
        # Instantiate the best agent with the optimized hyperparameters
        best_agent = PolicyAgents(
            self.Env,
            oiv = 0.1, 
            alpha=best_params['alpha'], 
            gamma=best_params['gamma'], 
            epsilon=1.0, 
            # epsilon_decay=best_params['epsilon_decay'], 
            epsilon_decay = 0.9999,
            epsilon_min=best_params['epsilon_min'], 
            lambd=lambd #best_params['lambd']
        )

        print("\nBest Hyperparameters Found:")
        print(best_params)
        print(f"Best Average Reward: {best_average_reward:.2f}")

        return best_agent, best_params
    
    def analyze_model_actions(self):
        """
        Analyze action distribution from either a saved model file or a TD agent
        
        Args:
        - model_path (str): Path to saved model JSON file (optional)
        - td_agent (TemporalDifference): Trained TD agent (optional)
        """
        # Get Q-values either from file or agent
        q_values = dict(self.Q)

        # Analyze action distribution
        left = 0
        stay = 0
        right = 0
        
        for state, values in q_values.items():
            action = np.argmax(values)
            if action == 0:
                left += 1
            elif action == 1:
                stay += 1
            else:
                right += 1
        
        # Calculate percentages
        total = left + stay + right
        left_pct = left/total * 100
        stay_pct = stay/total * 100
        right_pct = right/total * 100
        
        print(f"Action Distribution:")
        print(f"Left:  {left:4d} ({left_pct:.1f}%)")
        print(f"Stay:  {stay:4d} ({stay_pct:.1f}%)")
        print(f"Right: {right:4d} ({right_pct:.1f}%)")
        
        # Plot distribution
        self.plot_action_distribution(left, stay, right)
        
        return left, stay, right, left_pct, stay_pct, right_pct

    def plot_action_distribution(self, left, stay, right):
        """
        Plot the distribution of actions as a bar chart
        """
        actions = ['Left', 'Stay', 'Right']
        counts = [left, stay, right]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(actions, counts)
        plt.title('Distribution of Preferred Actions Across States', pad=20)
        plt.xlabel('Action')
        plt.ylabel('Number of States')
        
        # Add percentage labels on top of each bar
        total = sum(counts)
        for bar in bars:
            height = bar.get_height()
            percentage = height/total * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({percentage:.1f}%)',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()