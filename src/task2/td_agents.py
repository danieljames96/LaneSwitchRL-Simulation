import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import json
from datetime import datetime

class TemporalDifference:
    def __init__(self, env, oiv = 0, alpha=0.1, epsilon=0.1, lambd=0.9, gamma=0.9, epsilon_decay=0.999, epsilon_min=0.1):
        """
        Initializes the Temporal Difference learning agent.

        Parameters:
        - env: The environment instance in which the agent interacts.
        - oiv (float): Optimistic initial value for Q-values.
        - alpha (float): Learning rate, between 0 and 1.
        - epsilon (float): Exploration rate for epsilon-greedy policy.
        - lambd (float): Contribution of past rewards (eligibility trace decay factor).
        - gamma (float): Discount factor for future rewards.
        - epsilon_decay (float): Rate at which epsilon decays over episodes.
        - epsilon_min (float): Minimum value of epsilon after decay.
        
        Attributes:
        - Q (defaultdict): Q-value table with state-action pairs initialized to `oiv`.
        - E (defaultdict): Eligibility trace table with state-action pairs initialized to `oiv`.
        - best_Q (dict): Stores the best Q-value table found during training.
        - best_reward (float): Stores the highest average reward achieved.
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

        self.Q = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        self.E = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        
        # store 'best model'
        self.best_Q = None
        self.best_reward = float('-inf')
    
    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Parameters:
        - seed (int): Seed value for random number generation.
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def get_best_action(self, state: tuple):
        """
        Returns the best action for a given state based on current Q-values.

        Parameters:
        - state (tuple): Discretized representation of the environment state.

        Returns:
        - best_action (int): The action with the highest Q-value for the given state.
        """
        # if state not present, act random
        if state not in self.Q.keys():
            self.Env._log(f"State not found in Q: {state}")
            return random.randint(0,self.action_space-1)

        # get the dictionary of actions and their values for this state
        action_values = self.Q[state]
        # find the action with the maximum value
        best_action = np.argmax(action_values)
        return best_action

    # define epsilon greedypolicy
    def epsilon_greedy_policy(self, state):
        """
        Chooses an action using epsilon-greedy policy.

        Parameters:
        - state (tuple): Discretized state from the environment.

        Returns:
        - action (int): Action chosen based on epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_space-1)
        else:
            return self.get_best_action(state)
    
    def transform_state(self, state, initial_distance, num_discrete_levels=10):
        """
        Normalizes distance as a percentage of the initial distance and discretizes clearance rates.

        Parameters:
        - state (tuple): Raw state from the environment.
        - initial_distance (int): Initial distance for normalizing.
        - num_discrete_levels (int): Levels for discretizing clearance rates.

        Returns:
        - transformed_state (tuple): Discretized state including lane rates and risk factor.
        """
        # Normalize distance as a percentage of initial distance
        distance_percentage = int((state[0] / initial_distance) * 10)
        
        # Get current lane and clearance rates for adjacent lanes
        current_lane = int(state[1])
        risk_factor = round(state[2] * 5)
        clearance_rates = state[3:]
        
        # Weighted average for lanes to the left and right of the current lane
        def weighted_average(rates, start, end, direction):
            weights = [1 / (abs(i - current_lane) + 1) for i in range(start, end, direction)]
            total_weight = sum(weights)
            weighted_sum = sum(rate * weight for rate, weight in zip(rates[start:end:direction], weights))
            return weighted_sum / total_weight
        
        # Discretize clearance rates to specified levels (e.g., 1 to 10)
        min_rate, max_rate = self.Env.clearance_rate_min, self.Env.clearance_rate_max
        
        # Calculate weighted averages for left and right lanes
        left_lane_rate = (
            weighted_average(clearance_rates, 0, current_lane - 1, 1)
            if current_lane > 1 else 0
        )
        right_lane_rate = (
            weighted_average(clearance_rates, current_lane, len(clearance_rates), 1)
            if current_lane < len(clearance_rates) else 0
        )
        current_lane_rate = clearance_rates[current_lane - 1]
        
        # Discretize clearance rates within the specified range
        def discretize_rate(rate):
            if rate == 0:  # No lane in that direction
                return 0
            return int((rate - min_rate) / (max_rate - min_rate) * (num_discrete_levels - 1)) + 1

        left_lane_rate = discretize_rate(left_lane_rate)
        right_lane_rate = discretize_rate(right_lane_rate)
        current_lane_rate = discretize_rate(current_lane_rate)

        # Return transformed state with current lane, discrete clearance rates and discrete risk factor
        return (current_lane_rate, left_lane_rate, right_lane_rate, risk_factor) # distance_percentage, current_lane
    
    def train(self, num_episodes = 1000 , on_policy = True):
        """
        Trains the agent using Temporal Difference learning.

        Parameters:
        - num_episodes (int): Number of episodes to train.
        - on_policy (bool): Whether to use on-policy (SARSA) or off-policy (Q-learning).

        Returns:
        - total_reward_list (list): Rewards per episode.
        - total_steps_list (list): Steps per episode.
        """
        self.total_reward_list = []
        self.total_steps_list = []
        early_termination_count = 0

        for episode in tqdm(range(num_episodes)):
            episode_memory = []  # to be used when lambd = 1
            
            #reset episode, re-initialize E and total_reward
            state, _ = self.Env.reset(seed=self.seed_value+episode)
            state = self.transform_state(state, self.Env.initial_distance)
            
            self.set_seed(self.seed_value+episode)
            
            terminated, truncated = False, False
            self.E.clear()
            steps = 0
            episode_reward = 0

            #get first action
            action = self.epsilon_greedy_policy(state)

            while not terminated and not truncated:
                #perform action
                next_state, reward, terminated, truncated, info = self.Env.step(action)

                next_state = self.transform_state(next_state, self.Env.initial_distance)
                
                #accumulate steps and reward
                steps += 1
                episode_reward += reward

                #get next state and action
                next_action = self.epsilon_greedy_policy(next_state)
                
                #update tables(dictionaries)
                if self.lambd == 1:
                    episode_memory.append([state, action, reward])
                    state, action = next_state, next_action
                    state = tuple(state)
                    continue

                if on_policy:   # SARSA
                    if terminated or truncated:
                        delta = reward - self.Q[state][action]
                    else:
                        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
                else:           # Q-learning
                    if terminated or truncated:
                        delta = reward - self.Q[state][action]
                    else:
                        best_next_action = self.get_best_action(next_state)
                        delta = reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
                
                # if TD(0), no need to perform epsilon decay
                if self.lambd == 0:
                    # update Q values
                    self.Q[state][action] += self.alpha * delta
                
                # if TD(lambd), update E & Q
                else:
                    self.E[state][action] += 1
                    for state in self.Q.keys():
                        for action in range(self.action_space):
                            self.Q[state][action] += self.alpha * delta * self.E[state][action]
                            self.E[state][action] *= self.gamma * self.lambd
                    
                # Decay epsilon after each episode
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                # move to next state and action pair
                state, action = next_state, next_action
                
            early_termination_count += int(truncated)
        
            # Append total rewards and steps after the episode ends
            self.total_reward_list.append(episode_reward)
            
            if not truncated:
                self.total_steps_list.append(steps)
                
            if self.lambd == 1:
                G = 0
                for state, action, reward in reversed(episode_memory):
                    G = reward + self.gamma * G
                    self.Q[state][action] += self.alpha * (G - self.Q[state][action])
            
        print(f"Early Termination Count: {early_termination_count}")

        return self.total_reward_list, self.total_steps_list
    
    def plot_metrics(self, rewards, steps, window_size=50):
        """
        Plots the training metrics for rewards and steps with a rolling mean.

        Parameters:
        - rewards (list): List of rewards per episode.
        - steps (list): List of steps per episode.
        - window_size (int): Window size for rolling average.
        
        Returns:
        - fig (matplotlib.figure.Figure): Figure containing the plots.
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
        
    def format_state(self, state):
        """
        Formats the environment state by converting specific indices to integers and rounding others.

        Parameters:
        - state (list): Raw state list from the environment.

        Returns:
        - formatted_state (list): State with specified indices as integers and others rounded to one decimal.
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
        
    def evaluate(self, num_episodes=100, output_file=f"./logs/task2/test_log_tdlambda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"):
        """
        Evaluates the Temporal Difference agent in inference mode over a specified number of episodes.

        Parameters:
        - num_episodes (int): Number of episodes for evaluation.
        - output_file (str): Path for the JSON log file to record episode details.

        Returns:
        - rewards (list): Total reward per episode.
        - timesteps (list): Total steps per episode.
        - all_episode_reward_types (dict): Average reward type values across episodes.
        - output_file (str): Path of the generated log file.
        """
        
        early_termination_count = 0
        rewards = []
        timesteps = []
        all_episode_reward_types = []

        with open(output_file, 'w') as f:
            for episode in tqdm(range(num_episodes)):
                state, info = self.Env.reset(seed=self.seed_value+episode)
                
                self.set_seed(self.seed_value+episode)
                
                episode_details = {
                    "Episode": episode + 1,
                    "Initial State": self.format_state(state.tolist() if hasattr(state, 'tolist') else state),
                    "Timesteps": []
                }
                
                state = self.transform_state(state, self.Env.initial_distance)
                episode_steps, episode_rewards  = 0, 0
                terminated, truncated = False, False
                reward_type_values = info

                while not terminated and not truncated:
                    # Select action based on the trained policy without updating Q-values
                    action = self.get_best_action(state)  # Use the trained policy
                    action = int(action)
                    mapped_action = action - 1
                    
                    # Take action in the environment
                    next_state, reward, terminated, truncated, info = self.Env.step(action)
                    
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
                    
                    next_state = self.transform_state(next_state, self.Env.initial_distance)
                    
                    episode_rewards += reward
                    state = next_state
                    episode_steps += 1
                    
                    episode_details["Timesteps"].append(timestep_details)
                    
                episode_rewards = round(episode_rewards)
                rewards.append(episode_rewards)
                if not truncated:
                    timesteps.append(episode_steps)

                # Add total reward and timestep count to episode details
                episode_details["Total Reward"] = episode_rewards
                episode_details["Total Timesteps"] = episode_steps
                all_episode_reward_types.append(reward_type_values)

                # Write the full episode details as a JSON object to the file
                f.write(json.dumps(episode_details) + "\n")
                
                early_termination_count += int(truncated)
        
        df = pd.DataFrame(all_episode_reward_types)
        all_episode_reward_types = df.mean().to_dict()    
        
        print(f"Early terminations: {early_termination_count}")

        return rewards, timesteps, all_episode_reward_types, output_file
        
    def hyperparameter_tuning(self, hyperparameter_space, lambd=0, episodes=10000, on_policy=True, n_trials=50):
        """
        Tunes hyperparameters using Optuna to maximize average episode reward.

        Parameters:
            hyperparameter_space (dict): Ranges for hyperparameters to explore.
            episodes (int): Training episodes per trial.
            n_trials (int): Number of trials in Optuna study.

        Returns:
            tuple: Instance of the best agent and the optimal hyperparameters.
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
        best_agent = TemporalDifference(
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
        Analyzes the distribution of actions across states based on trained Q-values.

        Returns:
        - tuple: Counts and percentages of 'Left', 'Stay', and 'Right' actions.
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
        Plots the action distribution as a bar chart.
        
        Parameters:
        - left (int): Count of 'Left' actions.
        - stay (int): Count of 'Stay' actions.
        - right (int): Count of 'Right' actions.
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
    
    def save_q_table(self, file_path):
        """
        Save the Q-table to a JSON file.

        Parameters:
        - file_path (str): The path to save the Q-table JSON file.
        """
        # Convert defaultdict to a regular dictionary and stringify state keys
        q_table_dict = {str(state): list(q_values) for state, q_values in self.Q.items()}
        with open(file_path, 'w') as json_file:
            json.dump(q_table_dict, json_file, indent=4)
        print(f"Q-table saved to {file_path}")

    def load_q_table(self, file_path):
        """
        Load the Q-table from a JSON file.

        Parameters:
        - file_path (str): The path to the Q-table JSON file.
        """
        with open(file_path, 'r') as json_file:
            q_table_dict = json.load(json_file)
        # Convert JSON keys back to tuples and values to numpy arrays
        self.Q = defaultdict(lambda: np.zeros(self.action_space) + self.oiv,
                             {eval(state): np.array(q_values) for state, q_values in q_table_dict.items()})
        print(f"Q-table loaded from {file_path}")