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

# TD class with states represented with integers
class TemporalDifference:
    def __init__(self, env, oiv = 0, alpha=0.1, epsilon=0.1, lambd=0.9, gamma=0.9, epsilon_decay=0.999, epsilon_min=0.1, log_path='./logs/task2/training_logs.json'):
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
        self.log_path = log_path

        # Action space: 3 actions (0: move left, 1: stay, 2: move right)
        self.action_space = 3

        # use nested dictionaries for V, Q, E.
        # {state: [s_a1, s_a2, s_a3]}
        self.Q = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        self.E = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        
        # store 'best model'
        self.best_Q = None
        self.best_reward = float('-inf')
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def get_best_action(self, state: tuple):
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
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_space-1)
        else:
            return self.get_best_action(state)
    
    def transform_state(self, state, initial_distance, num_discrete_levels=10):
        """
        Normalize the distance to a percentage of the initial distance and discretize clearance rates.
        
        Args:
        - state: The original state tuple from the environment.
        - initial_distance: Initial distance to the destination for normalizing distance.
        - num_discrete_levels: Number of discrete levels for clearance rates.

        Returns:
        - transformed_state: Tuple containing the normalized distance percentage and discrete clearance rates.
        """
        # Normalize distance as a percentage of initial distance
        distance_percentage = int((state[16] / initial_distance) * 10)
        
        # Get current lane and clearance rates for adjacent lanes
        current_lane = int(state[17])
        risk_factor = round(state[18] * 5)
        clearance_rates = state[19:]
        
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
    
    def train(self, num_episodes = 1000 , on_policy = True, save_model = False):

        #initialize list to store episode history
        self.total_reward_list = []
        self.total_steps_list = []
        early_termination_count = 0

        for episode in tqdm(range(num_episodes)):
            episode_memory = []  # to be used when lambd = 1
            
            #reset episode, re-initialize E and total_reward
            state, _ = self.Env.reset()
            state = self.transform_state(state, self.Env.initial_distance)
            
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
            
            if save_model == True:
                # Calculate average reward over last 100 episodes (or all if less than 100)
                window_size = min(100, len(self.total_reward_list))
                avg_reward = sum(self.total_reward_list[-window_size:]) / window_size
                
                # Save if best performance
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.best_Q = dict(self.Q)  # Create a deep copy of current Q-values
                
            if self.lambd == 1:
                G = 0
                for state, action, reward in reversed(episode_memory):
                    G = reward + self.gamma * G
                    self.Q[state][action] += self.alpha * (G - self.Q[state][action])
            
        print(f"Early Termination Count: {early_termination_count}")

        return self.total_reward_list, self.total_steps_list
    
    def plot_metrics(self, rewards, steps, window_size=50):
        """
        Plot training and eval metrics (rewards and steps) with rolling mean (window_size)
        
        Args:
        rewards: list of episode rewards
        steps: list of episode steps
        window_size: size of rolling window for smoothing
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
    
    def q_table_to_dataframe(self):
        
        data = {
            "State": [],
            "Left": [],
            "Stay": [],
            "Right": []
        }
    
        for state, q_values in self.Q.items():
            data["State"].append(state)
            for i in range(len(q_values)):
                data['Left' if i==0 else 'Stay' if i==1 else 'Right'].append(q_values[i])

        # Convert the dictionary to a DataFrame
        q_df = pd.DataFrame(data)
        
        # Split the 'State' column into separate columns
        state_df = pd.DataFrame(q_df['State'].tolist(), columns=['Current Lane Rate', 'Left Lane Rate', 'Right Lane Rate', 'Risk Factor'])

        # Concatenate the split columns with the original DataFrame
        q_df = pd.concat([state_df, q_df.drop(columns=['State'])], axis=1)
        
        return q_df
        
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
        
    def evaluate(self, num_episodes=100, output_file=f"./logs/task2/test_log_tdlambda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"):

        """
        Evaluate the Temporal Difference agent in inference mode.

        Args:
        - agent: Instance of the trained TemporalDifference agent.
        - env: The environment in which to evaluate the agent.
        - num_episodes (int): Number of episodes to evaluate.
        - checkpoint_interval (int): Interval to log cumulative rewards at checkpoints.

        Returns:
        - avg_rewards: Average rewards per episode across all episodes.
        - avg_steps: Average steps per episode across all episodes.
        - checkpoint_rewards: Rewards recorded at specified checkpoint intervals.
        """
        
        early_termination_count = 0
        rewards = []
        timesteps = []

        with open(output_file, 'w') as f:
            for episode in tqdm(range(num_episodes)):
                state, _ = self.Env.reset()
                
                episode_details = {
                    "Episode": episode + 1,
                    "Initial State": self.format_state(state.tolist() if hasattr(state, 'tolist') else state),
                    "Timesteps": []
                }
                
                state = self.transform_state(state, self.Env.initial_distance)
                episode_steps, episode_rewards  = 0, 0
                terminated, truncated = False, False

                while not terminated and not truncated:
                    # Select action based on the trained policy without updating Q-values
                    action = self.get_best_action(state)  # Use the trained policy
                    action = int(action)
                    mapped_action = action - 1
                    
                    # Take action in the environment
                    next_state, reward, terminated, truncated, _ = self.Env.step(action)
                    
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

                # Write the full episode details as a JSON object to the file
                f.write(json.dumps(episode_details) + "\n")
                
                early_termination_count += int(truncated)
            
        print(f"Early terminations: {early_termination_count}")

        return rewards, timesteps, output_file
        
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

class RuleBasedAgent:
    def __init__(self, env, initial_distance=4000, num_lanes=5, strategy='fastest_adjacent'):
        """
        Initialize the Agent with a specified strategy.
        
        Args:
        - strategy (str): The strategy to use ('fastest_adjacent', 'stay').
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
        random.seed(seed)
        np.random.seed(seed)

    def choose_action(self, state):
        """
        Choose an action based on the current state and the selected strategy.
        
        Args:
        - state (np.array): Flattened state containing past three time steps.
        
        Returns:
        - action (int): 0 (move left), 1 (stay), or 2 (move right)
        """
        
        self.current_lane = int(state[17])  # Current lane at time step t
        self.clearance_rates = state[19:]  # Clearance rates at time step t

        if self.strategy == 'fastest_adjacent':
            return self._choose_action_fastest_adjacent()
        elif self.strategy == 'stay':
            return 1  # Always stay in the current lane
        else:
            raise ValueError("Invalid strategy. Choose 'fastest_adjacent' or 'stay'.")

    def _choose_action_fastest_adjacent(self):
        """Selects the adjacent lane with the highest clearance rate if it's better than the current lane."""
        
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
    
    def evaluate_agent(self, num_episodes=10, starting_lane = 1, output_file=f"./logs/task2/rule_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"):
        all_episode_rewards = []
        all_timesteps = []
        truncated_count = 0

        with open(output_file, 'w') as f:
            for episode in tqdm(range(num_episodes)):
                episode_rewards = []
                
                options = {
                    'starting_lane': starting_lane
                }
                state, _ = self.Env.reset(options=options)
                terminated = False
                truncated = False
                cumulative_reward = 0
                episode_steps = 0
                
                episode_details = {
                        "Episode": episode + 1,
                        "Initial State": self.format_state(state.tolist() if hasattr(state, 'tolist') else state),
                        "Timesteps": []
                    }

                while not terminated and not truncated:
                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, _ = self.Env.step(action)
                    cumulative_reward += reward
                    state = next_state
                    episode_steps += 1
                    
                    action = int(action)
                    mapped_action = action - 1
                    
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

                # Write the full episode details as a JSON object to the file
                f.write(json.dumps(episode_details) + "\n")

                # Append results for each episode
                all_episode_rewards.append(cumulative_reward)
                if not truncated:
                    all_timesteps.append(episode_steps)

        print(f"Truncated episodes: {truncated_count}")
        
        return all_episode_rewards, all_timesteps, output_file
    
    def plot_metrics(self, rewards, steps, window_size=50):
        """
        Plot training and eval metrics (rewards and steps) with rolling mean (window_size)
        
        Args:
        rewards: list of episode rewards
        steps: list of episode steps
        window_size: size of rolling window for smoothing
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