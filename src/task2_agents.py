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

    def _initialize_statistics(self):
        """Initialize statistics dictionary to store per-episode data."""
        self.episode_statistics = {
            "episode_rewards": [],
            "episode_steps": [],
            "lane_change_percentage": [],
            "early_termination_percentage": [],
            "clearance_rate_distribution": {
                "5-10%": 0,
                "10-20%": 0,
                ">20%": 0
            },
            "risk_factor_exceedance_percentage": []
        }
        
    def _update_clearance_statistics(self, clearance_rate):
        """Update clearance rate distribution based on the current rate."""
        if 5 <= clearance_rate < 10:
            self.episode_statistics["clearance_rate_distribution"]["5-10%"] += 1
        elif 10 <= clearance_rate < 20:
            self.episode_statistics["clearance_rate_distribution"]["10-20%"] += 1
        elif clearance_rate >= 20:
            self.episode_statistics["clearance_rate_distribution"][">20%"] += 1
    
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
    
    def transform_state(self, state, initial_distance, num_discrete_levels=5):
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
        risk_factor = int(state[18] * 5)
        clearance_rates = state[19:]
        
        # Discretize clearance rates to specified levels (e.g., 1 to 10)
        min_rate, max_rate = 5, max(20, max(clearance_rates))
        left_lane_rate = (clearance_rates[current_lane - 2] if current_lane > 1 else None)
        right_lane_rate = (clearance_rates[current_lane] if current_lane < len(clearance_rates) else None)
        current_lane_rate = clearance_rates[current_lane - 1]
        over_speed_limit = int(current_lane_rate > self.Env.speed_limit)
        
        # Discretize clearance rates within the specified range
        def discretize_rate(rate):
            if rate is None:
                return 0  # Assign 0 if no lane exists (e.g., left lane when in the leftmost position)
            return int((rate - min_rate) / (max_rate - min_rate) * (num_discrete_levels - 1)) + 1

        left_lane_rate = discretize_rate(left_lane_rate)
        right_lane_rate = discretize_rate(right_lane_rate)
        current_lane_rate = discretize_rate(current_lane_rate)

        # Return transformed state with current lane, discrete clearance rates and discrete risk factor
        return (current_lane_rate, left_lane_rate, right_lane_rate, risk_factor, over_speed_limit) # distance_percentage, current_lane
    
    def train(self, num_episodes = 1000 , on_policy = True, save_model = False):

        #initialize list to store episode history
        self.total_reward_list = []
        self.total_steps_list = []
        truncated_count = 0
        self._initialize_statistics()
        lane_change_count = 0
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
            high_risk_exceedances = 0

            #get first action
            action = self.epsilon_greedy_policy(state)
            if action != 1:
                lane_change_count += 1

            while not terminated and not truncated:
                #perform action
                next_state, reward, terminated, truncated, info = self.Env.step(action)

                next_state = self.transform_state(next_state, self.Env.initial_distance)

                # Check clearance rates
                for rate in self.Env.clearance_rates:
                    self._update_clearance_statistics(rate)
                    
                # Check risk factor exceedance
                if self.Env.risk_factor > self.Env.high_risk_threshold:
                    high_risk_exceedances += 1
                
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

                if truncated:
                    truncated_count += 1
                    
                # Decay epsilon after each episode
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                # move to next state and action pair
                state, action = next_state, next_action
            
            # Log episode-specific statistics
            self.episode_statistics["episode_rewards"].append(episode_reward)
            self.episode_statistics["episode_steps"].append(steps)
            self.episode_statistics["risk_factor_exceedance_percentage"].append(high_risk_exceedances / steps * 100)
            early_termination_count += int(truncated)
            
            # Lane change percentage calculation
            if episode % 10 == 0:  # Log every 10 episodes for granularity
                self.episode_statistics["lane_change_percentage"].append(lane_change_count / max(1, steps) * 100)
        
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
            
        # Calculate final statistics after all episodes
        self.episode_statistics["early_termination_percentage"].append(early_termination_count / num_episodes * 100)
            
        # Save statistics to JSON
        self._save_statistics_to_json()
            
        print(f"Truncated episodes: {truncated_count}")

        return self.total_reward_list, self.total_steps_list
    
    def _save_statistics_to_json(self):
        """Save the training statistics to a JSON file."""
        with open(self.log_path, 'w') as file:
            json.dump(self.episode_statistics, file, indent=4)
        print(f"Training statistics saved to {self.log_path}")
    
    def plot_training_metrics(self, window_size=50):
        """
        Plot training metrics (rewards and steps) with rolling mean (window_size)
        
        Args:
        rewards: list of episode rewards
        steps: list of episode steps
        window_size: size of rolling window for smoothing
        """
        rewards = self.total_reward_list
        steps = self.total_steps_list
        
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
        state_df = pd.DataFrame(q_df['State'].tolist(), columns=['Current Lane Rate', 'Left Lane Rate', 'Right Lane Rate', 'Risk Factor', 'Over Speed Limit'])

        # Concatenate the split columns with the original DataFrame
        q_df = pd.concat([state_df, q_df.drop(columns=['State'])], axis=1)
        
        return q_df
    
    def analyze_lane_speed_preference(self):

        q_df = self.q_table_to_dataframe()
        
        high_risk_mismatches = 0
        low_risk_mismatches = 0
        high_risk_total = 0
        low_risk_total = 0
        risk_threshold=self.Env.high_risk_threshold * 5
        results = []
        
        for _, row in q_df.iterrows():
            current_lane_rate = row['Current Lane Rate']
            left_lane_rate = row['Left Lane Rate']
            right_lane_rate = row['Right Lane Rate']
            risk_factor = row['Risk Factor']
            over_speed_limit = row['Over Speed Limit']
            
            # Determine which adjacent lane is faster
            if left_lane_rate > current_lane_rate:
                preferred_action = 'Left'
            elif right_lane_rate > current_lane_rate:
                preferred_action = 'Right'
            else:
                preferred_action = 'Stay'
            
            # Check if the Q-value for the preferred action is the highest
            highest_q_action = row[['Left', 'Stay', 'Right']].idxmax()
            
            # Determine if it's a high-risk or low-risk case
            is_high_risk = risk_factor >= risk_threshold
            
            # Count total high-risk and low-risk cases
            if is_high_risk:
                high_risk_total += 1
                if preferred_action != highest_q_action:
                    high_risk_mismatches += 1  # Count mismatch for high-risk cases
            else:
                low_risk_total += 1
                if preferred_action != highest_q_action:
                    low_risk_mismatches += 1  # Count mismatch for low-risk cases
            
            results.append({
                # 'State': row['Current Lane'],
                'Current Lane Rate': current_lane_rate,
                'Left Lane Rate': left_lane_rate,
                'Right Lane Rate': right_lane_rate,
                'Risk Factor': risk_factor,
                'Risk Level': 'High' if is_high_risk else 'Low',
                'Preferred Action (Fastest Lane)': preferred_action,
                'Highest Q-value Action': highest_q_action,
                'Highest Q-value': row[highest_q_action]
            })
            
        # Calculate percentages of mismatches for high-risk and low-risk cases
        high_risk_mismatch_percentage = (high_risk_mismatches / high_risk_total * 100) if high_risk_total > 0 else 0
        low_risk_mismatch_percentage = (low_risk_mismatches / low_risk_total * 100) if low_risk_total > 0 else 0
        
        results_df = pd.DataFrame(results)
        return results_df, high_risk_mismatch_percentage, low_risk_mismatch_percentage

    # Check if the agent tends to stay when risk factor is high
    def analyze_risk_factor_stay_preference(self):
        
        risk_threshold=self.Env.high_risk_threshold * 5
        
        q_df = self.q_table_to_dataframe()
        
        # Filter rows where risk factor is above the threshold
        high_risk_df = q_df[q_df['Risk Factor'] >= risk_threshold]
        
        # Count cases where 'Stay' has the highest Q-value in high-risk states
        high_risk_df['Highest Q-value Action'] = high_risk_df[['Left', 'Stay', 'Right']].idxmax(axis=1)
        stay_preference_count = (high_risk_df['Highest Q-value Action'] == 'Stay').sum()
        total_high_risk_cases = high_risk_df.shape[0]
        stay_preference_percentage = (stay_preference_count / total_high_risk_cases) * 100 if total_high_risk_cases > 0 else 0
        
        return {
            'Total High Risk Cases': total_high_risk_cases,
            'Stay Preference in High Risk': stay_preference_count,
            'Stay Preference Percentage': stay_preference_percentage
        }
    
    def evaluate(self, num_episodes=100):
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
        self._initialize_statistics()
        lane_change_count = 0
        early_termination_count = 0

        for episode in tqdm(range(num_episodes)):
            state, _ = self.Env.reset()
            state = self.transform_state(state, self.Env.initial_distance)
            steps, episode_reward  = 0, 0
            terminated, truncated = False, False
            high_risk_exceedances = 0

            while not terminated and not truncated:
                # Select action based on the trained policy without updating Q-values
                action = self.get_best_action(state)  # Use the trained policy
                if action != 1:
                    lane_change_count += 1
                
                # Take action in the environment
                next_state, reward, terminated, truncated, _ = self.Env.step(action)
                next_state = self.transform_state(next_state, self.Env.initial_distance)
                
                # Check clearance rates
                for rate in self.Env.clearance_rates:
                    self._update_clearance_statistics(rate)
                
                # Check risk factor exceedance
                if self.Env.risk_factor > self.Env.high_risk_threshold:
                    high_risk_exceedances += 1
                
                episode_reward += reward
                state = next_state
                steps += 1
            
             # Log episode-specific statistics
            self.episode_statistics["episode_rewards"].append(episode_reward)
            self.episode_statistics["episode_steps"].append(steps)
            self.episode_statistics["risk_factor_exceedance_percentage"].append(high_risk_exceedances / steps * 100)
            early_termination_count += int(truncated)

            # Lane change percentage calculation
            if episode % 10 == 0:
                self.episode_statistics["lane_change_percentage"].append(lane_change_count / max(1, steps) * 100)
                
        # Calculate final statistics after all episodes
        self.episode_statistics["early_termination_percentage"].append(early_termination_count / num_episodes * 100)
        
        # Save statistics to JSON
        self._save_statistics_to_json()

        return self.episode_statistics["episode_rewards"], self.episode_statistics["episode_steps"]

    def plot_evaluation_metrics(self, rewards, steps, window_size=10):
        """
        Plot evaluation metrics (rewards and steps per episode) with a rolling mean.

        Args:
        - rewards: List of cumulative rewards per episode.
        - steps: List of steps taken per episode.
        - checkpoint_rewards: List of average rewards at checkpoint intervals.
        - window_size: Window size for rolling mean smoothing.
        """
        # Convert rewards and steps to DataFrames for easier plotting
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

        # Plot rewards per episode
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_rewards, x='Episode', y='Reward', label='Total Reward', color='blue', alpha=0.6)
        sns.lineplot(data=df_rewards, x='Episode', y='Rolling Mean', label=f'Rolling Mean (window={window_size})', color='red')
        plt.title('Total Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

        # Plot steps per episode
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_steps, x='Episode', y='Steps', label='Steps per Episode', color='blue', alpha=0.6)
        sns.lineplot(data=df_steps, x='Episode', y='Rolling Mean', label=f'Rolling Mean (window={window_size})', color='red')
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.legend()
        plt.show()
        
    def hyperparameter_tuning(self, hyperparameter_space, episodes=10000, on_policy=True, n_trials=50):
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
            self.epsilon_decay = trial.suggest_float('epsilon_decay', *hyperparameter_space['epsilon_decay'])
            self.epsilon_min = trial.suggest_float('epsilon_min', *hyperparameter_space['epsilon_min'])
            self.lambd = trial.suggest_float('lambd', *hyperparameter_space['lambd'])

            # Train the agent with the current hyperparameters
            rewards = self.train(num_episodes=episodes, on_policy = on_policy)

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
            epsilon_decay=best_params['epsilon_decay'], 
            epsilon_min=best_params['epsilon_min'], 
            lambd=best_params['lambd']
        )

        print("\nBest Hyperparameters Found:")
        print(best_params)
        print(f"Best Average Reward: {best_average_reward:.2f}")

        return best_agent, best_params

class RuleBasedAgent:
    def __init__(self, initial_distance=4000, num_lanes=5, strategy='fastest_adjacent'):
        """
        Initialize the Agent with a specified strategy.
        
        Args:
        - strategy (str): The strategy to use ('fastest_adjacent', 'stay').
        """
        self.strategy = strategy
        self.initial_distance = initial_distance
        self.num_lanes = num_lanes
        
        self.current_lane = None
        self.clearance_rates = None

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