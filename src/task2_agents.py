import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# TD class with states represented with integers
class TemporalDifference:
    def __init__(self, env, oiv = 0, alpha=0.1, epsilon=0.1, lambd=0.9, gamma=0.9):
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

        # Action space: 4 actions (0: move left, 1: stay, 2: move right, 3: rest)
        self.action_space = 3 # 4

        # use nested dictionaries for V, Q, E.
        # {state: [s_a1, s_a2, s_a3]}
        self.Q = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        self.E = defaultdict(lambda: np.zeros(self.action_space) + self.oiv)
        
        # store 'best model'
        self.best_Q = None
        self.best_reward = float('-inf')

    def get_best_action(self, state: tuple):
        # if state not present, act random
        if state not in self.Q.keys():
            return random.randint(0,self.action_space-1)

        # get the dictionary of actions and their values for this state
        action_values = self.Q[state]
        # find the action with the maximum value
        best_action = np.argmax(action_values)
        return best_action

    # define epsilon greedypolicy
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0,2)
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
        # fatigue_counter = 1 if int(state[18]) > 3 else 0
        clearance_rates = state[19:]
        
        # Discretize clearance rates to specified levels (e.g., 1 to 10)
        min_rate, max_rate = 5, max(20, max(clearance_rates))
        left_lane_rate = (clearance_rates[current_lane - 2] if current_lane > 1 else None)
        right_lane_rate = (clearance_rates[current_lane] if current_lane < len(clearance_rates) else None)
        current_lane_rate = clearance_rates[current_lane - 1]
        
        # Discretize clearance rates within the specified range
        def discretize_rate(rate):
            if rate is None:
                return 0  # Assign 0 if no lane exists (e.g., left lane when in the leftmost position)
            return int((rate - min_rate) / (max_rate - min_rate) * (num_discrete_levels - 1)) + 1

        left_lane_rate = discretize_rate(left_lane_rate)
        right_lane_rate = discretize_rate(right_lane_rate)
        current_lane_rate = discretize_rate(current_lane_rate)

        # Return transformed state with normalized distance and discrete clearance rates
        return (distance_percentage, current_lane, current_lane_rate, left_lane_rate, right_lane_rate) # fatigue_counter
    
    def train(self, num_episodes = 1000 , on_policy = True, save_model = False, checkpoint_interval = 1000, log = False):

        #initialize list to store episode history
        self.total_reward_list = []
        self.total_steps_list = []
        truncated_count = 0

        for episode in tqdm(range(num_episodes)):
            episode_memory = []  # to be used when lambd = 1
            
            #reset episode, re-initialize E and total_reward
            state, _ = self.Env.reset()
            state = self.transform_state(state, self.Env.initial_distance)
            
            terminated = False
            truncated = False
            self.E.clear()
            steps = 0
            total_reward = 0

            #get first action
            action = self.epsilon_greedy_policy(state)
            
            if log:
                self.Env.render()
                self.Env.logger.info(f"Action: {action}")

            while not terminated and not truncated:
                #perform action
                next_state, reward, terminated, truncated, info = self.Env.step(action)

                next_state = self.transform_state(next_state, self.Env.initial_distance)

                #accumulate steps and reward
                steps += 1
                total_reward += reward

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

                # move to next state and action pair
                state, action = next_state, next_action
                
                if log:
                    self.Env.render()
                    self.Env.logger.info(f"Action: {action}")
            
            # Append total rewards and steps after the episode ends
            self.total_reward_list.append(total_reward)
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

        if episode % checkpoint_interval == 0:
            print(f'Sum of rewards at episode {episode} is {total_reward}' )
            
        print(f"Truncated episodes: {truncated_count}")

        return self.total_reward_list, self.total_steps_list
    
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
    
    def evaluate(self, num_episodes=100, checkpoint_interval=50):
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
        all_rewards = []  # List to store total rewards per episode
        all_steps = []    # List to store the number of steps per episode
        checkpoint_rewards = []  # List to store rewards at each checkpoint

        for episode in tqdm(range(num_episodes)):
            state, _ = self.Env.reset()
            state = self.transform_state(state, self.Env.initial_distance)
            cumulative_reward = 0
            steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # Select action based on the trained policy without updating Q-values
                action = self.get_best_action(state)  # Use the trained policy
                
                # Take action in the environment
                next_state, reward, terminated, truncated, _ = self.Env.step(action)
                next_state = self.transform_state(next_state, self.Env.initial_distance)
                
                cumulative_reward += reward
                state = next_state
                steps += 1

            # Store results after each episode
            all_rewards.append(cumulative_reward)
            all_steps.append(steps)

            # Store rewards at checkpoint intervals
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_rewards.append(np.mean(all_rewards[-checkpoint_interval:]))

        return all_rewards, all_steps, checkpoint_rewards

    def plot_evaluation_metrics(self, rewards, steps, checkpoint_rewards, window_size=10):
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

        # # Plot checkpoint rewards if available
        # if checkpoint_rewards:
        #     plt.figure(figsize=(14, 6))
        #     plt.plot(np.arange(1, len(checkpoint_rewards) + 1) * checkpoint_interval, checkpoint_rewards, 'o-', color='purple', alpha=0.8)
        #     plt.title(f'Checkpoint Rewards (every {checkpoint_interval} episodes)')
        #     plt.xlabel('Episode')
        #     plt.ylabel('Average Reward')
        #     plt.show()

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
        self.clearance_rates = state[19:24]  # Clearance rates at time step t

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