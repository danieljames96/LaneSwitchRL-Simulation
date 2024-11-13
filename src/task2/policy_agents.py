import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

class PolicyAgents:
    def __init__(self, env, model):
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
        self.model = model
    
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
    
    def evaluate(self, num_episodes, output_file=None):
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
        all_episode_reward_types = []
        early_termination_count = 0
        action_distribution = []

        # Create a JSON file for writing
        with open(output_file, 'w') as f:
            # Run the model for the specified number of episodes
            for episode in tqdm(range(num_episodes)):
                obs, info = self.Env.reset(seed=self.seed_value+episode)
                episode_reward = 0
                episode_timestep = 0
                action_count = [0, 0, 0]
                episode_details = {
                    "Episode": episode + 1,
                    "Initial State": self.format_state(obs.tolist() if hasattr(obs, 'tolist') else obs),
                    "Timesteps": []
                }
                
                terminated, truncated = False, False
                reward_type_values = info

                while True:  # Run until the episode ends
                    action, _states = self.model.predict(obs, deterministic=True)
                    action = int(action)
                    mapped_action = action - 1
                    obs, reward, terminated, truncated, info = self.Env.step(action)
                    episode_reward += reward
                    episode_timestep += 1
                    action_count[action] += 1
                    
                    for key in info.keys():
                        if key not in reward_type_values:
                            reward_type_values[key] = 0
                        reward_type_values[key] += round(info[key],1)

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
                        early_termination_count += int(truncated)

                        # Add total reward and timestep count to episode details
                        episode_details["Total Reward"] = episode_reward
                        episode_details["Total Timesteps"] = episode_timestep

                        # Write the full episode details as a JSON object to the file
                        f.write(json.dumps(episode_details) + "\n")
                        break
                
                all_episode_reward_types.append(reward_type_values)  
                action_distribution.append(action_count)

        # Calculate average reward
        reward_ave = round(sum(rewards) / num_episodes)
        print(f"Average reward of {num_episodes} episodes is {reward_ave}.")

        # Calculate average timestep
        timestep_ave = round(sum(timesteps) / num_episodes)
        print(f"Average timesteps of {num_episodes} episodes is {timestep_ave}.")
        
        df = pd.DataFrame(all_episode_reward_types)
        all_episode_reward_types = df.mean().to_dict()
        
        action_distribution = [sum(x) / len(x) for x in zip(*action_distribution)]    
        
        print(f"Early terminations: {early_termination_count}")
        
        return rewards, timesteps, all_episode_reward_types, action_distribution
    
    def plot_test_results(self, rewards, timesteps, interval=10):
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
        
    def analyze_model_actions(self, action_dist):
        """
        Analyze action distribution from a list of action counts.
        """

        # Analyze action distribution
        left = int(action_dist[0])
        stay = int(action_dist[1])
        right = int(action_dist[2])
        
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