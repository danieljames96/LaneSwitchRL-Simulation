import json
import numpy as np
import matplotlib.pyplot as plt


'''
Functions for Model Testing
- Average reward
- Average timestep
- Dotplots of rewards and timesteps per episode
- Boxplots of rewards and timesteps per episode (min, max, median, std dev)
'''

def format_state(state):
    """
    Formats the state array so that specific indices are integers, 
    and others are rounded to one decimal place.
    State: [distance, current lane, clearance rates * 5 lanes] * 3 timesteps
    """
    formatted_state = []
    for i, value in enumerate(state):
        if i in [1, 8, 15]:  # Indices to be converted to integers
            formatted_state.append(int(value))
        else:
            formatted_state.append(round(value, 1))  # Round to one decimal place
    return formatted_state


def test_model(model, env, num_episodes, output_file=None):
    """
    Tests the model over a specified number of episodes, records test details for each episode.

    Args:
    - model: Trained model to be tested.
    - env (gym.Env): The environment to evaluate the model on.
    - num_episodes (int): Total number of episodes to run the evaluation.
    - output_file (str, optional): The path to the JSON file where episode details will be saved. If not provided,
      a timestamped default filename will be used.

    Returns:
    - episode_rewards (list): List of total rewards for each episode.
    - timesteps (list): List of total timestpes for each episode.
    - JSON file: record all details of each episode and each timestep.
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
                "Initial State": format_state(obs.tolist() if hasattr(obs, 'tolist') else obs),
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
                    "State": format_state(obs.tolist() if hasattr(obs, 'tolist') else obs),
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


def plot_test_dotplots(rewards, timesteps, interval=10):
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


def plot_test_boxplots(json_file):
    """
    Reads the JSON file generated by test_model, extracts the total rewards and timesteps per episode,
    and creates a single set of side-by-side box plots with outliers.
    Annotates min, median, max, and standard deviation considering outliers.

    Args:
    - json_file (str): Path to the JSON file with episode data.
    """
    episode_rewards = []
    episode_timesteps = []

    # Open and read the JSON file
    with open(json_file, 'r') as f:
        for line in f:
            episode_data = json.loads(line)
            if "Total Reward" in episode_data and "Total Timesteps" in episode_data:
                episode_rewards.append(episode_data["Total Reward"])
                episode_timesteps.append(episode_data["Total Timesteps"])

    # Calculate statistics with outliers
    min_reward = round(min(episode_rewards))
    max_reward = round(max(episode_rewards))
    median_reward = round(np.median(episode_rewards))
    std_reward = round(np.std(episode_rewards))

    min_timesteps = round(min(episode_timesteps))
    max_timesteps = round(max(episode_timesteps))
    median_timesteps = round(np.median(episode_timesteps))
    std_timesteps = round(np.std(episode_timesteps))

    # Create subplots for side-by-side box plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the box plot for rewards per episode with outliers
    rewards_box = ax1.boxplot(episode_rewards, vert=True, patch_artist=True, showfliers=True)
    ax1.set_title("Box Plot of Rewards")
    ax1.set_ylabel("Reward")

    # Annotate min, median, max, and standard deviation for rewards
    ax1.text(1.1, min_reward, f'Min: {min_reward}', ha='left', va='center', color="blue")
    ax1.text(1.1, median_reward, f'Median: {median_reward}', ha='left', va='center', color="blue")
    ax1.text(1.1, max_reward, f'Max: {max_reward}', ha='left', va='center', color="blue")
    ax1.text(0.05, 0.05, f'Std Dev: {std_reward}', ha='left', va='bottom', color="blue",
             transform=ax1.transAxes, fontsize=10)

    # Plot the box plot for timesteps per episode with outliers
    timesteps_box = ax2.boxplot(episode_timesteps, vert=True, patch_artist=True, showfliers=True)
    ax2.set_title("Box Plot of Timesteps")
    ax2.set_ylabel("Timesteps")

    # Annotate min, median, max, and standard deviation for timesteps
    ax2.text(1.1, min_timesteps, f'Min: {min_timesteps}', ha='left', va='center', color="green")
    ax2.text(1.1, median_timesteps, f'Median: {median_timesteps}', ha='left', va='center', color="green")
    ax2.text(1.1, max_timesteps, f'Max: {max_timesteps}', ha='left', va='center', color="green")
    ax2.text(0.05, 0.05, f'Std Dev: {std_timesteps}', ha='left', va='bottom', color="green",
             transform=ax2.transAxes, fontsize=10)

    # Display the plots
    plt.tight_layout()
    plt.show()

def calculate_action_percentages(log_file):
    """
    Calculates and prints the distribution of actions taken at each timestep across multiple episodes, based on data from a JSON log file.

    Args:
    - log_file (str): Path to the JSON file containing log data for multiple episodes.
      Each line in the file represents an episode in JSON format, with each episode containing a list of "Timesteps".
      Each timestep includes an "Action" field, representing the action taken at that timestep.
    """
    data = []
    
    # Load the JSON file
    with open(log_file, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and add it to the list
            data.append(json.loads(line))
    
    # Initialize counters for each action
    action_counts = {-1: 0, 0: 0, 1: 0}
    total_actions = 0
    total_num_of_episodes = len(data)  # Count the total number of episodes

    # Iterate over all episodes and timesteps
    for episode in data:
        for timestep in episode["Timesteps"]:
            action = timestep["Action"]
            if action in action_counts:
                action_counts[action] += 1
            total_actions += 1

    # Calculate the percentage of each action
    action_percentages = {action: (count / total_actions) * 100 for action, count in action_counts.items()}

    # Print the results
    print(f"Total actions across {total_num_of_episodes} episodes: {total_actions}")
    for action, percentage in action_percentages.items():
        print(f"Action {action}: {percentage:.1f}%")


'''
Functions for Clerance Rate Analysis
- Percentage of timesteps that clearance rates < 5
- Percentage of timesteps that clearance rates < 10
- Boxplots of clearance rates of 5 lanes
'''
def calculate_clearance_rate_percentages(log_file):
    """
    Calculates and prints the percentage of timesteps where the clearance rate in each lane is 
    below specified thresholds (5 and 10) across test episodes, based on data from a JSON log file.

    Args:
    - log_file (str): Path to the JSON file containing log data for test episodes.
      Each line in the file represents an episode in JSON format, with an "Initial State" and a list of "Timesteps".
      Each timestep contains a "State" field with clearance rates for 5 lanes.
    """
    # Load the JSON file
    with open(log_file, 'r') as f:
        data = [json.loads(line) for line in f]  # Load each line as a JSON object

    # Initialize counters
    lane_counts_less_than_5 = [0] * 5  # Counts for clearance rate < 5 for each lane
    lane_counts_less_than_10 = [0] * 5  # Counts for clearance rate < 10 for each lane
    total_timesteps = 0  # Total number of timesteps across all episodes
    num_of_episodes = len(data)  # Count the total number of episodes

    # Iterate over all episodes
    for episode in data:
        # Include clearance rates from the initial state
        initial_clearance_rates = episode["Initial State"][-5:]
        total_timesteps += 1  # Count the initial state as a timestep

        for i, rate in enumerate(initial_clearance_rates):
            if rate < 5:
                lane_counts_less_than_5[i] += 1
            if rate < 10:
                lane_counts_less_than_10[i] += 1

        # Iterate over each timestep in the episode
        for timestep in episode["Timesteps"]:
            # Extract the last 5 values in "State" as clearance rates for the 5 lanes
            clearance_rates = timestep["State"][-5:]
            total_timesteps += 1

            # Count occurrences where clearance rate < 5 and < 10 for each lane
            for i, rate in enumerate(clearance_rates):
                if rate < 5:
                    lane_counts_less_than_5[i] += 1
                if rate < 10:
                    lane_counts_less_than_10[i] += 1

    # Calculate percentages for each lane
    lane_percentages_less_than_5 = [(count / total_timesteps) * 100 for count in lane_counts_less_than_5]
    lane_percentages_less_than_10 = [(count / total_timesteps) * 100 for count in lane_counts_less_than_10]

    # Print the results
    print(f"Total timesteps across {num_of_episodes} episodes (including initial states): {total_timesteps}")
    for i in range(5):
        print(f"Lane {i + 1}:")
        print(f"  {lane_percentages_less_than_5[i]:.1f}% of timesteps had a clearance rate less than 5")
        print(f"  {lane_percentages_less_than_10[i]:.1f}% of timesteps had a clearance rate less than 10")


def plot_clearance_rate_boxplots(log_file):
    """
    Generates side-by-side box plots to visualize the distribution of clearance rates across 5 lanes,
    with and without outliers, based on data from a JSON log file.

    Args:
    - log_file (str): Path to the JSON file containing log data for multiple episodes.
      Each line in the file represents an episode in JSON format, with an "Initial State" and a list of "Timesteps".
      Each "State" field in the initial state and timesteps contains clearance rates for 5 lanes.
    """
    # Initialize lists to hold clearance rates for each lane
    clearance_rates = [[] for _ in range(5)]  # One list per lane

    # Load the JSON file
    with open(log_file, 'r') as f:
        data = [json.loads(line) for line in f]  # Load each line as a JSON object

    # Iterate over all episodes
    for episode in data:
        # Include clearance rates from the initial state
        initial_clearance_rates = episode["Initial State"][-5:]
        for i, rate in enumerate(initial_clearance_rates):
            clearance_rates[i].append(rate)

        # Iterate over each timestep in the episode
        for timestep in episode["Timesteps"]:
            # Extract the last 5 values in "State" as clearance rates for the 5 lanes
            timestep_clearance_rates = timestep["State"][-5:]
            for i, rate in enumerate(timestep_clearance_rates):
                clearance_rates[i].append(rate)

    # Create subplots for side-by-side box plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the box plot with outliers
    axs[0].boxplot(clearance_rates, tick_labels=[f'Lane {i+1}' for i in range(5)])
    axs[0].set_title("Clearance Rate Distribution Across 5 Lanes (With Outliers)")
    axs[0].set_xlabel("Lane")
    axs[0].set_ylabel("Clearance Rate")

    # Plot the box plot without outliers
    axs[1].boxplot(clearance_rates, tick_labels=[f'Lane {i+1}' for i in range(5)], showfliers=False)
    axs[1].set_title("Clearance Rate Distribution Across 5 Lanes (Without Outliers)")
    axs[1].set_xlabel("Lane")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
