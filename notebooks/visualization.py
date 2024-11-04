import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

def plot_training_metrics(rewards, steps, window_size=50):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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
    
    sns.scatterplot(data=df_rewards, x='Episode', y='Reward', 
                    alpha=0.3, color='blue', ax=ax1, label='Reward')
    sns.lineplot(data=df_rewards, x='Episode', y='Rolling Mean',
                 color='red', ax=ax1, linewidth=2, label=f'Rolling Mean (window={window_size})')
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')

    sns.scatterplot(data=df_steps, x='Episode', y='Steps',
                    alpha=0.3, color='blue', ax=ax2, label='Steps')
    sns.lineplot(data=df_steps, x='Episode', y='Rolling Mean',
                 color='red', ax=ax2, linewidth=2, label=f'Rolling Mean (window={window_size})')
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Steps')
    
    plt.tight_layout()
    plt.show()

def plot_snapshot(env, save_path="snapshots", save_id=None):

    n_lanes = env.lanes
    distance = env.distance
    current_lane = env.current_lane
    clearance_rates = env.clearance_rates

    fig, ax = plt.subplots(figsize=(8, 6))

    for lane in range(1, n_lanes + 1):
        color = 'blue' if lane == current_lane else 'lightgray'
        ax.add_patch(patches.Rectangle((0, lane), distance, 1, color=color, alpha=0.3))
        ax.text(distance + 50, lane + 0.5, f"Clearance: {clearance_rates[lane - 1]}", 
                va='center', ha='left', fontsize=10)

    ax.text(0, n_lanes + 1.5, f"Distance to Destination: {distance} meters", 
            fontsize=12, ha='left', color='black')

    ax.scatter([distance], [current_lane + 0.5], color='red', s=100, label="Agent")

    ax.set_xlim(-100, env.initial_distance + 100)
    ax.set_ylim(0.5, n_lanes + 1)
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Lanes")
    ax.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    if save_id is not None:
        filename = os.path.join(save_path, f"snapshot_{save_id:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_gif_from_snapshots(save_path="snapshots", gif_filename="agent_simulation.gif"):

    images = []
    for file_name in sorted(os.listdir(save_path)):
        if file_name.endswith(".png"):
            file_path = os.path.join(save_path, file_name)
            images.append(imageio.imread(file_path))
    
    imageio.mimsave(gif_filename, images, duration=0.5)
    print(f"GIF saved as {gif_filename}")
