import json
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from PIL import Image
import io

# Function to plot a snapshot of the current environment state
def plot_snapshot(env, save_path="snapshots", save_id=None, clearance_history=[]):
    """
    Plots a snapshot of the current environment state, showing the agent's position, 
    distance to the destination, and clearance rates of each lane.
    
    Args:
    - env (object): Environment object containing lane count, distance to destination, 
                    current lane, and clearance rates.
    - save_path (str): Path where snapshots will be saved. Default is 'snapshots'.
    - save_id (int, optional): ID for saving the snapshot image. If provided, saves the image;
                               otherwise, displays the image.
    - clearance_history (list): List of past clearance rates to display previous rates.
    """
    n_lanes = env.lanes
    distance = env.distance
    current_lane = env.current_lane
    clearance_rates = env.clearance_rates

    fig, ax = plt.subplots(figsize=(12, 9))

    for lane in range(1, n_lanes + 1):
        color = 'blue' if lane == current_lane else 'lightgray'
        ax.add_patch(patches.Rectangle((0, lane), distance, 1, color=color, alpha=0.3))

        if clearance_rates is not None and len(clearance_rates) >= lane:
            current_rate = clearance_rates[lane - 1]
            rate_color = 'gray'
            if current_rate >= 18.0:
                rate_color = 'green'
            elif current_rate <= 16.0:
                rate_color = 'red'
            ax.text(distance + 50, lane + 0.5, f"Clearance: {current_rate}", 
                    va='center', ha='left', fontsize=12, color=rate_color)
        else:
            print(f"Warning: Clearance rate data for lane {lane} is not available.")

    if len(clearance_history) >= 2:
        clearance_step_1 = clearance_history[-2] if len(clearance_history) > 1 else []
        clearance_step_2 = clearance_history[-1]
        clearance_text = f"Timestep -2 Clearance: {clearance_step_1}\nTimestep -1 Clearance: {clearance_step_2}"
    else:
        clearance_text = "No previous clearance data available."

    ax.text(distance + 200, n_lanes + 2, clearance_text,
            fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

    ax.text(0, n_lanes + 1.5, f"Distance to Destination: {distance} meters", 
            fontsize=12, ha='left', color='black')

    ax.scatter([distance], [current_lane + 0.5], color='red', s=100, label="Agent")

    ax.set_xlim(-100, distance + 300)  
    ax.set_ylim(0.5, n_lanes + 3)
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Lanes")
    ax.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    if save_id is not None:
        filename = os.path.join(save_path, f"snapshot_{save_id:04d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Function to create a GIF from saved snapshots
def create_gif_from_snapshots(save_path, gif_filename, target_size=(512, 512)):
    """
    Creates a GIF from saved snapshot images.

    Args:
    - save_path (str): Path where snapshot images are saved.
    - gif_filename (str): Name of the output GIF file.
    - target_size (tuple): Size of each image in the GIF. Default is (512, 512).
    """
    images = []

    for file_name in sorted(os.listdir(save_path)):
        if file_name.endswith(".png"):  
            file_path = os.path.join(save_path, file_name)
            image = Image.open(file_path)
            image = image.resize(target_size, Image.LANCZOS)
            images.append(image)

    if not images:
        print("Error: No images found to create GIF.")
        return

    imageio.mimsave(gif_filename, images, duration=0.1) 
    print(f"GIF saved as {gif_filename}")

# Function to load JSON data with handling for line-wrapped JSON format
def load_json_wrapped(json_path):
    """
    Loads JSON content from a file and handles line-wrapped format issues.

    Args:
    - json_path (str): Path to the JSON file.

    Returns:
    - list: Loaded content as a list of dictionaries.
    """
    with open(json_path, 'r') as f:
        try:
            content = f.readlines()
            wrapped_content = "[" + ",".join(content) + "]"  # Wraps all lines in a list
            data = json.loads(wrapped_content)
            return data
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            raise e

# Function to extract timesteps data from the first line of JSON and cache it for later use
def extract_timesteps_from_json(file_path):
    """
    Extracts timestep data from the first line of a JSON file and caches it for later use.

    Args:
    - file_path (str): Path to the JSON file containing timesteps data.
    """
    global cached_timesteps
    data_list = load_json_wrapped(file_path)  # Use wrapped method to load JSON file
    
    # Extract `Timesteps` data only from the first line
    if data_list and "Timesteps" in data_list[0]:
        cached_timesteps = data_list[0]["Timesteps"]
        print("Timesteps data from the first line successfully cached, containing {} timesteps".format(len(cached_timesteps)))
    else:
        print("No Timesteps data found in the first line")

def plot_snapshot_IO(env, clearance_history=[]):
    """
    Plots a snapshot of the environment state and returns it as an image object.
    Useful for generating GIFs or other uses without saving directly to a file.

    Args:
    - env (object): Environment object with lanes, distance, current lane, and clearance rates.
    - clearance_history (list): List of previous clearance rates for reference.

    Returns:
    - Image: The generated snapshot image.
    """
    n_lanes = env.lanes
    distance = env.distance
    current_lane = env.current_lane
    clearance_rates = env.clearance_rates

    fig, ax = plt.subplots(figsize=(12, 9))

    for lane in range(1, n_lanes + 1):
        color = 'blue' if lane == current_lane else 'lightgray'
        ax.add_patch(patches.Rectangle((0, lane), distance, 1, color=color, alpha=0.3))

        if clearance_rates is not None and len(clearance_rates) >= lane:
            current_rate = clearance_rates[lane - 1]
            rate_color = 'gray'
            if current_rate >= 18.0:
                rate_color = 'green'
            elif current_rate <= 16.0:
                rate_color = 'red'
            ax.text(distance + 50, lane + 0.5, f"Clearance: {current_rate}", 
                    va='center', ha='left', fontsize=12, color=rate_color)

    ax.text(0, n_lanes + 1.5, f"Distance to Destination: {distance} meters", 
            fontsize=12, ha='left', color='black') 

    ax.scatter([distance], [current_lane + 0.5], color='red', s=100, label="Agent")

    ax.set_xlim(0, distance + 300)  
    ax.set_ylim(0.5, 6)  
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Lanes")
    ax.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    buf.seek(0)
    return Image.open(buf)

# Function to create a visualization GIF from cached timestep data
def create_visualization_from_cached_data(gif_filename="agent_simulation_json.gif", step_interval=50):
    """
    Generates a GIF using cached timesteps data, capturing snapshots every 'step_interval' steps.

    Args:
    - gif_filename (str): Name of the output GIF file.
    - step_interval (int): Number of steps to skip between frames (e.g., every 50 steps).
    """
    clearance_history = []
    images = []

    for step_idx, step_data in enumerate(cached_timesteps):
        if step_idx % step_interval == 0:  # Only process every 'step_interval' steps
            lanes = 5 
            distance = step_data["State"][0]  
            current_lane = step_data["State"][1]  
            clearance_rates = step_data["State"][2:7]  

            clearance_history.append(clearance_rates)

            env_data = type('EnvData', (object,), {
                "lanes": lanes,
                "distance": distance,
                "current_lane": current_lane,
                "clearance_rates": clearance_rates
            })

            snapshot_image = plot_snapshot_IO(env_data, clearance_history=clearance_history)
            images.append(snapshot_image)

    if images:
        images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=500, loop=0)
        print(f"GIF saved at {gif_filename}")
    else:
        print("No images were generated to save as GIF.")
