import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class TrafficEnvironment(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self, lanes=5, initial_distance=4000, reward_shaping_flag=False):
        """
        Args:
        - lanes (int): Number of lanes (default is 5).
        - initial_distance (int): The distance from destination.
        - reward_shaping_flag (bool): Flag to enable or disable reward shaping.
        """
        super(TrafficEnvironment, self).__init__()
        
        self.lanes = lanes
        self.initial_distance = initial_distance
        self.state = None
        self.rounding_precision = 1
        self.clearance_rate_min = 0
        self.max_time_steps = 10000

        self.reward_shaping_flag = reward_shaping_flag
        
        # Define action space (0: left, 1: stay, 2: right)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: (distance, current lane, clearance rates)
        obs_dim = 3 * (2 + self.lanes)
        self.observation_space = spaces.Box(low=0, high=4000, shape=(obs_dim,), dtype=np.float32)
        
        # Define action mapping
        self.action_mapping = {0: -1, 1: 0, 2: 1}
        
        self.reset()

    def reset(self, seed=None):
        """
        Resets the environment to the initial state.
        Returns:
        - state (np.array): The initial observation of the environment.
        - info (dict): Additional Information
        """
        super().reset(seed=seed)
        # Initialize the state
        self.distance = self.initial_distance
        self.current_lane = 1  # Always start from first lane
        # Initialize clearance rates randomly between 15 and 20 for all lanes
        self.clearance_rates = np.round(np.random.uniform(15, 20, size=self.lanes), self.rounding_precision)

        # Reset state history
        self.state_history = [(self.distance, self.current_lane, *self.clearance_rates)]
        self.time_step = 0
        
        obs = self._get_obs()
        info = {}
        
        # Return the initial state
        return obs, info

    def _get_obs(self):
        """
        Constructs the current observation from the state history.
        Returns:
        - flat_state (np.array): Flattened 1D observation of the state.
        """
        history_length = len(self.state_history)
        if history_length < 3:
            padding = [self.state_history[0]] * (3 - history_length)
            padded_history = padding + self.state_history
        else:
            padded_history = self.state_history[-3:]
        
        # Convert to a single observation array
        flat_state = np.concatenate([np.array(state) for state in padded_history], axis=None)
        
        return flat_state.astype(np.float32)

    def _attempt_lane_change(self, action):
        """
        Attempts to change lane based on the action.
        Args:
        - action (int): -1 for left, 1 for right.
        Returns:
        - penalty (float): The penalty to be applied to the reward.
        """
        penalty = -5  # Penalty for attempting a lane change

        # Check if the lane change is within bounds
        new_lane = self.current_lane + action
        if 1 <= new_lane <= self.lanes:
            # Attempt the lane change with a 50% success rate
            if random.random() < 0.5:
                self.current_lane = new_lane  # Lane change succeeds
        # If the lane change is invalid, the penalty is still applied

        return penalty

    def _update_clearance_rates(self):
        # Update clearance rates based on adjacent lanes' speeds and add uncertainty term N(0, 0.1)
        updated_rates = self.clearance_rates.copy()
        for i in range(self.lanes):
            # Adding the uncertainty term N(0, 0.1)
            uncertainty = np.random.normal(0, 0.1)

            # Random Event: 5% chance of slowdown (20%-50%) and 5% chance of speedup (20%-40%)
            random_event = random.random()
            # 5% chance of slowdown
            if random_event < 0.05:
                updated_rates[i] -= self.clearance_rates[i] * random.uniform(0.2, 0.5)
            # 5% chance of speedup
            elif random_event >= 0.05 and random_event < 0.1:
                updated_rates[i] += self.clearance_rates[i] * random.uniform(0.2, 0.4)

            # Update based on adjacent lanes, with sgn function and uncertainty
            if i > 0:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i - 1] - self.clearance_rates[i])
            if i < self.lanes - 1:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i + 1] - self.clearance_rates[i])

            # Apply the uncertainty term to each lane
            updated_rates[i] += uncertainty

            # Round clearance rates
            updated_rates[i] = round(updated_rates[i], self.rounding_precision)

            # Ensure clearance rates don't drop below a minimum threshold
            updated_rates[i] = max(updated_rates[i], self.clearance_rate_min)

        self.clearance_rates = updated_rates
    
    def step(self, action):
        """
        Takes an action and returns the next state, reward, terminated, truncated, and info.
        Args:
        - action (int): The action taken by the agent (0 for left, 1 for stay, 2 for right).
        Returns:
        - next_state (np.array): The next observation after taking the action.
        - reward (float): The reward received after taking the action.
        - terminated (bool): Whether the episode ended naturally (e.g., reaching the destination).
        - truncated (bool): Whether the episode ended due to truncation (e.g., exceeding max time steps).
        - info (dict): Additional information about the environment.
        """
        # Map the discrete action to the original action space (-1, 0, 1)
        mapped_action = self.action_mapping[action]
        
        reward = 0
        terminated = False
        truncated = False

        # Handle lane change
        if mapped_action != 0:
            reward += self._attempt_lane_change(mapped_action)
        # No action needed for staying in the current lane

        # Update lane clearance rates based on neighboring lanes
        self._update_clearance_rates()

        # Compute the distance covered in the current lane
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        distance_covered = min(clearance_rate, self.distance)
        self.distance -= distance_covered
        self.distance = round(self.distance, self.rounding_precision)

        # Time penalty
        reward += distance_covered - 10
        reward = round(reward, self.rounding_precision)

        # Now, after all the updates for the current time step, append the state to history
        self.state_history.append((self.distance, self.current_lane, *self.clearance_rates))

        # Check if the episode is done (if the destination is reached)
        if self.distance <= 0:
            terminated = True
            self.distance = 0

        # Check if the episode is truncated (e.g., max time steps reached)
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            truncated = True

        # Reward shaping: Penalty if the agent stayed in a slower lane for the last 3 timesteps
        reward_shaping = 0
        if self.reward_shaping_flag and len(self.state_history) >= 3:
            # Check if the agent has stayed in the same lane for the last 3 timesteps
            stayed_in_same_lane = (
                self.state_history[-1][1] == self.state_history[-2][1] == self.state_history[-3][1]
            )
            
            if stayed_in_same_lane:
                current_lane_index = self.current_lane - 1  # Adjust for zero-indexed array
                slower_in_all_timesteps = True  # Assume condition is met unless proven otherwise

                for t in range(-3, 0):  # Check the last three timesteps
                    # Current lane's clearance at timestep t
                    current_clearance = self.state_history[t][2 + current_lane_index]
                    
                    # Get the clearance of adjacent lanes if they exist
                    left_lane_clearance = self.state_history[t][2 + current_lane_index - 1] if current_lane_index > 0 else None
                    right_lane_clearance = self.state_history[t][2 + current_lane_index + 1] if current_lane_index < self.lanes - 1 else None

                    # Check conditions for left and right lanes
                    left_lane_faster_and_clear = (
                        left_lane_clearance is not None and 
                        left_lane_clearance > 5 and 
                        left_lane_clearance > current_clearance
                    )
                    right_lane_faster_and_clear = (
                        right_lane_clearance is not None and 
                        right_lane_clearance > 5 and 
                        right_lane_clearance > current_clearance
                    )

                    # The penalty condition: at least one adjacent lane must be faster and clearer than the current lane
                    if not (left_lane_faster_and_clear or right_lane_faster_and_clear):
                        # If this condition fails for any timestep, mark as False and stop checking
                        slower_in_all_timesteps = False
                        break

                # Apply penalty if the condition is met for all three timesteps
                if slower_in_all_timesteps:
                    reward_shaping = -600
                    reward += reward_shaping

        # Return the next observation using _get_obs()
        next_state = self._get_obs()
        info = {"reward_shaping": reward_shaping}  

        return next_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Renders the environment state to the console.
        Args:
        - mode (str): The mode of rendering. 'human' for console output.
        """
        if mode == 'human':
            print(f"Time Step: {self.time_step}")
            print(f"Distance to Destination: {self.distance}")
            print(f"Current Lane: {self.current_lane}")
            print(f"Clearance Rates: {self.clearance_rates}")