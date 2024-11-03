import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import logging

class CustomTrafficEnvironment(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self, lanes=5, initial_distance=4000, max_fatigue=10, max_fatigue_penalty=50, fatigue_growth='linear', rain_probability=0.1, max_time_steps = 10000, logging_level=logging.INFO):
        """
        Args:
        - lanes (int): Number of lanes (default is 5).
        - initial_distance (int): The distance from destination.
        """
        super(CustomTrafficEnvironment, self).__init__()
        
        # Configure self.logger
        # self.logger.basicConfig(level=self.logger_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        log_file='./logs/environment_logs/custom_traffic_env.log'
        
        # Only add handlers if they are not already attached
        if not self.logger.hasHandlers():
            # File handler for writing logs to a file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging_level)
            
            # Optional: Console handler for showing logs in the notebook output
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)

            # Set a format for the log messages
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add both handlers to the logger
            self.logger.addHandler(file_handler)
            # self.logger.addHandler(console_handler)
        
        self.lanes = lanes
        self.initial_distance = initial_distance
        self.max_fatigue = max_fatigue
        self.fatigue_growth = fatigue_growth
        self.rain_probability = rain_probability
        self.max_time_steps = max_time_steps
        self.max_fatigue_penalty = max_fatigue_penalty
        self.is_raining = False
        self.state_history = []
        self.rounding_precision = 1
        self.clearance_rate_min = 5
        self.fatigue_counter = 0
        
        # Define action space (-1: left, 0: stay, 1: right, 2: rest)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: (distance, current lane, fatigue counter, clearance rates)
        low_obs = np.array([0, 1, 0] + [self.clearance_rate_min] * self.lanes, dtype=np.float32)
        high_obs = np.array([self.initial_distance] + [lanes] + [self.max_fatigue] + [float('inf')] * self.lanes, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Define action mapping
        self.action_mapping = {0: -1, 1: 0, 2: 1, 3: 2}
        
        self.reset()
        self.logger.debug("Environment initialized.")

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        Returns:
        - state (np.array): The initial observation of the environment.
        - info (dict): Additional Information
        """
        super().reset(seed=seed)
        
        # Initialize the state
        self.distance = self.initial_distance
        if options:
            self.current_lane = options.get('starting_lane', 1)
        else:
            self.current_lane = 1
        self.is_raining = False
        # Initialize clearance rates randomly between 15 and 20 for all lanes
        self.clearance_rates = np.round(np.random.uniform(15, 20, size=self.lanes), self.rounding_precision)
        self.fatigue_counter = 0

        # Reset state history
        self.state_history = [(self.distance, self.current_lane, self.fatigue_counter, *self.clearance_rates)]
        self.time_step = 0
        
        self.logger.debug("Environment reset.")
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
        
        self.logger.debug(f"Observation generated: {flat_state}")
        return flat_state.astype(np.float32)
    
    def _calculate_fatigue_penalty(self):
        """
        Calculate the fatigue penalty based on the growth function.
        """
        if self.fatigue_counter >= self.max_fatigue:
            penalty = self.max_fatigue_penalty
        else:
            if self.fatigue_growth == 'linear':
                penalty = 2 * self.fatigue_counter
            elif self.fatigue_growth == 'quadratic':
                penalty = self.fatigue_counter ** 2
            else:
                penalty = self.fatigue_counter
        
        self.logger.debug(f"Fatigue penalty calculated: {penalty}")
        return penalty
        
    def _attempt_lane_change(self, action):
        """
        Attempts to change lane based on the action.
        Args:
        - action (int): -1 for left, 1 for right.
        Returns:
        - penalty (float): The penalty to be applied to the reward.
        """
        penalty = -5
        new_lane = self.current_lane + action
        if 1 <= new_lane <= self.lanes:
            if random.random() < 0.5:
                self.current_lane = new_lane
                self.logger.debug(f"Lane change succeeded to lane {self.current_lane}")
            else:
                self.logger.debug("Lane change failed.")
        else:
            self.logger.warning(f"Lane change action out of bounds. Current lane: {self.current_lane}, New Lane: {new_lane},  Action: {action}")
        
        return penalty

    def _update_clearance_rates(self):
        # Update clearance rates based on adjacent lanes' speeds and add uncertainty term N(0, 0.1)
        updated_rates = self.clearance_rates.copy()
        
        if random.random() < self.rain_probability:
            self.is_raining = True
            self.logger.debug("Rain started.")
        else:
            self.is_raining = False
            self.logger.debug("No rain.")
        
        for i in range(self.lanes):
            # Apply lane-specific normal distributions if it's raining
            if self.is_raining:
                if i == 0 or i == self.lanes - 1:  # Lanes 1 and 5
                    uncertainty = np.random.normal(-0.2, 0.1)
                elif i == 1 or i == 3:  # Lanes 2 and 4
                    uncertainty = np.random.normal(-0.1, 0.1)
                else:  # Lane 3
                    uncertainty = np.random.normal(0, 0.1)
            else:
                uncertainty = np.random.normal(0, 0.1)  # Default non-rainy condition

            # Random Event: 5% chance of slowdown (20%-50%) and 5% chance of speedup (20%-40%)
            random_event = random.random()
            # 5% chance of slowdown
            if random_event < 0.05:
                updated_rates[i] -= self.clearance_rates[i] * random.uniform(0.2, 0.5)
                self.logger.debug(f"Clearance rate slowed in lane {i+1}")
            # 5% chance of speedup
            elif random_event >= 0.05 and random_event < 0.1:
                updated_rates[i] += self.clearance_rates[i] * random.uniform(0.2, 0.4)
                self.logger.debug(f"Clearance rate increased in lane {i+1}")

            # Update based on adjacent lanes, with sgn function and uncertainty
            if i > 0:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i - 1] - self.clearance_rates[i])
            if i < self.lanes - 1:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i + 1] - self.clearance_rates[i])

            updated_rates[i] += uncertainty
            updated_rates[i] = round(updated_rates[i], self.rounding_precision)
            updated_rates[i] = max(updated_rates[i], self.clearance_rate_min)
            
            # self.logger.debug(f"Lane {i+1} clearance rate updated from {self.clearance_rates[i]} to {updated_rates[i]}")

        self.clearance_rates = updated_rates
        self.logger.debug("Clearance rates updated.")

    def step(self, action):
        """
        Takes an action and returns the next state, reward, terminated, truncated, and info.
        Args:
        - action (int): The action taken by the agent (-1 for left, 0 for stay, 1 for right).
        Returns:
        - next_state (np.array): The next observation after taking the action.
        - reward (float): The reward received after taking the action.
        - terminated (bool): Whether the episode ended naturally (e.g., reaching the destination).
        - truncated (bool): Whether the episode ended due to truncation (e.g., exceeding max time steps).
        - info (dict): Additional information about the environment.
        """
        # Map the discrete action to the original action space (-1, 0, 1, 2)
        mapped_action = self.action_mapping[action]
        
        reward = 0
        terminated = False
        truncated = False
        
        # Time penalty
        reward -= 10
        
        # Handle 'rest' action
        if mapped_action == 2:
            if self.current_lane == 1 or self.current_lane == self.lanes:
                self.fatigue_counter = 0 # Reset fatigue counter
                reward -= 20
                self.logger.debug("Rest action taken; fatigue counter reset.")
            else:
                # Invalid rest action, penalize for attempting to rest in middle lanes
                reward -= 30
                self.logger.warning("Invalid rest action in middle lane.")
            self._update_clearance_rates()
        
        else:
            # Increment fatigue counter and calculate fatigue penalty
            # if random.random() < 0.5 and self.fatigue_counter < self.max_fatigue:
            #     self.fatigue_counter += 1
            
            # fatigue_penalty = self._calculate_fatigue_penalty()
            # reward -= fatigue_penalty

            # Handle lane change if not staying
            if mapped_action != 0:
                reward += self._attempt_lane_change(mapped_action)
            # No action needed for staying in the current lane

            # Update lane clearance rates based on neighboring lanes
            self._update_clearance_rates()

            # Compute the distance covered in the current lane
            clearance_rate = self.clearance_rates[self.current_lane - 1]
            distance_covered = clearance_rate # if self.fatigue_counter < self.max_fatigue else clearance_rate / 2
            self.distance -= distance_covered
            self.distance = round(self.distance, self.rounding_precision)

            # Reward for distance covered
            reward += distance_covered
        
        reward = round(reward, self.rounding_precision)

        # Now, after all the updates for the current time step, append the state to history
        self.state_history.append((self.distance, self.current_lane, self.fatigue_counter, *self.clearance_rates))

        # Check if the episode is done (if the destination is reached)
        if self.distance <= 0:
            terminated = True
            self.distance = 0
            self.logger.debug("Destination reached; episode terminated.")

        # Check if the episode is truncated (e.g., max time steps reached)
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            truncated = True
            self.logger.debug("Maximum time steps reached; episode truncated.")
        
        # Return the next observation using _get_obs()
        next_state = self._get_obs()
        info = {}
        self.logger.debug(f"Step completed with reward: {reward}")
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Renders the environment state to the console.
        Args:
        - mode (str): The mode of rendering. 'human' for console output.
        """
        if mode == 'human':
            self.logger.info(f"Time Step: {self.time_step}")
            self.logger.info(f"Distance to Destination: {self.distance}")
            self.logger.info(f"Current Lane: {self.current_lane}")
            self.logger.info(f"Clearance Rates: {self.clearance_rates}")
            self.logger.info(f"Fatigue Counter: {self.fatigue_counter}")
            self.logger.info(f"Is Raining: {self.is_raining}")
            self.logger.info('**************************************************')