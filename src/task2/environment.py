import json
import logging
import os
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomTrafficEnvironment(gym.Env):
    """
    A custom environment simulating a multi-lane traffic scenario with dynamic conditions,
    including rain and accidents. The agent aims to minimize penalties and reach the destination
    by changing lanes and managing risk.

    Attributes:
        lanes (int): Number of lanes in the environment.
        initial_distance (int): Initial distance from the starting point to the destination.
        max_time_steps (int): Maximum allowable time steps per episode.
        clearance_rates (np.array): Array storing the current clearance rates for each lane.
        is_raining (bool): Flag indicating if it's raining.
        risk_factor (float): Risk associated with lane changes and speed.
    """
    metadata = {'render.modes':['human']}
    
    def __init__(self, lanes=5, initial_distance=4000, max_time_steps = 10000, seed=None, logging_enabled=False, config=None):
        """
        Initializes the environment, setting up lanes, initial distance, time steps,
        and other parameters related to penalties and conditions.

        Parameters:
            lanes (int): Number of lanes in the environment.
            initial_distance (int): Distance from the starting point to the destination.
            max_time_steps (int): Maximum time steps allowed per episode.
            seed (int, optional): Seed for reproducibility.
            logging_enabled (bool): Enable detailed logging if True.
            config (dict or str, optional): Configuration dictionary or path to JSON file.
        """
        
        super(CustomTrafficEnvironment, self).__init__()
        
        self.seed_value = seed
        self.lanes = lanes
        self.max_time_steps = max_time_steps
        self.initial_distance = initial_distance
        self.logger = self._initialize_logger(logging_enabled)
        self.logging_enabled = logging_enabled
        self.config = self._load_config(config)
        
        # Initialize other environment parameters
        self._initialize_parameters()
        self._define_action_observation_space()

        # Reset to the initial environment state
        self.reset()
    
    def _load_config(self, config):
        """
        Loads configuration from a dictionary or JSON file.

        Parameters:
            config (dict or str): Configuration dictionary or path to JSON file.

        Returns:
            dict: Configuration parameters.
        """
        if config is None:
            return {}  # Return empty dict if no config is provided
        elif isinstance(config, str):
            # If config is a file path, load the JSON file
            if os.path.isfile(config):
                with open(config, 'r') as f:
                    config_data = json.load(f)
                return config_data
            else:
                raise FileNotFoundError(f"Config file {config} not found.")
        elif isinstance(config, dict):
            # If config is a dictionary, use it directly
            return config
        else:
            raise ValueError("Config must be a dictionary or a valid file path to a JSON file.")
    
    def _initialize_logger(self, logging_enabled):
        """Sets up logging for the environment if logging is enabled."""
        logger = logging.getLogger(__name__)
        if logging_enabled and not logger.hasHandlers():
            handler = logging.FileHandler('./logs/task2/custom_traffic_env.log')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_parameters(self):
        """Initializes parameters related to penalties, probabilities, and risk factors."""
        self.clearance_rate_min = self.config.get('clearance_rate_min', 5)
        self.clearance_rate_max = self.config.get('clearance_rate_max', 30)
        self.clearance_rate_change_factor = self.config.get('clearance_rate_change_factor', 0.2)
        self.rain_probability = self.config.get('rain_probability', 0.2)
        self.rain_edge_lane_effect = self.config.get('rain_edge_lane_effect', -0.3)
        self.rain_center_lane_effect = self.config.get('rain_center_lane_effect', -0.2)
        self.accident_threshold = self.config.get('accident_threshold', 0.9)
        self.base_accident_probability = self.config.get('base_accident_probability', 0.001)
        self.speed_limit = self.config.get('speed_limit', 25)
        self.speed_limit_penalty = self.config.get('speed_limit_penalty', -6)
        self.safe_speed = self.config.get('safe_speed', 10)
        self.accident_penalty = self.config.get('accident_penalty', -85)
        self.high_risk_threshold = self.config.get('high_risk_threshold', 0.8 * self.accident_threshold)
        self.high_risk_penalty = self.config.get('high_risk_penalty', -4)
        self.lane_change_risk = self.config.get('lane_change_risk', 0.05)
        self.high_speed_risk = self.config.get('high_speed_risk', 0.15)
        self.lane_change_penalty = self.config.get('lane_change_penalty', 0)
        self.time_penalty = self.config.get('time_penalty', -2)
        self.wrong_lane_penalty = self.config.get('wrong_lane_penalty', -30)
        self.low_clearance_penalty_factor = self.config.get('low_clearance_penalty_factor', 1.6)
        self.lane_change_probability = self.config.get('lane_change_probability', 0.6)
        self.slowdown_probability = self.config.get('slowdown_probability', 0.05)
        self.speed_up_probability = self.config.get('speed_up_probability', 0.05)
        self.slowdown_factor = self.config.get('slowdown_factor', (0.2, 0.5))
        self.speed_up_factor = self.config.get('speed_up_factor', (0.2, 0.4))
        self.distance_reward_factor = self.config.get('distance_reward_factor', 0)
        self.rounding_precision = self.config.get('rounding_precision', 1)
        self.action_mapping = self.config.get('action_mapping', {0: -1, 1: 0, 2: 1})
    
    def _define_action_observation_space(self):
        """Defines the action and observation spaces for the environment."""
        self.action_space = spaces.Discrete(3)  # Actions: left, stay, right
        low_obs = np.array([0, 1, 0] + [self.clearance_rate_min] * self.lanes, dtype=np.float32)
        high_obs = np.array([self.initial_distance] + [self.lanes] + [1] + [self.clearance_rate_max] * self.lanes, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    
    def set_seed(self, seed):
        # Set the seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _log(self, message, level=logging.INFO):
        """
        Log messages if logging is enabled.
        
        Parameters:
            message (str): Message to log.
            level (int): Logging level (e.g., INFO, DEBUG).
        """
        if self.logging_enabled:
            self.logger.log(level, message)
    
    def enable_logging(self):
        """Enable logging for debugging purposes."""
        self.logging_enabled = True

    def disable_logging(self):
        """Disable logging to reduce console output."""
        self.logging_enabled = False
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        Returns:
        - state (np.array): The initial observation of the environment.
        - info (dict): Additional Information
        """
        self.seed_value = seed if seed is not None else self.seed_value
        super().reset(seed=self.seed_value)
        
        if seed is not None:
            self.set_seed(seed)
        
        # Initialize the state
        self.distance = self.initial_distance
        self.current_lane = options.get('starting_lane', 1) if options else 1
        self.is_raining = False
        self.risk_factor = 0
        self.time_step = 0
        
        # Initialize clearance rates randomly between 15 and 20 for all lanes
        self.clearance_rates = np.round(np.random.uniform(15, 20, size=self.lanes), self.rounding_precision)

        # Reset state history
        self.state = (self.distance, self.current_lane, self.risk_factor, *self.clearance_rates)
        
        obs = self._get_obs()
        info = {
                'time_reward': 0,
                'lane_change_reward': 0,
                'clearance_rate_reward': 0,
                'risk_accident_reward': 0
                }
        
        # Return the initial state
        return obs, info

    def _get_obs(self):
        """
        Constructs the current observation from the state.
        
        Returns:
        - flat_state (np.array): Flattened 1D observation of the state.
        """
        
        # Convert to a single observation array
        flat_state = np.array(self.state)
        
        self.logger.debug(f"Observation generated: {flat_state}")
        return flat_state.astype(np.float32)
        
    def _attempt_lane_change(self, action):
        """
        Attempts a lane change based on the specified action.

        Parameters:
            action (int): -1 for left, 1 for right.

        Returns:
            float: Penalty applied based on success/failure of lane change.
        """
        penalty = self.lane_change_penalty
        new_lane = self.current_lane + action
        if 1 <= new_lane <= self.lanes:
            
            current_clearance_rate = self.clearance_rates[self.current_lane - 1]
            new_clearance_rate = self.clearance_rates[self.current_lane - 1 + action]
            penalty += (new_clearance_rate - current_clearance_rate) * self.low_clearance_penalty_factor
            
            if random.random() < self.lane_change_probability:
                self.current_lane = new_lane
                self.logger.debug(f"Lane change succeeded to lane {self.current_lane}")
            else:
                self.logger.debug("Lane change failed.")
            
        else:
            # Penalize for out-of-bounds lane change
            penalty += self.wrong_lane_penalty
            self.logger.debug(f"Lane change action out of bounds. Current lane: {self.current_lane}, New Lane: {new_lane},  Action: {action}")
        
        return penalty

    def _update_clearance_rates(self):
        """Updates clearance rates for each lane based on conditions and events."""
        
        updated_rates = self.clearance_rates.copy()
        self.is_raining = random.random() < self.rain_probability
        self.logger.debug(f"Raining: {self.is_raining}")
        
        for i in range(self.lanes):
            # Apply lane-specific normal distributions if it's raining
            uncertainty = np.random.normal(
                self.rain_edge_lane_effect if (i == 0 or i == self.lanes - 1) \
                    else self.rain_center_lane_effect if (i == 1 or i == 3) else 0,
                0.1
            ) if self.is_raining else np.random.normal(0, 0.1)

            # Random Event: 5% chance of slowdown (20%-50%) and 5% chance of speedup (20%-40%)
            if random.random() < self.slowdown_probability:
                updated_rates[i] -= self.clearance_rates[i] * random.uniform(*self.slowdown_factor)
            elif random.random() < self.speed_up_probability:
                updated_rates[i] += self.clearance_rates[i] * random.uniform(*self.speed_up_factor)

            # Update based on adjacent lanes, with sgn function and uncertainty
            if i > 0:
                updated_rates[i] += self.clearance_rate_change_factor * np.sign(self.clearance_rates[i - 1] - self.clearance_rates[i])
            if i < self.lanes - 1:
                updated_rates[i] += self.clearance_rate_change_factor * np.sign(self.clearance_rates[i + 1] - self.clearance_rates[i])

            updated_rates[i] += uncertainty
            updated_rates[i] = round(min(max(updated_rates[i], self.clearance_rate_min), self.clearance_rate_max), 1)

        self.clearance_rates = updated_rates
    
    def _accident_check(self, action):
        """
        Checks for potential accidents based on the agent's action and current risk factor.

        Parameters:
            action (int): Action taken by the agent.

        Returns:
            tuple: (bool, float) indicating if truncated and the penalty applied.
        """
        penalty = 0
        
        # Update risk factor based on action taken
        if action != 0:
            self.risk_factor = min(self.risk_factor+self.lane_change_risk, 1)
        else:
            self.risk_factor = max(self.risk_factor-self.lane_change_risk, 0)
        
        # Penalize for high speed and reward for safe speed
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        if clearance_rate > self.speed_limit:
            penalty += self.speed_limit_penalty
            self.risk_factor = min(self.risk_factor+self.high_speed_risk, 1)
        elif clearance_rate <= self.safe_speed:
            self.risk_factor = max(self.risk_factor-self.high_speed_risk, 0)
        
        accident_chance = self.base_accident_probability * (self.risk_factor / self.accident_threshold)
        
        self.risk_factor = round(self.risk_factor, 3)
        
        if random.random() < accident_chance:
            truncated=True
            penalty += self.accident_penalty
        else:
            truncated=False
            if self.risk_factor > self.high_risk_threshold:
                penalty += self.high_risk_penalty
            
        return truncated, penalty

    def step(self, action):
        """
        Executes a step by taking an action, updating the environment, and calculating the reward.

        Parameters:
            action (int): The action taken by the agent (0 for left, 1 for stay, 2 for right).

        Returns:
            tuple: (next_state, reward, terminated, truncated, info) reflecting the result of the action.
        """
        # Map the discrete action to the original action space (-1, 0, 1)
        mapped_action = self.action_mapping[action]
        
        self._log(f"Taking action: {'left' if mapped_action==-1 else 'stay' if mapped_action==0 else 'right'}")
        self.render()
        
        reward = 0
        terminated = False
        truncated = False
        lane_change_reward = 0
        clearance_rate_reward = 0
        risk_accident_reward = 0
        
        # Time penalty
        reward += self.time_penalty
        
        # Handle lane change if not staying
        if mapped_action != 0:
            lane_change_reward = self._attempt_lane_change(mapped_action)
            reward += lane_change_reward
        else:
            current_clearance_rate = self.clearance_rates[self.current_lane - 1]
            left_clearance_rate = self.clearance_rates[self.current_lane - 2] if self.current_lane > 1 else 0
            right_clearance_rate = self.clearance_rates[self.current_lane] if self.current_lane < self.lanes else 0
            
            # Reward for staying in the lane with the current clearance rate
            clearance_rate_reward = (current_clearance_rate - max(left_clearance_rate, right_clearance_rate)) * self.low_clearance_penalty_factor
            reward += clearance_rate_reward

        # Update lane clearance rates based on neighboring lanes
        self._update_clearance_rates()

        # Compute the distance covered in the current lane
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        distance_covered = min(clearance_rate, self.distance)
        self.distance -= distance_covered
        self.distance = round(self.distance, self.rounding_precision)
        
        # Check for accidents and update risk factor
        truncated, risk_accident_reward = self._accident_check(mapped_action)
        reward += risk_accident_reward
        
        # Reward for distance covered
        reward += distance_covered * self.distance_reward_factor
        
        reward = round(reward, self.rounding_precision)
        self._log(f"Reward: {reward}")

        # Now, after all the updates for the current time step, append the state to history
        self.state = (self.distance, self.current_lane, self.risk_factor, *self.clearance_rates)

        # Check if the episode is done (if the destination is reached)
        if self.distance <= 0:
            terminated = True
            self.distance = 0
            self.render()
            self.logger.debug("Destination reached; episode terminated.")

        # Check if the episode is truncated (e.g., max time steps reached)
        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            truncated = True
            self.render()
            self.logger.debug("Maximum time steps reached; episode truncated.")
        
        # Return the next observation using _get_obs()
        next_state = self._get_obs()
        info = {
                'time_reward': self.time_penalty,
                'lane_change_reward': lane_change_reward,
                'clearance_rate_reward': clearance_rate_reward,
                'risk_accident_reward': risk_accident_reward
                }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Renders the environment state to the console.
        Args:
        - mode (str): The mode of rendering. 'human' for console output.
        """
        if mode == 'human':
            self._log(f"Time Step: {self.time_step}")
            self._log(f"Distance to Destination: {self.distance}")
            self._log(f"Current Lane: {self.current_lane}")
            self._log(f"Clearance Rates: {self.clearance_rates}")
            self._log(f"Is Raining: {self.is_raining}")
            self._log(f"Risk Factor: {self.risk_factor}")
            self._log('**************************************************')