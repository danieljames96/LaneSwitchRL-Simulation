import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import logging

# Set up the logger
logger = logging.getLogger(__name__)
log_file='./logs/task2/custom_traffic_env.log'
if not logger.hasHandlers():
    # handler = logging.StreamHandler()
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class CustomTrafficEnvironment(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self, lanes=5, initial_distance=4000, max_time_steps = 10000, seed=None, logging_enabled=False):
        """
        Args:
        - lanes (int): Number of lanes (default is 5).
        - initial_distance (int): The distance from destination.
        """
        
        super(CustomTrafficEnvironment, self).__init__()
        self.seed_value = seed
        
        self.lanes = lanes
        self.max_time_steps = max_time_steps
        self.initial_distance = initial_distance
        self.logger = logger
        self.logging_enabled = logging_enabled
        
        self.state_history = []
        self.rounding_precision = 1
        self.risk_factor = 0
        
        self.clearance_rate_min = 5
        self.clearance_rate_max = 30
        self.clearance_rate_change_factor = 0.2

        self.rain_probability = 0.2
        self.rain_edge_lane_effect = -0.3
        self.rain_center_lane_effect = -0.2
        self.is_raining = False

        self.accident_threshold = 0.9
        self.accident_probability = 0.001
        self.speed_limit = 25
        self.safe_speed = 10
        self.accident_penalty = -40
        self.high_risk_threshold = 0.8 * self.accident_threshold  # 80% of the accident threshold
        self.high_risk_penalty = -4 # High-risk penalty
        self.lane_change_risk = 0.05
        self.high_speed_risk = 0.1

        self.lane_change_penalty = -0.5
        self.time_penalty = -1
        self.wrong_lane_penalty = -5
        self.low_clearance_penalty_factor = -0.5

        self.lane_change_probability = 0.6
        self.slowdown_probability = 0.05
        self.speed_up_probability = 0.05

        self.slowdown_factor = (0.2, 0.5)
        self.speed_up_factor = (0.2, 0.4)
        self.distance_reward_factor = 0.2
        
        # Define action space (0: left, 1: stay, 2: right)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: (distance, current lane, clearance rates)
        low_obs = np.array([0, 1] + [self.clearance_rate_min] * self.lanes, dtype=np.float32)
        high_obs = np.array([self.initial_distance] + [lanes] + [self.clearance_rate_max] * self.lanes, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Define action mapping
        self.action_mapping = {0: -1, 1: 0, 2: 1}
        
        if seed is not None:
            self.set_seed(seed)
        
        self.reset()
        self.logger.debug("Environment initialized.")
    
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
        
        # Initialize the state
        self.distance = self.initial_distance
        if options:
            self.current_lane = options.get('starting_lane', 1)
        else:
            self.current_lane = 1
        self.is_raining = False
        # Initialize clearance rates randomly between 15 and 20 for all lanes
        self.clearance_rates = np.round(np.random.uniform(15, 20, size=self.lanes), self.rounding_precision)

        # Reset state history
        self.state_history = [(self.distance, self.current_lane, self.risk_factor, *self.clearance_rates)]
        self.time_step = 0
        self.risk_factor = 0
        
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
        
    def _attempt_lane_change(self, action):
        """
        Attempts to change lane based on the action.
        Args:
        - action (int): -1 for left, 1 for right.
        Returns:
        - penalty (float): The penalty to be applied to the reward.
        """
        penalty = self.lane_change_penalty
        new_lane = self.current_lane + action
        if 1 <= new_lane <= self.lanes:
            
            current_clearance_rate = self.clearance_rates[self.current_lane - 1]
            new_clearance_rate = self.clearance_rates[self.current_lane - 1 + action]
            
            # Penalize for lane change if the clearance rate is lower than the current lane
            if new_clearance_rate < current_clearance_rate and current_clearance_rate < self.speed_limit:
                penalty += (current_clearance_rate - new_clearance_rate) * self.low_clearance_penalty_factor
            
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
                    uncertainty = np.random.normal(self.rain_edge_lane_effect, 0.1)
                elif i == 1 or i == 3:  # Lanes 2 and 4
                    uncertainty = np.random.normal(self.rain_center_lane_effect, 0.1)
                else:  # Lane 3
                    uncertainty = np.random.normal(0, 0.1)
            else:
                uncertainty = np.random.normal(0, 0.1)  # Default non-rainy condition

            # Random Event: 5% chance of slowdown (20%-50%) and 5% chance of speedup (20%-40%)
            random_event = random.random()
            # 5% chance of slowdown
            if random_event < self.slowdown_probability:
                updated_rates[i] -= self.clearance_rates[i] * random.uniform(*self.slowdown_factor)
                self.logger.debug(f"Clearance rate slowed in lane {i+1}")
            # 5% chance of speedup
            elif random_event >= self.slowdown_probability and random_event < self.slowdown_probability + self.speed_up_probability:
                updated_rates[i] += self.clearance_rates[i] * random.uniform(*self.speed_up_factor) 
                self.logger.debug(f"Clearance rate increased in lane {i+1}")

            # Update based on adjacent lanes, with sgn function and uncertainty
            if i > 0:
                updated_rates[i] += self.clearance_rate_change_factor * np.sign(self.clearance_rates[i - 1] - self.clearance_rates[i])
            if i < self.lanes - 1:
                updated_rates[i] += self.clearance_rate_change_factor * np.sign(self.clearance_rates[i + 1] - self.clearance_rates[i])

            updated_rates[i] += uncertainty
            updated_rates[i] = round(updated_rates[i], self.rounding_precision)
            updated_rates[i] = max(updated_rates[i], self.clearance_rate_min)
            updated_rates[i] = min(updated_rates[i], self.clearance_rate_max)
            
            # self.logger.debug(f"Lane {i+1} clearance rate updated from {self.clearance_rates[i]} to {updated_rates[i]}")

        self.clearance_rates = updated_rates
        self.logger.debug("Clearance rates updated.")
    
    def _accident_check(self, action):
        # Check for accidents based on risk factor and accident probability
        penalty = 0
        
        if action != 0:
            self.risk_factor = min(self.risk_factor+self.lane_change_risk, 1)
        else:
            self.risk_factor = max(self.risk_factor-self.lane_change_risk, 0)
        
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        if clearance_rate > self.speed_limit:
            self.risk_factor = min(self.risk_factor+self.high_speed_risk, 1)
        elif clearance_rate <= self.safe_speed:
            self.risk_factor = max(self.risk_factor-self.high_speed_risk, 1)
        
        accident_chance = self.accident_probability * (self.risk_factor / self.accident_threshold)
        
        self.risk_factor = round(self.risk_factor, 3)
        
        if random.random() < accident_chance:
            self.logger.debug("Accident occurred.")
            truncated=True
            penalty += self.accident_penalty
        else:
            truncated=False
            if self.risk_factor > self.high_risk_threshold:
                penalty += self.high_risk_penalty
            
        return truncated, penalty

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
        # Map the discrete action to the original action space (-1, 0, 1)
        mapped_action = self.action_mapping[action]
        
        self._log(f"Taking action: {'left' if mapped_action==-1 else 'stay' if mapped_action==0 else 'right'}")
        self.render()
        
        reward = 0
        terminated = False
        truncated = False
        
        # Time penalty
        reward += self.time_penalty
        
        # Handle lane change if not staying
        if mapped_action != 0:
            reward += self._attempt_lane_change(mapped_action)
        else:
            current_clearance_rate = self.clearance_rates[self.current_lane - 1]
            left_clearance_rate = self.clearance_rates[self.current_lane - 2] if self.current_lane > 1 else 0
            right_clearance_rate = self.clearance_rates[self.current_lane] if self.current_lane < self.lanes else 0
            
            # Penalize for staying in lane if the clearance rate is lower than adjacent lanes
            if ((current_clearance_rate < left_clearance_rate and left_clearance_rate < self.speed_limit) \
                or (current_clearance_rate < right_clearance_rate and right_clearance_rate < self.speed_limit)):
                reward += (current_clearance_rate - max(left_clearance_rate, right_clearance_rate)) * self.low_clearance_penalty_factor

        # Update lane clearance rates based on neighboring lanes
        self._update_clearance_rates()

        # Compute the distance covered in the current lane
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        distance_covered = clearance_rate
        self.distance -= distance_covered
        self.distance = round(self.distance, self.rounding_precision)
        
        truncated, accident_penalty = self._accident_check(mapped_action)
        
        reward += accident_penalty
        
        # Reward for distance covered
        reward += distance_covered * self.distance_reward_factor
        
        reward = round(reward, self.rounding_precision)

        # Now, after all the updates for the current time step, append the state to history
        self.state_history.append((self.distance, self.current_lane, self.risk_factor, *self.clearance_rates))

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
            self._log(f"Time Step: {self.time_step}")
            self._log(f"Distance to Destination: {self.distance}")
            self._log(f"Current Lane: {self.current_lane}")
            self._log(f"Clearance Rates: {self.clearance_rates}")
            self._log(f"Is Raining: {self.is_raining}")
            self._log(f"Risk Factor: {self.risk_factor}")
            self._log('**************************************************')