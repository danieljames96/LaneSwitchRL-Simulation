import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class TrafficEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, lanes=5, initial_distance=4000):
        super(TrafficEnvironment, self).__init__()
        
        self.lanes = lanes
        self.initial_distance = initial_distance
        self.rounding_precision = 1
        self.clearance_rate_min = 0
        self.max_time_steps = 10000

        # Define action space (0: left, 1: stay, 2: right)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space to match the new state dimension (16 features)
        obs_dim = 16

        # Define low and high bounds for each dimension of the observation
        low = np.array([0,                # distance_index (0-3)
                        0,                # current_lane (0-4)
                        0, 0, 0, 0, 0,    # clearance_rates for each lane (5 lanes)
                        0, 0, 0, 0, 0,    # clearance_rate_index for each lane (5 lanes)
                        -4000,            # left_clearance_diff
                        -4000,            # right_clearance_diff
                        0,                # fastest_lane (0-4)
                        -4000],           # fastest_lane_diff
                       dtype=np.float32)
        
        high = np.array([3,               # distance_index (0-3)
                         4,               # current_lane (0-4)
                         4000, 4000, 4000, 4000, 4000,  # clearance_rates for each lane
                         6, 6, 6, 6, 6,    # clearance_rate_index for each lane (0-6)
                         4000,             # left_clearance_diff
                         4000,             # right_clearance_diff
                         4,                # fastest_lane (0-4)
                         4000],            # fastest_lane_diff
                        dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Define action mapping
        self.action_mapping = {0: -1, 1: 0, 2: 1}
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize the state
        self.distance = self.initial_distance
        self.current_lane = 1  # Start from the first lane
        self.clearance_rates = np.round(np.random.uniform(15, 20, size=self.lanes), self.rounding_precision)

        # Reset state history to track last 3 timesteps
        self.state_history = [(self.distance, self.current_lane, *self.clearance_rates)]
        self.time_step = 0

        # Return the initial state
        obs = self._get_obs()
        return obs, {}

    def group_distance(self, distance):
        """Groups the distance into index categories."""
        if distance < 1000:
            return 0
        elif distance < 2000:
            return 1
        elif distance < 3000:
            return 2
        else:
            return 3

    def group_clearance_rate(self, rate):
        """Groups the clearance rate into index categories."""
        if rate < 5:
            return 0
        elif rate < 10:
            return 1
        elif rate < 15:
            return 2
        elif rate < 20:
            return 3
        elif rate < 100:
            return 4
        elif rate < 500:
            return 5
        else:
            return 6

    def _get_obs(self):
        """Constructs the current observation based on the last 3 timesteps."""
        # Use last 3 timesteps if available
        recent_history = self.state_history[-3:]

        # Compute the average clearance rate for the driver across the last 3 timesteps
        avg_clearance_rate_driver = np.mean([
            state[2 + (state[1] - 1)]  # Extract clearance rate for the driver's lane at each timestep
            for state in recent_history
        ])
        
        # Compute the average clearance rate per lane over the last 3 timesteps
        avg_clearance_rates = np.mean([state[2:] for state in recent_history], axis=0)
        fastest_lane = np.argmax(avg_clearance_rates)
        fastest_lane_rate = avg_clearance_rates[fastest_lane]
        
        # Group current distance and clearance rate
        distance_index = self.group_distance(self.distance)
        clearance_rate_index = [self.group_clearance_rate(rate) for rate in self.clearance_rates]
        
        # Calculate left and right clearance rate differences
        clearance_rate = self.clearance_rates[self.current_lane - 1]
        # Calculate left clearance difference if there is a lane to the left
        if self.current_lane > 1:
            left_clearance_diff = self.clearance_rates[self.current_lane - 2] - clearance_rate
        else:
            left_clearance_diff = 0  # No left lane
        # Calculate right clearance difference if there is a lane to the right
        if self.current_lane < self.lanes:
            right_clearance_diff = self.clearance_rates[self.current_lane] - clearance_rate
        else:
            right_clearance_diff = 0  # No right lane
        
        # Calculate the difference with the fastest lane
        fastest_lane_diff = fastest_lane_rate - avg_clearance_rate_driver

        # Observation as a flat state array
        obs = np.array([
            distance_index,
            self.current_lane,
            *self.clearance_rates,          # Clearance rates for each lane (5 values)
            *clearance_rate_index,          # Clearance rate index for each lane (5 values)
            left_clearance_diff,
            right_clearance_diff,
            fastest_lane,
            fastest_lane_diff
        ], dtype=np.float32)
        
        return obs

    def _attempt_lane_change(self, action):
        penalty = -5
        new_lane = self.current_lane + action
        if 1 <= new_lane <= self.lanes:
            if random.random() < 0.5:
                self.current_lane = new_lane
        return penalty

    def _update_clearance_rates(self):
        updated_rates = self.clearance_rates.copy()
        for i in range(self.lanes):
            uncertainty = np.random.normal(0, 0.1)
            random_event = random.random()
            if random_event < 0.05:
                updated_rates[i] -= self.clearance_rates[i] * random.uniform(0.2, 0.5)
            elif random_event < 0.1:
                updated_rates[i] += self.clearance_rates[i] * random.uniform(0.2, 0.4)
            if i > 0:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i - 1] - self.clearance_rates[i])
            if i < self.lanes - 1:
                updated_rates[i] += 0.2 * np.sign(self.clearance_rates[i + 1] - self.clearance_rates[i])
            updated_rates[i] += uncertainty
            updated_rates[i] = round(updated_rates[i], self.rounding_precision)
            updated_rates[i] = max(updated_rates[i], self.clearance_rate_min)
        self.clearance_rates = updated_rates

    def step(self, action):
        mapped_action = self.action_mapping[action]
        reward = 0
        terminated = False
        truncated = False

        if mapped_action != 0:
            reward += self._attempt_lane_change(mapped_action)

        self._update_clearance_rates()

        clearance_rate = self.clearance_rates[self.current_lane - 1]
        distance_covered = min(clearance_rate, self.distance)
        self.distance -= distance_covered
        self.distance = round(self.distance, self.rounding_precision)
        reward += distance_covered - 10

        distance_index = self.group_distance(self.distance)
        self.state_history.append((self.distance, self.current_lane, *self.clearance_rates))

        if self.distance <= 0:
            terminated = True
            self.distance = 0

        self.time_step += 1
        if self.time_step >= self.max_time_steps:
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Time Step: {self.time_step}")
            print(f"Distance to Destination: {self.distance}")
            print(f"Current Lane: {self.current_lane}")
            print(f"Clearance Rates: {self.clearance_rates}")
