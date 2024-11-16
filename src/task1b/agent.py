import numpy as np
import random

class RandomAgent:
    def __init__(self, change_lane_percentage=100):
        """
        Initialize the RandomAgent.
        
        Args:
        - change_lane_percentage (float): Percentage chance of changing lanes (0-100).
        """
        self.change_lane_percentage = max(0, min(100, change_lane_percentage))
    
    def choose_action(self, state):
        """
        Choose an action based on the current state and the change lane percentage.
        
        Args:
        - state: Current state observation
        
        Returns:
        - action: 0 (move left), 1 (stay), or 2 (move right)
        """
        current_lane = int(state[15])
        num_lanes = 5  # Based on environment specification
        
        # Decide whether to change lane based on the percentage
        if random.random() * 100 < self.change_lane_percentage:
            # Choose to change lane
            possible_actions = []
            if current_lane > 1:
                possible_actions.append(0)  # Can move left
            if current_lane < num_lanes:
                possible_actions.append(2)  # Can move right
            
            if possible_actions:
                return random.choice(possible_actions)
        
        # If not changing lane or no valid lane change possible
        return 1  # Stay in the current lane


import numpy as np

class RuleBasedAgent:
    def __init__(self):
        # The base threshold for determining if the current lane is too slow relative to the average
        self.base_slow_threshold = 0.8
        # The base threshold for determining if an adjacent lane is significantly faster
        self.base_fast_threshold = 1.2
        # Initial distance (used for threshold calculations)
        self.initial_distance = 4000

    def choose_action(self, state):
        """
        Chooses an action (0: move left, 1: stay, 2: move right) based on the current state of the environment.

        Args:
        - state: numpy array containing the environment state with format:
          [distance_t-2, lane_t-2, clearance_rates_t-2..., 
           distance_t-1, lane_t-1, clearance_rates_t-1...,
           distance_t, lane_t, clearance_rates_t...]

        Returns:
        - action (int): 0 for left, 1 for stay, and 2 for right.
        """
        # Extract the most recent state (last 7 values)
        current_state = state[-7:]
        
        # Get the distance to destination, current lane, and clearance rates
        distance = current_state[0]
        current_lane = int(current_state[1])  # Ensure lane is an integer
        clearance_rates = current_state[2:]  # Last 5 values are clearance rates
        
        # Total number of lanes in the environment
        total_lanes = len(clearance_rates)
        # Calculate the average clearance rate across all lanes
        avg_clearance = np.mean(clearance_rates)

        # Dynamically calculate thresholds based on the remaining distance
        slow_threshold = self.calculate_slow_threshold(distance)
        fast_threshold = self.calculate_fast_threshold(distance)

        # Decision-making process: Check if the current lane is too slow
        if clearance_rates[current_lane - 1] < avg_clearance * slow_threshold:
            # Get adjacent lane rates with safety checks
            left_lane_rate = clearance_rates[current_lane - 2] if current_lane > 1 else float('-inf')
            right_lane_rate = clearance_rates[current_lane] if current_lane < total_lanes else float('-inf')
            current_lane_rate = clearance_rates[current_lane - 1]
            
            if left_lane_rate > current_lane_rate and left_lane_rate >= right_lane_rate:
                return 0  # Move to the left lane (action 0)
            elif right_lane_rate > current_lane_rate and right_lane_rate > left_lane_rate:
                return 2  # Move to the right lane (action 2)

        # Check if any adjacent lane is significantly faster
        left_significantly_faster = (
            current_lane > 1 and 
            clearance_rates[current_lane - 2] > clearance_rates[current_lane - 1] * fast_threshold
        )
        right_significantly_faster = (
            current_lane < total_lanes and 
            clearance_rates[current_lane] > clearance_rates[current_lane - 1] * fast_threshold
        )

        if left_significantly_faster and (not right_significantly_faster or 
                                        clearance_rates[current_lane - 2] >= clearance_rates[current_lane]):
            return 0  # Move left if it's significantly faster (action 0)
        elif right_significantly_faster:
            return 2  # Move right if it's significantly faster (action 2)

        # If no better lanes are found, stay in the current lane
        return 1  # Stay in current lane (action 1)

    def calculate_slow_threshold(self, distance):
        """
        Calculate a threshold to determine if the current lane is too slow.
        The threshold increases as the distance decreases.
        
        Args:
        - distance: Current distance to the destination.

        Returns:
        - slow_threshold: The calculated slow threshold.
        """
        return self.base_slow_threshold + (0.2 * distance / self.initial_distance)

    def calculate_fast_threshold(self, distance):
        """
        Calculate a threshold to determine if an adjacent lane is significantly faster.
        The threshold decreases as the distance decreases.
        
        Args:
        - distance: Current distance to the destination.

        Returns:
        - fast_threshold: The calculated fast threshold.
        """
        return self.base_fast_threshold - (0.2 * distance / self.initial_distance)

def test_rule_based_agent():
    """
    Test function to verify the RuleBasedAgent's behavior with various scenarios
    """
    agent = RuleBasedAgent()
    
    # Create test states with different scenarios
    test_states = [
        # Normal case: middle lane, varying speeds
        np.array([
            1000, 3, 15, 16, 14, 18, 15,  # t-2
            900, 3, 15, 16, 14, 18, 15,   # t-1
            800, 3, 15, 16, 14, 18, 15    # t
        ]),
        # Edge case: leftmost lane
        np.array([
            1000, 1, 14, 18, 15, 16, 15,  # t-2
            900, 1, 14, 18, 15, 16, 15,   # t-1
            800, 1, 14, 18, 15, 16, 15    # t
        ]),
        # Edge case: rightmost lane
        np.array([
            1000, 5, 15, 16, 15, 18, 14,  # t-2
            900, 5, 15, 16, 15, 18, 14,   # t-1
            800, 5, 15, 16, 15, 18, 14    # t
        ]),
        # Case with significantly faster adjacent lane
        np.array([
            1000, 3, 15, 16, 14, 20, 15,  # t-2
            900, 3, 15, 16, 14, 20, 15,   # t-1
            800, 3, 15, 16, 14, 20, 15    # t
        ])
    ]
    
    print("Testing RuleBasedAgent decisions:")
    for i, state in enumerate(test_states):
        action = agent.choose_action(state)
        current_lane = int(state[-6])
        clearance_rates = state[-5:]
        
        print(f"\nTest case {i + 1}:")
        print(f"Distance: {state[-7]}")
        print(f"Current lane: {current_lane}")
        print(f"Clearance rates: {clearance_rates}")
        print(f"Chosen action: {action} ({'Left' if action == 0 else 'Stay' if action == 1 else 'Right'})")
        print(f"Slow threshold: {agent.calculate_slow_threshold(state[-7]):.2f}")
        print(f"Fast threshold: {agent.calculate_fast_threshold(state[-7]):.2f}")
