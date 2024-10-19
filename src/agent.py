import numpy as np
import random

## Code a rule based agent ##
class RuleBasedAgent:
    def __init__(self):
        pass
    
    def choose_action(self, state):
        """
        Choose an action based on the current state, considering all lanes.
        Args:
        - state: A list of tuples, each containing (distance, current_lane, *clearance_rates)
        Returns:
        - action: -1 (move left), 0 (stay), or 1 (move right)
        """
        # We'll use the most recent state (the last in the list)
        current_state = state[-1]
        _, current_lane, *clearance_rates = current_state
        
        # Find the lane with the highest clearance rate
        # Adding 1 to align with 1-based indexing
        best_lane = np.argmax(clearance_rates) + 1
        
        if best_lane == current_lane:
            return 0  # Stay in the current lane
        else:
            return self.get_best_action(current_lane, best_lane, len(clearance_rates))
        
    def get_best_action(self, current_lane, target_lane, num_lanes):
        """
        Determine the best action to move towards the target lane.
        Args:
        - current_lane: The current lane (1-based)
        - target_lane: The target lane (1-based)
        - num_lanes: Total number of lanes
        Returns:
        - action: -1 (move left), 0 (stay), or 1 (move right)
        """
        if current_lane < target_lane:
            return 1  # Move right
        elif current_lane > target_lane:
            return -1  # Move left
        else:
            return 0  # Stay (should not happen in this context)


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
        - state: A list of tuples, each containing (distance, current_lane, *clearance_rates)
        
        Returns:
        - action: -1 (move left), 0 (stay), or 1 (move right)
        """
        current_state = state[-1]
        _, current_lane, *clearance_rates = current_state
        num_lanes = len(clearance_rates)
        
        # Decide whether to change lane based on the percentage
        if random.random() * 100 < self.change_lane_percentage:
            # Choose to change lane
            possible_actions = []
            if current_lane > 1:
                possible_actions.append(-1)  # Can move left
            if current_lane < num_lanes:
                possible_actions.append(1)   # Can move right
            
            if possible_actions:
                return random.choice(possible_actions)
        
        # If not changing lane or no valid lane change possible
        return 0  # Stay in the current lane
