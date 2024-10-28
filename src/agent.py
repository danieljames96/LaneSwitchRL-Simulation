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


class RuleBasedAgent2:
    def __init__(self):
        # The base threshold for determining if the current lane is too slow relative to the average
        self.base_slow_threshold = 0.8
        # The base threshold for determining if an adjacent lane is significantly faster
        self.base_fast_threshold = 1.2

    def choose_action(self, state):
        """
        Chooses an action (-1: move left, 0: stay, 1: move right) based on the current state of the environment.

        Args:
        - state: A list of tuples representing the last three time steps. Each tuple contains
                 the distance to the destination, the current lane, and clearance rates of 5 lanes.

        Returns:
        - action (int): -1 for left, 0 for stay, and 1 for right.
        """

        # Extract the most recent state (the last time step in the list)
        current_state = state[-1]
        # Get the distance to the destination
        distance = current_state[0]
        # Get the current lane number
        current_lane = current_state[1]
        # Get the clearance rates for all lanes
        clearance_rates = current_state[2:]

        # Total number of lanes in the environment
        total_lanes = len(clearance_rates)
        # Calculate the average clearance rate across all lanes
        avg_clearance = np.mean(clearance_rates)

        # Dynamically calculate thresholds based on the remaining distance
        slow_threshold = self.calculate_slow_threshold(distance)
        fast_threshold = self.calculate_fast_threshold(distance)

        # Decision-making process: Check if the current lane is too slow
        if clearance_rates[current_lane - 1] < avg_clearance * slow_threshold:
            # If the current lane is slower than the average by a certain factor (slow_threshold),
            # consider switching to a faster adjacent lane.

            left_lane_rate = clearance_rates[current_lane - 2] if current_lane > 1 else float('-inf')
            right_lane_rate = clearance_rates[current_lane] if current_lane < total_lanes else float('-inf')
            current_lane_rate = clearance_rates[current_lane - 1]
            
            if left_lane_rate > current_lane_rate and left_lane_rate >= right_lane_rate:
                return -1  # Move to the left lane
            elif right_lane_rate > current_lane_rate and right_lane_rate > left_lane_rate:
                return 1   # Move to the right lane


        # If the current lane is not too slow, check if any adjacent lane is significantly faster
        left_significantly_faster = (current_lane > 1 and 
                                    clearance_rates[current_lane - 2] > clearance_rates[current_lane - 1] * fast_threshold)
        right_significantly_faster = (current_lane < total_lanes and 
                                    clearance_rates[current_lane] > clearance_rates[current_lane - 1] * fast_threshold)

        if left_significantly_faster and (not right_significantly_faster or 
                                        clearance_rates[current_lane - 2] >= clearance_rates[current_lane]):
            return -1  # Move left if it's significantly faster (and better or equal to right)
        elif right_significantly_faster:
            return 1   # Move right if it's significantly faster (and better than left)

        # If no better lanes are found, stay in the current lane
        return 0


    def calculate_slow_threshold(self, distance):
        """
        Calculate a threshold to determine if the current lane is too slow.
        The threshold increases as the distance decreases (encouraging faster lanes as the agent nears the destination).
        
        Args:
        - distance: Current distance to the destination.

        Returns:
        - slow_threshold: The calculated slow threshold.
        """
        # Increase the slow threshold as the distance decreases, making the agent more likely to avoid slow lanes
        return self.base_slow_threshold + (0.2 * distance / 4000)  # Assuming max distance is 4000 

    def calculate_fast_threshold(self, distance):
        """
        Calculate a threshold to determine if an adjacent lane is significantly faster.
        The threshold decreases as the distance decreases, making the agent more willing to switch to faster lanes.
        
        Args:
        - distance: Current distance to the destination.

        Returns:
        - fast_threshold: The calculated fast threshold.
        """
        # Decrease the fast threshold as the distance decreases, making the agent more aggressive in switching to faster lanes
        return self.base_fast_threshold - (0.2 * distance / 4000)  # Assuming max distance is 4000 
