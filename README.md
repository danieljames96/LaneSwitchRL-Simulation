# LaneSwitchRL-Simulation
RL-based simulation of lane-switching in traffic

## Environment

1. **Number of lanes**: `5`
2. **Initial Distance**: The distance to the destination will be initially set to `4000m`.
3. **Initial Lane**: The car starts on lane 1.
4. **Actions**: Switching between adjacent lanes.
5. **Clearance Rate**: No. of meters travelled per unit time.
6. We observe the relative clearance rate between lanes as they are dependent on each other.
7. Side lanes 1 and 5 only have one adjacent lane (2 and 4). If their relative clearance rate is lower than the neighbours then 0.2 will be added to the clearance rate of the side lanes and subtracted from the neighbours.
8. The converse is true for point 5.
9. Lanes 2, 3 and 4 are affected by both it's neighbours.
10. The changes in points 5, 6 and 7 are computed concurrently.
11. Everything outside consideration is modelled by uncertainty term `N(0,0.1)` as well as `random_event`.
12. The **environment clearance rate** is modelled by the equation:  
    `vt,j = vt-1,j + N(0,0.1) + random_event + 0.2 × sgn(vt-1,j-1 − vt-1,j) + 0.2 × sgn(vt-1,j+1 − vt-1,j)`  
    Note: If no adjacent lane exists, the sgn() function can be ignored.
13. **Random Event**: A 5% chance per time step for a lane to experience a slowdown of 20%-50% (uniform probability distribution) of vt-1,j, or 5% chance that the clearance rate is increased by 20-40% (also uniform probability distribution) of vt-1,j.
14. **Unit Time**: This refers to every 10 seconds.
15. Decision is based on historical records of `past three time steps` (t-2, t-1, t).
16. The `state` contains three tuples of seven values corresponding to his location dt away from the destination, his lane lt, and the clearance rate vt,j for each of the 5 lanes (j 𝜖 {1,2,3,4,5}).
17. **Reward Calculation**: 
- The reward is based on the distance covered per time step minus a fixed penalty (-10) for each time step. 
- An additional penalty of -5 is applied for lane-changing attempts (whether successful or not).
18. **Action Space**: 
- The agent can attempt to move left (-1), stay in the current lane (0), or move right (1).
- If the action is impossible (e.g., moving left in the leftmost lane), Ah Tan stays in the lane.
19. **Lane Change Success Rate**: When Ah Tan attempts to change lanes, there is only a 50% chance of success. Regardless of whether the lane change succeeds or fails, the penalty for attempting a lane change (-5) is applied. If the lane change fails, Ah Tan remains in the same lane for that time step.
20. **End Condition**: The episode ends when Ah Tan reaches his destination, i.e., when the distance to the destination (dt) becomes less than or equal to zero. At this point, the episode is considered done.

## Assumptions:

1. **Initial Clearance Rates**: We initialize clearance rates randomly between 15 and 20 for all lanes.

## Custom Environment Changes:

### **Fatigue**:
1. New Action for Rest
    - Expanded **action space** to `Discrete(4)` to include the **'rest' action**:
        - **0**: Move left
        - **1**: Stay
        - **2**: Move right
        - **3**: Rest (only available in lanes 1 or 5).

2. Fatigue Mechanism Added
    - Introduced a **fatigue counter** that increments each step unless the agent takes the **'rest' action**.
    - Fatigue penalty grows over time and is subtracted from the reward.

3. Fatigue Growth Options
    - Added linear, exponential, and quadratic growth options for fatigue penalties.

4. Integrated Fatigue into Rewards
    - Fatigue penalty is included in the reward function, increasing long-term decision complexity.

5. Observation Space Update
    - Added **fatigue counter** to the observation space, making it visible to the agent.

6. Reward Shaping for Rest
    - Taking the **'rest' action** resets the fatigue counter but incurs a small penalty.
    - Attempting to rest in invalid lanes results in a higher penalty.

7. Adjusted `step()` Method
    - Modified the `step()` method to handle the new action and fatigue penalties. The agent does not cover any distance in this time step.