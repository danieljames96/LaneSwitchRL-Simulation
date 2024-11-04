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

### State Index Mapping

The flattened state array includes the last three time steps, each containing information about the distance, current lane, risk factor, and clearance rates.

| **Index** | **Element**              | **Time Step**  |
|-----------|--------------------------|----------------|
| 0         | Distance                 | t-2            |
| 1         | Current Lane             | t-2            |
| 2         | Risk Factor              | t-2            |
| 3         | Clearance Rate (Lane 1)  | t-2            |
| 4         | Clearance Rate (Lane 2)  | t-2            |
| 5         | Clearance Rate (Lane 3)  | t-2            |
| 6         | Clearance Rate (Lane 4)  | t-2            |
| 7         | Clearance Rate (Lane 5)  | t-2            |
| 8         | Distance                 | t-1            |
| 9         | Current Lane             | t-1            |
| 10        | Risk Factor              | t-1            |
| 11        | Clearance Rate (Lane 1)  | t-1            |
| 12        | Clearance Rate (Lane 2)  | t-1            |
| 13        | Clearance Rate (Lane 3)  | t-1            |
| 14        | Clearance Rate (Lane 4)  | t-1            |
| 15        | Clearance Rate (Lane 5)  | t-1            |
| 16        | Distance                 | t              |
| 17        | Current Lane             | t              |
| 18        | Risk Factor              | t              |
| 19        | Clearance Rate (Lane 1)  | t              |
| 20        | Clearance Rate (Lane 2)  | t              |
| 21        | Clearance Rate (Lane 3)  | t              |
| 22        | Clearance Rate (Lane 4)  | t              |
| 23        | Clearance Rate (Lane 5)  | t              |

**Total Length**: 24 elements

### **Min Clearance Rate**
The minimum clearance rate is set to a specific number > 0 to ensure that the traffic is moving and does not stagnate to zero.

### **Rainfall Condition**

1. Introduction of Rainfall Condition
    The environment has been updated to simulate the effects of rainfall on lane clearance rates. During each time step, there is a random probability that it will rain, affecting lane dynamics differently:

    - **Lanes 1 and 5**: Experience higher water accumulation, with a clearance rate adjustment following `N(-0.2, 0.1)`.
    - **Lanes 2 and 4**: Experience moderate water accumulation, with a clearance rate adjustment following `N(-0.1, 0.1)`.
    - **Lane 3**: Has minimal water accumulation, with a clearance rate adjustment following `N(0, 0.1)`.

2. Implementation Details
    - A new parameter, **`rain_probability`**, controls the likelihood of rainfall at each time step (default is 0.1).
    - If it rains, lane-specific normal distributions are applied to update clearance rates.
    - The agent must adapt to changing conditions caused by rainfall, making decisions based on the observed clearance rates.

3. Design Considerations
    - The **`is_raining`** flag is not included in the state to maintain complexity and ensure the agent learns from indirect observations.
    - This change aims to introduce variability and challenge, simulating realistic weather-induced impacts on traffic.

### Risk Factor
