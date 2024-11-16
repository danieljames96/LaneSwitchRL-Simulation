# LaneSwitchRL-Simulation
RL-based simulation of lane-switching in traffic

## How to run the code
Note: All code dependencies are available in `pyproject.toml`

### Task 1
#### Environment Implementation
    1. Env is saved at `./src/task1a/environment_gym.py`

#### Rule-Based Agent
    1. Go to `./notebooks/task1Rulebased.ipynb`
    2. Run the cells in the notebook
#### RL-Agents
##### TD(λ)
    1. Go to `./notebooks/task1TDAgent.ipynb`
    2. Run the cells in the notebook
    3. Best models are currently available under `./models/`. Update the paths in the notebook to load these files. 
##### DQN
    1. Go to `./notebooks/task1DQN.ipynb`
    2. Run the cells in the notebook
    3. Best models are currently available under `./models/`. Update the paths in the notebook to load these files.
##### PPO
    1. Go to `./notebooks/task1PPO.ipynb`
    2. Run the cells in the notebook
    3. Best models are currently available under `./models/`. Update the paths in the notebook to load these files.

### Task 2
    1. Go to `./notebooks/task2.ipynb`
    2. Run the cells in the notebook
    3. The logs will be generated under `./notebooks/logs/task2/`
    4. Best models are currently available under `./models/task2/`. Update the paths in the notebook to load these files.
