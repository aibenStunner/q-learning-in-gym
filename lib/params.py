from pathlib import Path
from typing import NamedTuple

class Params(NamedTuple):
    total_episodes: int  # total number of episodes
    learning_rate: float  # learning rate
    discount_factor: float  # discount factor (gamma)
    initial_epsilon: float  # initial exploration probability
    epsilon_decay: float # rate at which exploration should be reduced over time
    final_epsilon: float # final exploration probability
    map_size: int  # number of tiles of one side of the squared environment
    seed: int  # define a seed so that we get reproducible results
    is_slippery: bool  # if true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # number of runs
    action_size: int  # number of possible actions
    state_size: int  # number of possible states
    proba_frozen: float  # probability that a tile is frozen
    savefig_folder: Path  # directory where plots are saved