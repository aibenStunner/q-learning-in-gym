from pathlib import Path
from typing import NamedTuple, Optional


class Params(NamedTuple):
    exp_name: str  # the name of this experiment
    seed: int  # define a seed so that we get reproducible results
    save_exp_folder: Path  # directory to save info from experiment
    learning_rate: float = 1e-4  # learning rate
    discount_factor: float = 0.99  # discount factor (gamma)
    initial_epsilon: float = 1  # initial exploration probability
    epsilon_decay: float = 0.10  # the fraction of `n_runs` it takes from initial_epsilon to get to final_epsilon
    final_epsilon: float = 0.01  # final exploration probability
    n_runs: int = 10000000  # number of runs for the experiment(s)
    total_episodes: Optional[int] = None  # total number of episodes
    is_slippery: Optional[
        bool
    ] = None  # if true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    action_size: Optional[int] = None  # number of possible actions
    state_size: Optional[int] = None  # number of possible states
    proba_frozen: Optional[float] = None  # probability that a tile is frozen
    env_id: Optional[str] = None  # the id of the environment
    num_envs: Optional[int] = 1  # the number of parallel game environments
    buffer_size: Optional[int] = 1000000  # the replay memory buffer size
    tau: Optional[float] = 1.0  # the target network update rate
    target_network_frequency: Optional[
        int
    ] = None  # the timesteps it takes to update the target network
    batch_size: Optional[int] = 32
    learning_starts: Optional[int] = 80000  # timestep to start learning
    train_frequency: Optional[int] = 4  # the frequency of training
    torch_deterministic: Optional[
        bool
    ] = True  # if toggled, `torhc.backends.cudnn.deterministic=False`
    cuda: Optional[bool] = True  # if toggled, cuda will be enabled by default
    capture_video: Optional[
        bool
    ] = False  # whether to capture videos of the agent performances (check out `videos` folder)
    save_model: Optional[
        bool
    ] = False  # whether to save model into `runs/{run_name}` folder
