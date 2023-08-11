import pandas as pd
import numpy as np
from .params import Params

def post_process(
    episodes: np.typing.NDArray, 
    params: Params, 
    rewards: np.typing.NDArray, 
    steps: np.typing.NDArray, 
    map_size: int
):
    """convert the results of the simulation into dataframes"""
    results = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    results["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    results["map_size"] = np.repeat(f"{map_size}x{map_size}", results.shape[0])

    _steps = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    _steps["map_size"] = np.repeat(f"{map_size}x{map_size}", _steps.shape[0])
    return results, _steps