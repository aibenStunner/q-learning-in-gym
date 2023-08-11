import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_rewards_and_steps(
    rewards_df: pd.DataFrame, 
    steps_df: pd.DataFrame,
    savefig_folder: Path,
    save_fig: bool,
):
    """plot the steps and rewards from dataframes"""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()

    if save_fig:
        img_title = "frozenlake-v1_steps_and_rewards.png"
        fig.savefig(savefig_folder / img_title, bbox_inches="tight")
        
    plt.show()