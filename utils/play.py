import gymnasium as gym


def play(env: gym.Env, num_episodes: int, agent=None):
    for episode in range(num_episodes):
        print(f'Playing episode {episode}')

        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            if (agent):
                # select an action A_{t} using S_{t} as input for the agent
                action = agent.get_action(
                    state=state,
                    action_space=env.action_space
                )
            else:
                action = env.action_space.sample()

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(
                action.item())

            # update if the environment is done
            done = terminated or truncated

    env.close()
    return
