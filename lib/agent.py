import numpy as np
import gymnasium as gym

class QLearningAgent:
    def __init__(
        self, 
        learning_rate: float,
        initial_epsilon: float, 
        epsilon_decay: float, 
        final_epsilon: float, 
        discount_factor: float, 
        state_size: int, 
        action_size: int,
        rng_seed: int
    ):
        """ Initialize Reinforcement Learning agent with empty table
        of state-action values (qtable), a learning rate and an epsilon

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor (gamma) or computing the Q-value of state
            state_size: The number of possible observations (states) in the observation space
            action_size: The number of possible actions in given observation (state)
            rng_seed: The seed to initialize Random Generator
        """
        self.state_size = state_size
        self.action_size = action_size

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.reset_qtable()
        self.rng = np.random.default_rng(rng_seed)

    def get_action(self, state: int, action_space: gym.Space) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probablility epsilon to ensure exploration
        """
        explore_eploit_tradeoff = self.rng.uniform(0, 1)

        # explore the environment
        if explore_eploit_tradeoff < self.epsilon:
            action = action_space.sample()
        
        # exploit the environment (act greedily by taking the largest Q-value for state)
        # with probability (1 - epsilon)
        else:
            # break ties randomly
            # chose random action if actions are same for state
            # else act greedily
            if np.all(self.qtable[state, :]) == self.qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(self.qtable[state, :])
        
        return action

    def update(
        self, 
        state: int, 
        action: int, 
        reward: float,
        terminated: bool, 
        new_state: int
    ):
        """ Updates the Q-value of an action

        Q(s,a) := Q(s,a) + lr [R(s,a) + discount_factor * max Q(s',a') - Q(s,a)]
        """
        next_q_value = (not terminated) * np.max(self.qtable[new_state, :])
        temporal_difference = (
            reward + self.discount_factor * next_q_value - self.qtable[state, action]
        )
        self.qtable[state, action] = self.qtable[state, action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))
