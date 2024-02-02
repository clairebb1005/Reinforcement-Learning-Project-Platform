import numpy as np


class RolloutBuffer:
    """
    A buffer for storing trajectories observed during training.

    Attributes:
    states (list): List to store the states.
    actions (list): List to store the actions.
    rewards (list): List to store the rewards.
    next_states (list): List to store the next states.
    dones (list): List to store the done flags (episode termination flags).
    log_probs (list): List to store the log probabilities of the actions.
    state_values (list): List to store the state values.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.state_values = []

    def push(self, state, action, reward, next_state, done, log_prob, state_value):
        """
        Store an experience in the buffer.

        Args:
            state (list of float): The observed state.
            action (tuple): The action pair (discrete, continuous) taken.
            reward (float): The reward received after taking the action.
            next_state (list of float): The next state.
            done (bool): A boolean indicating if the episode ended after the action.
            log_prob (list of float): The log probability of actions under the policy.
            state_value (float): The value of the state as estimated by the critic network.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)

    def get(self):
        """
        Retrieve all stored experiences.

        Returns:
            A tuple containing numpy arrays of states, actions, rewards, next_states,
            dones, log_probs, and state_values.
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.state_values),
        )

    def clear(self):
        """
        Clear all stored data in the buffer.
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.state_values.clear()
