import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer):
    """
    Initializes the weights of a linear layer using Xavier uniform initialization.

    Args:
        layer (nn.Module): an instance of nn.Linear.

    Returns:
        None: The layer is modified in-place with initialized weights.
    """
    if isinstance(layer, nn.Linear):
        #  Proper initialization prevents the gradients from becoming too small (vanishing gradient)
        #  or too large (exploding gradient), which can hinder the learning process.
        #  Different initialization methods, like Xavier or He initialization,
        #  are designed to maintain the gradients at an appropriate scale,
        #  leading to more efficient training and faster convergence.
        nn.init.xavier_uniform_(layer.weight)


class ActorNetwork(nn.Module):
    """
        Actor Network for PPO.

    Args:
        state_dim (int): Dimension of the state space.
        continuous_action_dim (int): Dimension of the continuous action space.
        discrete_action_dim (int): Dimension of the discrete action space.
        hidden_dim (int): Dimension of the hidden layers.
    """

    def __init__(
        self, state_dim, continuous_action_dim, discrete_action_dim, hidden_dim=128
    ):
        super(ActorNetwork, self).__init__()
        # Common layers
        self.fc = nn.Linear(state_dim, hidden_dim)

        # Continuous actor layers
        self.mu = nn.Linear(hidden_dim, continuous_action_dim)
        self.log_std = nn.Linear(hidden_dim, continuous_action_dim)

        # Discrete actor layer
        # "logits" refers to the raw, unnormalized outputs of the last linear layer of the network.
        # Logits are unnormalized log probabilities associated with each action.
        # These logits are typically transformed into probabilities using a softmax function for multi-class classification
        # or a sigmoid function for binary classification.
        self.logits = nn.Linear(hidden_dim, discrete_action_dim)

        # Initialize the weights
        # apply(): is a method of nn.Module that applies a function to all the modules (layers) of the network.
        # ensure that all layers start with weights in a range that is suitable for learning.
        self.apply(layer_init)

    def forward(self, state):
        """
        Forward pass for the Actor Network.

        Args:
            state (Tensor): The input state.

        Returns:
            Tuple of Tensors: Continuous means, std deviations, and discrete logits.
        """
        #  relu: introduces non-linearity into the model, which is essential for the network to learn complex patterns.
        x = F.relu(self.fc(state))

        # Continuous actions
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.exp()

        # Discrete actions
        logits = self.logits(x)

        return mu, std, logits


class CriticNetwork(nn.Module):
    """
    Critic Network for PPO.

    Args:
        state_dim (int): Dimension of the state space.
        hidden_dim (int): Dimension of the hidden layers.
    """

    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.apply(layer_init)

    def forward(self, state):
        """
        Forward pass for the Critic Network.

        Args:
            state (Tensor): The input state.

        Returns:
            Tensor: The value of the state.
        """
        x = F.relu(self.fc(state))
        value = self.value(x)

        return value
