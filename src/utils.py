import torch
import yaml


def load_config(config_file):
    """
    Loads a configuration file.

    Args:
       config_file (str): The file path of the configuration file.

    Returns:
       dict: A dictionary containing the configuration parameters.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def to_tensor(states, actions, old_log_probs, rewards, dones, state_values, device):
    """
    Converts lists of states, actions, and other data to PyTorch tensors.

    Args:
        states (ndarray): Array of states.
        actions (ndarray): Array of discrete and continuous action pairs.
        old_log_probs (ndarray): Array of log probabilities of action pairs.
        rewards (ndarray): Array of rewards.
        dones (ndarray): Array indicating if the state is terminal (done).
        state_values (ndarray): Array of state values.
        device (torch.device): The device to which the tensors will be moved.

    Returns:
        tuple: Tuple containing the tensors for states, actions (discrete and continuous),
               old log probabilities, rewards, dones, and state values.
    """
    states = torch.tensor(states, dtype=torch.float).detach().to(device)
    actions_d = (
        torch.tensor([a[0] for a in actions], dtype=torch.long).detach().to(device)
    )
    actions_c = (
        torch.tensor([a[1] for a in actions], dtype=torch.float).detach().to(device)
    )
    actions_c = actions_c.squeeze(-1)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).detach().to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)
    state_values = torch.tensor(state_values, dtype=torch.float).detach().to(device)
    return states, actions_d, actions_c, old_log_probs, rewards, dones, state_values
