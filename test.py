import numpy as np
import random
import argparse
import torch
import matplotlib.pyplot as plt

from gym_platform.envs.platform_env import PlatformEnv
from torch.distributions import Normal, Categorical
from src.network import ActorNetwork


def is_success(env, done):
    """
    Determine if an episode is successful by checking the x-axis position of the player.

    Args:
        env: The environment instance.
        done: The done flag from the environment.

    Returns:
        bool: True if the episode is successful, False otherwise.
    """
    player_reached_end = (
        env.player.position[0] >= env._right_bound() - env.player.size[0]
    )
    return done and player_reached_end


def test(model_path, num_episodes, render=True, plot=True):
    """
    Tests the trained PPO agent in the specified environment.

    Args:
        model_path (str): The path to the saved model_final file.
        num_episodes (int): Number of episodes to run the agent for testing.
        render (bool): Whether to render the environment for each step.
        plot (bool): Whether to plot the total rewards and lengths of episodes.
    """
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    env = PlatformEnv()
    env.seed(seed)

    state_dim = env.observation_space[0].shape[0]
    discrete_actions_dim = env.action_space[0].n
    continuous_params_dim = len(env.action_space[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path)
    actor = ActorNetwork(
        state_dim, discrete_actions_dim, continuous_params_dim, 128
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    # Evaluation loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        success = False
        episode_reward = 0
        episode_length = 0

        while not done:
            if render:
                env.render()
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)

            with torch.no_grad():
                mu, std, logits = actor(state_tensor)
                action_d = Categorical(logits=logits).sample()
                action_c = Normal(mu, std).sample()
                action_c_ = action_c.unsqueeze(1)
                action = (action_d.item(), action_c_.cpu().tolist())

            (next_state, _), reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            episode_length += 1

        if is_success(env, done):
            success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode: {episode + 1}, Total Reward: {episode_reward}, Length: {episode_length}"
        )

    average_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = success_count / num_episodes

    print(
        f"Average Reward: {average_reward}, Reward Std: {std_reward}, Success Rate: {success_rate}"
    )

    if plot:
        plot_results(episode_rewards, episode_lengths)

    env.close()


def plot_results(rewards, lengths):
    """
    Plots the rewards and lengths of episodes.

    Args:
        rewards (list): List of rewards per episode.
        lengths (list): List of lengths per episode.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Length")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained PPO agent in a Platform environment."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/PPO_PlatformEnv.pth",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to test the agent.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during testing."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot episode rewards and lengths after testing.",
    )
    args = parser.parse_args()

    test(args.model_path, args.num_episodes, args.render, args.plot)
