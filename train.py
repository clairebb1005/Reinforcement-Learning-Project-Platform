import torch
import numpy as np
import wandb
import os
import random

from gym_platform.envs.platform_env import PlatformEnv
from src.ppo import PPOAgent
from src.rollout import RolloutBuffer
from src.utils import load_config


def train(config, env, state_dim, continuous_action_dim, discrete_action_dim):
    """
    Trains the PPO agent in the specified environment.

    Runs the training loop for the PPO agent, initializing the agent, setting
    up the environment, and collecting experiences for policy updates. Handles
    logging of training metrics, model checkpointing, and agent updates.

    Args:
        config (dict): Configuration dict with parameter settings.
        env (gym.Env): Environment instance for agent training.
        state_dim (int): Dimensionality of the state space.
        continuous_action_dim (int):  Number of continuous actions.
        discrete_action_dim (int): Number of discrete actions.
    """
    buffer = RolloutBuffer()
    device = torch.device(
        "cuda" if config["cuda"] and torch.cuda.is_available() else "cpu"
    )

    if config["log_result"]:
        wandb.init(
            project=config["project_name"],
            name=config["exp_name"],
            notes=config["exp_notes"],
        )

    agent = PPOAgent(
        config,
        state_dim,
        continuous_action_dim,
        discrete_action_dim,
        config["a_hidden_dim"],
        config["c_hidden_dim"],
        device,
        config["initial_lr"],
        config["min_lr"],
        config["lr_stepsize"],
        config["decrease_rate"],
        config["ppo_gamma"],
        config["eps_clip"],
        config["alpha_d"],
        config["alpha"],
        config["K_epochs"],
        wandb,
    )

    # Load model checkpoint if configured
    if config["load_model"]:
        ckpt_load_path = config["ckpt_load_path"]
        if os.path.exists(ckpt_load_path):
            print(f"Loading checkpoint from {ckpt_load_path}")
            agent.load(ckpt_load_path)

    ckpt_save_path = None
    if config["save_model"]:
        os.makedirs(config["ckpt_save_path"], exist_ok=True)
        ckpt_save_path = os.path.join(config["ckpt_save_path"], "PPO_PlatformEnv.pth")
        print(f"Save checkpoint path: {ckpt_save_path}")

    # Start the training loop
    running_reward, running_episodes, i_episode, lengths = 0, 0, 0, 0
    time_step = 0

    while time_step <= config["max_training_timesteps"]:
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for t in range(config["max_episode_steps"]):
            action, log_prob, state_value = agent.select_action(state)
            (next_state, _), reward, done, _ = env.step(action)

            # Store the transition in the buffer
            buffer.push(
                state.tolist(),
                action,
                reward,
                next_state.tolist(),
                done,
                log_prob.cpu().tolist(),
                state_value.item(),
            )
            state = next_state
            time_step += 1
            episode_reward += reward

            # Update the agent at defined intervals
            if time_step % config["update_timestep"] == 0:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    old_log_probs,
                    state_values,
                ) = buffer.get()
                agent.learn(
                    states, actions, rewards, dones, old_log_probs, state_values
                )

                if config["save_model"]:
                    agent.save(ckpt_save_path)

                agent.actor_scheduler.step(time_step)
                agent.critic_scheduler.step(time_step)
                buffer.clear()

            if done:
                lengths += t
                break

        running_reward += episode_reward
        running_episodes += 1
        i_episode += 1

        # Log performance metrics
        if config["log_result"] and i_episode % config["log_reward_freq"] == 0:
            log_avg_reward = running_reward / running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            wandb.log({"log_avg_reward": log_avg_reward}, step=time_step)
            running_reward = 0
            running_episodes = 0

        if config["log_result"] and i_episode % config["log_len_freq"] == 0:
            wandb.log(
                {"avg_episode_length": lengths / config["log_len_freq"]}, step=time_step
            )
            lengths = 0


if __name__ == "__main__":
    config = load_config("config.yaml")

    seed = config["seed"]
    # Sets the seed for Python's built-in random module, ensuring reproducibility for random number generation.
    random.seed(seed)
    # Sets the seed for NumPy's random number generator, similarly ensuring reproducibility.
    np.random.seed(seed)

    # Creates an instance of the PlatformEnv environment.
    env = PlatformEnv()
    # Sets the seed for the environment's random number generator,
    # ensuring consistent random elements within the environment across runs.
    env.seed(seed)

    state_dim = env.observation_space[0].shape[0]
    continuous_params_dim = len(env.action_space[1])
    discrete_action_dim = env.action_space[0].n

    train(config, env, state_dim, continuous_params_dim, discrete_action_dim)
