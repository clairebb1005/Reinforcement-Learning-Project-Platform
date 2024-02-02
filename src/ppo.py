import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal, Categorical
from .network import ActorNetwork, CriticNetwork
from .utils import to_tensor


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.

    Args:
        config (dict): Configuration parameters.
        state_dim (int): Dimension of the state space.
        continuous_action_dim (int): Dimension of the continuous action space.
        discrete_action_dim (int): Dimension of the discrete action space.
        device (torch.device): Device on which to perform computations.
        initial_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        lr_stepsize (int): Learning rate scheduler step size.
        decrease_rate (float): Rate of decrease in learning rate.
        ppo_gamma (float): Discount factor for future rewards.
        eps_clip (float): Clipping parameter for PPO.
        alpha_d (float): Weight for discrete action loss component.
        alpha (float): Weight for continuous action loss component.
        k_epochs (int): Number of epochs for policy update.
        wandb: Weights & Biases logging tool (if used).
    """

    def __init__(
        self,
        config,
        state_dim,
        continuous_action_dim,
        discrete_action_dim,
        a_hidden_dim,
        c_hidden_dim,
        device,
        initial_lr,
        min_lr,
        lr_stepsize,
        decrease_rate,
        ppo_gamma,
        eps_clip,
        alpha_d,
        alpha,
        k_epochs,
        wandb,
    ):
        self.config = config
        self.device = device
        self.gamma = ppo_gamma
        self.eps_clip = eps_clip
        self.alpha_d = alpha_d
        self.alpha = alpha
        self.k_epochs = k_epochs
        self.wandb = wandb
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.step_size = lr_stepsize
        self.decrease_rate = decrease_rate

        self.actor = ActorNetwork(
            state_dim, continuous_action_dim, discrete_action_dim, a_hidden_dim
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr)
        self.actor_scheduler = self.schedule_lr(self.actor_optimizer)

        self.critic = CriticNetwork(state_dim, c_hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_lr)
        self.critic_scheduler = self.schedule_lr(self.critic_optimizer)

    def select_action(self, state):
        """
        Selects an action based on the current state using the policy network.

        Args:
            state (ndarray): The current state.

        Returns:
            tuple: A tuple containing the actions, log probability of the actions, and state value.
        """
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device) # [0.01895735 0.  0.22222222 0. 0.90909091 1., 0.95744681 0.  0. ]
        # mu: tensor([-0.4304,  0.3822,  0.0593], device='cuda:0', grad_fn=<AddBackward0>)
        # std: tensor([0.9032, 0.7602, 1.1433], device='cuda:0', grad_fn=<ExpBackward0>)
        # logits: tensor([-0.6137,  0.2388, -0.0249], device='cuda:0', grad_fn=<AddBackward0>)
        mu, std, logits = self.actor(state_tensor)
        # state_value: tensor([-0.1909], device='cuda:0', grad_fn=<AddBackward0>)
        state_value = self.critic(state_tensor)

        # creates a categorical distribution for discrete actions.
        # It represents a distribution over a fixed number of categories.
        disc_dist = Categorical(logits=logits) # Categorical(probs: torch.Size([3]), logits: torch.Size([3]))
        # samples a discrete action from the categorical distribution.
        action_d = disc_dist.sample() # tensor(1, device='cuda:0')
        # defines normal distribution for continuous actions.  three separate normal distributions.
        cont_dist = Normal(mu, std) # Normal(loc: torch.Size([3]), scale: torch.Size([3]))
        # samples a continuous action from the normal distribution.
        action_c = cont_dist.sample() # tensor([-0.8785, -0.1565, -0.3843]) represents possible actions the agent might take in the continuous space
        # Adjusts the shape of the continuous action tensor to fit the environment's requirements.
        action_c_ = action_c.unsqueeze(1) # tensor([[-0.8785], [-0.1565], [-0.3843]])
        # combines the discrete and continuous actions into a single action tuple.
        # .cpu(): You need to bring tensor back to the CPU if you want to convert it to a standard Python list
        # or perform other operations that are not supported on GPU tensors.
        # .item(): extracts the value of single-element tensor and converts it to a standard Python number
        action = (action_d.item(), action_c_.cpu().tolist())

        # compute the log probabilities of the sampled actions under the current policy.
        log_prob_d = disc_dist.log_prob(action_d) # tensor(-0.7860, device='cuda:0', grad_fn=<SqueezeBackward1>)

        # Sums the log probabilities over all dimensions except the batch dimension:
        # log_prob_c represents the log probabilities of the continuous actions.
        # If the continuous action space has multiple dimensions (parameters),
        # it will have a log probability for each dimension of each action in the batch.
        # The .sum(-1) sums these log probabilities across all dimensions of the action space
        # for each action in the batch, but not across the batch itself.
        # This summation is necessary because the probability of a multi-dimensional action
        # is the product of the probabilities of its individual dimensions, and in log space,
        # products are converted to sums.
        # The probability of all independent events occurring together is called the joint probability.

        # combines the log probabilities of all dimensions for each action,
        # which is required for computing the probability of multi-dimensional actions.
        log_prob_c = cont_dist.log_prob(action_c).sum(-1) # tensor(-2.9642, device='cuda:0', grad_fn=<SumBackward1>)
        # combines the log probabilities into a single tensor.
        log_prob = torch.stack([log_prob_d, log_prob_c])

        return action, log_prob, state_value

    def learn(self, states, actions, rewards, dones, old_log_probs, state_values):
        """
        Updates the policy based on the collected data.

        Args:
            states, actions, rewards, dones, old_log_probs, state_values: Collected data from rollouts.
        """
        (
            states,
            actions_d,
            actions_c,
            old_log_probs,
            rewards,
            dones,
            state_values,
        ) = to_tensor(
            states, actions, old_log_probs, rewards, dones, state_values, self.device
        )
        rewardstg = self.rewards_to_go(rewards, dones) # Tensor(200,)

        # Detaching these tensors gradients from flowing into these parts of the graph during the optimization process.
        # We don't want the gradient of the loss function to backpropagate into the reward or state value predictions.
        advantages = rewardstg.detach() - state_values.detach()  # Tensor(200,)
        #  Advantage: represents the benefit of taking a particular action over the average action for a given state.
        # If the advantage is positive, it means the action taken was better than what the critic expected; if negative, it was worse.
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )  # for stability

        for _ in range(self.k_epochs):
            self._update_policy(
                states, actions_d, actions_c, old_log_probs, advantages, rewardstg
            )

    def _update_policy(
        self, states, actions_d, actions_c, old_log_probs, advantages, rewardstg
    ):
        """
        Internal method to update the policy network.

        Args:
            states, actions_d, actions_c, old_log_probs, advantages, rewardstg: Data for policy update.
        """
        # Call forward function of actor
        mu, std, logits = self.actor(states)
        # defines a normal distribution for continuous actions.
        dist_c = Normal(mu, std)
        # creates a categorical distribution for discrete actions.
        dist_d = Categorical(logits=logits)

        new_log_prob_d = dist_d.log_prob(actions_d) # Tensor(200,), action_d:(Tensor(200,))

        # dist_d.probs: gets the probability of each action in the discrete action space

        # gather(1, actions_d.view(-1, 1)): collects the probabilities of the actions that were actually taken.

        # view(-1, 1): This reshapes actions_d into a two-dimensional tensor where the second dimension has size 1.
        # Essentially, it converts a flat list of action indices into a column vector.
        # The view() function reshapes a tensor without changing its data.
        # Here, actions_d.view(-1, 1) reshapes the actions_d tensor into a two-dimensional tensor with a single column,
        # where -1 tells PyTorch to infer the appropriate size for this dimension based on the original size of the tensor.

        # gather(1, ..): This gathers values along dimension 1 (the second dimension) of the distribution's probability tensor (dist_d.probs).
        # It selects the probabilities corresponding to the action indices

        # squeeze(): removes any extra dimensions of size 1.
        # it picks the probability corresponding to the action taken by the agent.
        prob_d = dist_d.probs.gather(1, actions_d.view(-1, 1)).squeeze()  # Tensor(200,): [0.4197, 0.3188, 0.4197,...]
        new_log_prob_c = dist_c.log_prob(actions_c) # tensor(200,3): [[-2.5136, -0.3742, -1.2915], [],...]
        new_log_prob_c = new_log_prob_c.sum(-1) # tensor(200,):[[-4.1793, -3.0019,...]]

        # selects the first element from every row in old_log_probs
        ratio_d = torch.exp(new_log_prob_d - old_log_probs[:, 0]) # tensor(200,)
        #  this selects the second element from every row
        ratio_c = torch.exp(new_log_prob_c - old_log_probs[:, 1]) # tensor(200,)

        surr1_d = ratio_d * advantages # # tensor(200,): 1.8778, 0.54166, ...
        surr2_d = (
            torch.clamp(ratio_d, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        )
        # The negative sign is used because optimization involves gradient descent,
        # and we want to maximize the rewards (minimize the negative of rewards).
        # .mean() computes the average loss over the batch.
        loss_d = -torch.min(surr1_d, surr2_d).mean()  # tensor(-9.5367e-09, device='cuda:0', grad_fn=<NegBackward0>)

        # *prob_d:
        # Parameterized action space: each action is a combination of a discrete decision and its continuous parameters.
        # (this multiplication ensures that the gradient accounts for how both the discrete and
        # continuous aspects of the action contribute to the overall policy.)
        # ** Important: actions are not just discrete choices but have associated parameters that need to be learned simultaneously.
        surr1_c = ratio_c * advantages * prob_d
        surr2_c = (
            torch.clamp(ratio_c, 1 - self.eps_clip, 1 + self.eps_clip)
            * advantages
            * prob_d
        )
        loss_c = -torch.min(surr1_c, surr2_c).mean()

        actor_loss = self.alpha_d * loss_d + self.alpha * loss_c

        # Before you calculate the gradients for a batch, you need to zero out the gradients from the previous batch.
        # This is because gradients in PyTorch accumulate by default.
        self.actor_optimizer.zero_grad()
        # This computes the gradient of the loss (actor_loss in your case) with respect to the model parameters.
        # Essentially, it's where backpropagation happens.
        actor_loss.backward()
        # This updates the model parameters using the gradients computed by .backward().
        # This step is what actually trains the model, adjusting weights to minimize the loss.
        self.actor_optimizer.step()

        new_state_values = self.critic(states)
        new_state_values = torch.squeeze(new_state_values)  # tensor(200,)
        # The critic aims to accurately predict the expected return from each state,
        # and the rewards-to-go represent a more immediate estimate of this return.
        critic_loss = nn.MSELoss()(new_state_values, rewardstg)  # tensor(1.3932, device='cuda:0', grad_fn=<MseLossBackward0>)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._log_wandb(
            {
                "loss_d": loss_d,
                "loss_c": loss_c,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "actor_lr": self.actor_scheduler.get_lr()[0],
                # returns a list of learning rates for all parameter groups in the optimizer.
                # [0] accesses the first element of this list,
                # which is the current learning rate for the first (or only) parameter group.
                "critic_lr": self.critic_scheduler.get_lr()[0],
            }
        )

    def save(self, checkpoint_path_save):
        """
        Saves the model_final state.

        Args:
            checkpoint_path_save (str): Path to save the checkpoint.
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path_save)

    def load(self, checkpoint_path_load):
        """
        Loads the model_final state from a checkpoint.

        Args:
            checkpoint_path_load (str): Path to load the checkpoint from.
        """
        try:
            checkpoint = torch.load(checkpoint_path_load)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            self._reinitialize_schedulers()
        except FileNotFoundError:
            print(f"Checkpoint file not found: {checkpoint_path_load}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def _reinitialize_schedulers(self):
        """
        Reinitializes the learning rate schedulers.
        """
        self.actor_scheduler = self.schedule_lr(self.actor_optimizer)
        self.critic_scheduler = self.schedule_lr(self.critic_optimizer)

    def _log_wandb(self, metrics):
        """
        Log metrics to wandb, if it is enabled and available.

        Args:
            metrics (dict): Metrics to log.
        """
        if self.config.get("log_result", True) and self.wandb is not None:
            self.wandb.log(metrics)

    def schedule_lr(self, optimizer):
        """
        Creates a learning rate scheduler for the given optimizer.

        This scheduler adjusts the learning rate based on a predefined schedule. The learning rate
        decreases by a factor of (1 - decrease_rate) every lr_stepsize steps, down to a minimum
        of min_lr. This approach helps in fine-tuning the model by slowly reducing the learning
        rate during training, preventing large updates that could destabilize the learning process.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.

        Returns:
            torch.optim.lr_scheduler.LambdaLR: A LambdaLR learning rate scheduler.
        """
        # It decreases the learning rate by a factor of (1 - self.decrease_rate) every self.step_size steps,
        # but not below self.min_lr.
        lambda_lr = lambda step: max(
            (1 - self.decrease_rate) ** (step // self.step_size),
            self.min_lr / self.initial_lr,
        )
        #  Creates a LambdaLR scheduler that uses lambda_lr to adjust the learning rate of the given optimizer.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        return scheduler

    def rewards_to_go(self, rewards, dones):
        """
        Calculates the 'rewards-to-go' for each time step in an episode.

        'Rewards-to-go' is a method used in policy gradient algorithms where the return for each time step
        is calculated as the sum of discounted future rewards from that time step to the end of the episode.

        Args:
            rewards (Tensor): List of rewards for each time step in the episode.
            dones (Tensor): List indicating whether the episode ended at each time step.

        Returns:
            torch.Tensor: A tensor containing the calculated 'rewards-to-go' for each time step.
        """
        rewardstg = []
        discounted_reward = 0

        #  Iterates through the rewards and done flags in reverse order (starting from the end of an episode).
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # Updates the discounted_reward by adding the current reward and the discounted future reward.
            # (1.0 - done): zeros out future rewards at the end of an episode.
            discounted_reward = reward + (self.gamma * discounted_reward * (1.0 - done))
            # Inserts the calculated discounted_reward at the beginning of the rewardstg list,
            # reversing the order back to normal. insert at position 0 every time
            rewardstg.insert(0, discounted_reward)

        # Converts rewardstg to a tensor
        rewardstgs = torch.tensor(rewardstg, dtype=torch.float32).to(self.device)
        # This helps in reducing variance and stabilizing training.
        rewardstgs = (rewardstgs - rewardstgs.mean()) / (rewardstgs.std() + 1e-7)
        return rewardstgs
