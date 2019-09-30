import random
import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from torch import cat
from torch import clamp
from torch.distributions import normal
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

import gym
import roboschool

model_name = "SAC-RoboschoolHopper-v1"

num_iterations = 3000000
learning_rate = 0.0003
discount_rate = 0.99
replay_buffer_max_size = 1000000
target_smoothing_coefficient = 0.0005
target_update_interval = 1
num_gradient_steps = 1
num_env_steps = 1
reward_scale = 5
minibatch_size = 256

writer = SummaryWriter(log_dir="./runs/v0-1mil-iter-256-node-hidden-layers-buffer-1mil")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
cpu_device = torch.device("cpu")

# define actor network
class SACRoboschoolHopperActorNN(nn.Module):
    def __init__(self):
        super(SACRoboschoolHopperActorNN, self).__init__()
        self.fc1 = nn.Linear(15, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, 3)
        self.log_stdev = nn.Linear(256, 3)
        self.normal_dist = normal.Normal(0, 1)

    def forward(self, x_state):
        x_state = F.relu(self.fc1(x_state))
        x_state = F.relu(self.fc2(x_state))
        mean = self.mean(x_state)
        log_stdev = self.log_stdev(x_state)
        action = mean + self.normal_dist.sample(sample_shape=log_stdev.shape) * torch.exp(log_stdev)
        squashed_action = torch.tanh(action)
        action_dist = normal.Normal(mean, torch.exp(log_stdev))
        log_prob_squashed_a = action_dist.log_prob(action) - torch.sum(torch.log(clamp(tensor(1).view(squashed_action.shape) - squashed_action**2, min=1e-8)), dim=1)  # TODO check dims
        return action, log_prob_squashed_a


# define critic network
class SACRoboschoolHopperCriticNN(nn.Module):
    def __init__(self):
        super(SACRoboschoolHopperCriticNN, self).__init__()
        self.fc1 = nn.Linear(18, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x_state, x_action):
        x = cat((x_state, x_action), dim=1)  # concatenate inputs along 0th dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define soft state value network
class SACRoboschoolHopperStateValueNN(nn.Module):
    def __init__(self):
        super(SACRoboschoolHopperStateValueNN, self).__init__()
        self.fc1 = nn.Linear(15, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x_state):
        x = F.relu(self.fc1(x_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize parameter vectors ψ, ψ¯, θ, φ.
state_value_net = SACRoboschoolHopperStateValueNN().to(device)
state_value_target_net = SACRoboschoolHopperStateValueNN().to(device)
critic_net_1 = SACRoboschoolHopperCriticNN().to(device)
critic_net_2 = SACRoboschoolHopperCriticNN().to(device)
actor_net = SACRoboschoolHopperActorNN().to(device)

# make the state value target net parameters the same
state_value_target_net.load_state_dict(state_value_net.state_dict())

# initialize replay buffer D
replay_buffer = []

# initialize train and test environments
env = gym.make('RoboschoolHopper-v1')
curr_state = env.reset()
curr_state = tensor(curr_state).float().to(device)

test_env = gym.make('RoboschoolHopper-v1')
curr_test_state = test_env.reset()
greatest_avg_episode_rewards = -np.inf

# initialize optimizers for each network except target (parameters updated manually)
state_value_net_optimizer = optim.Adam(state_value_net.parameters(), lr=learning_rate)
critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=learning_rate)
critic_net_2_optimizer = optim.Adam(critic_net_2.parameters(), lr=learning_rate)
actor_net_optimizer = optim.Adam(actor_net.parameters(), lr=learning_rate)

# for each iteration do
for t in range(num_iterations):
    # for each environment step do
    # (in practice, at most one env step per gradient step)
    # at ∼ πφ(at|st)
    action, log_prob = actor_net(curr_state.view(1, -1,).float()).detach().to(cpu_device).numpy().squeeze()
    # action_np = action.detach().to(cpu_device).numpy().squeeze()

    # st+1 ∼ p(st+1|st, at)
    next_state, reward, done, _ = env.step(action)
    reward = reward * reward_scale

    # D ← D ∪ {(st, at, r(st, at), st+1)}
    replay_buffer.append((curr_state.view(1, -1, ), tensor(action).to(device).view(1, -1, ), log_prob.to(device).view(1, -1, ),
                          tensor(reward).float().to(device).view(1, 1, ), tensor(next_state).to(device).view(1, -1, ),
                          tensor(done).to(device).view(1, 1, )))
    if len(replay_buffer) > replay_buffer_max_size + 10:
        replay_buffer = replay_buffer[10:]

    # for each gradient step do
    for gradient_step in range(num_gradient_steps):
        # Sample mini-batch of N transitions (s, a, r, s') from D
        transitions_minibatch = random.choices(replay_buffer, k=minibatch_size)
        minibatch_states, minibatch_actions, minibatch_action_log_probs, minibatch_rewards, minibatch_next_states, minibatch_dones = [cat(mb, dim=0) for mb in zip(*transitions_minibatch)]
        minibatch_states = minibatch_states.float()

        # ψ ← ψ − λV ∇ˆψJV (ψ)
        state_value_net.zero_grad()
        # state_value_error = torch.mean(0.5 * torch.mean(state_value_net(minibatch_states) - torch.mean(torch.min(critic_net_1(minibatch_states, minibatch_actions),critic_net_2(minibatch_states, minibatch_actions)) - torch.log(actor_net(minibatch_states)))) ** 2)  # TODO fix?
        state_value_net_loss = torch.mean(0.5 * (state_value_net(minibatch_states) - (torch.min(critic_net_1(minibatch_states, minibatch_actions), critic_net_2(minibatch_states, minibatch_actions)) - torch.log(clamp(actor_net(minibatch_states), min=1e-8)))) ** 2)  # TODO fix?
        state_value_net_loss.backward()
        state_value_net_optimizer.step()
        writer.add_scalar('Loss/state_value_net', state_value_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

        # θi ← θi − λQ∇ˆθiJQ(θi) for i ∈ {1, 2}
        critic_net_1.zero_grad()
        critic_net_1_loss = torch.mean(0.5 * (critic_net_1(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate*state_value_target_net(minibatch_next_states)*(-minibatch_dones.float() + 1))) ** 2)
        critic_net_1_loss.backward()
        critic_net_1_optimizer.step()
        writer.add_scalar('Loss/critic_net_1', critic_net_1_loss.detach().to(cpu_device).numpy().squeeze(), t)

        critic_net_2.zero_grad()
        critic_net_2_loss = torch.mean(0.5 * (critic_net_2(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate * state_value_target_net(minibatch_next_states)*(-minibatch_dones.float() + 1))) ** 2)
        critic_net_2_loss.backward()
        critic_net_2_optimizer.step()
        writer.add_scalar('Loss/critic_net_2', critic_net_2_loss.detach().to(cpu_device).numpy().squeeze(), t)

        # φ ← φ − λπ∇ˆφJπ(φ)
        actor_net.zero_grad()
        minibatch_actions_new, minibatch_action_log_probs_new = actor_net(minibatch_states)
        actor_net_loss = torch.mean(minibatch_action_log_probs_new - torch.min(critic_net_1(minibatch_states, minibatch_actions_new), critic_net_2(minibatch_states, minibatch_actions_new)))  # TODO fix?
        actor_net_loss.backward()
        actor_net_optimizer.step()
        writer.add_scalar('Loss/actor_net', actor_net_loss.detach().to(cpu_device).numpy().squeeze(), t)
        # print(actor_net_loss.grad_fn())

        # ψ¯ ← τψ + (1 − τ )ψ¯
        for state_value_target_net_parameter, state_value_net_parameter in zip(state_value_target_net.parameters(), state_value_net.parameters()):
            state_value_target_net_parameter.data = target_smoothing_coefficient*state_value_net_parameter + (1 - target_smoothing_coefficient)*state_value_target_net_parameter
    # end for

    if t % (num_iterations // 1000) == 0 or t == num_iterations - 1:
        print("iter", t)
        torch.save(state_value_net.state_dict(), 'models/current/' + model_name + '-state_value_net.pkl')
        torch.save(state_value_target_net.state_dict(), 'models/current/' + model_name + '-state_value_target_net.pkl')
        torch.save(critic_net_1.state_dict(), 'models/current/' + model_name + '-critic_net_1.pkl')
        torch.save(critic_net_2.state_dict(), 'models/current/' + model_name + '-critic_net_2.pkl')
        torch.save(actor_net.state_dict(), 'models/current/' + model_name + '-actor_net.pkl')

    if not done:
        curr_state = tensor(next_state).float().to(device)
    else:
        curr_state = env.reset()
        curr_state = tensor(curr_state).float().to(device)

    if t % (num_iterations // 25) == 0 or t == num_iterations - 1:
        render = False
        num_eval_episodes = 10

        test_obs = test_env.reset()
        episode_rewards = []
        episode_reward = 0
        while len(episode_rewards) < num_eval_episodes:
            test_action = actor_net(tensor(test_obs).float().to(device)).detach().to(cpu_device).numpy().squeeze()
            test_obs, test_reward, test_done, _ = test_env.step(test_action)
            episode_reward += test_reward
            if test_done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                test_obs = test_env.reset()
            if render:
                test_env.render()

        avg_episode_rewards = np.mean(np.asarray(episode_rewards))
        writer.add_scalar('Reward/test', avg_episode_rewards, t)
        if avg_episode_rewards > greatest_avg_episode_rewards:
            torch.save(actor_net.state_dict(), 'models/current/best/best-' + model_name + '-actor_net.pkl')
# end for

render = True
num_eval_episodes = 10

obs = env.reset()
episode_rewards = []
episode_reward = 0
while len(episode_rewards) < num_eval_episodes:
    action = actor_net(tensor(obs).float().to(device)).detach().to(cpu_device).numpy().squeeze()
    obs, reward, done, _ = env.step(action)
    episode_reward += reward
    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0
        obs = env.reset()
    if render:
        env.render()

episode_rewards = np.asarray(episode_rewards)
episode_length_histogram = plt.hist(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Total Reward")
plt.ylabel("Frequency")
plt.savefig("episode_rewards_hist.png")
plt.savefig("models/current/episode_rewards_hist.png")
print("Mean total episode reward:", np.mean(episode_rewards))