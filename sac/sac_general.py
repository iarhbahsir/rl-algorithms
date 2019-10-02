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

import mujoco_py
import gym

environment_name = 'Hopper-v2'

model_name = "SAC-{}".format(environment_name)

parameters = {
    'num_iterations': {
        'LunarLanderContinuous-v2': 1_000_000,
        'Hopper-v2': 1_000_000
    },
    'learning_rate': {
        'LunarLanderContinuous-v2': 0.0003,
        'Hopper-v2': 0.0003
    },
    'discount_rate': {
        'LunarLanderContinuous-v2': 0.99,
        'Hopper-v2': 0.99
    },
    'replay_buffer_max_size': {
        'LunarLanderContinuous-v2': 100_000,
        'Hopper-v2': 1_000_000
    },
    'target_smoothing_coefficient': {
        'LunarLanderContinuous-v2': 0.005,
        'Hopper-v2': 0.005
    },
    'target_update_interval': {
        'LunarLanderContinuous-v2': 1,
        'Hopper-v2': 1
    },
    'num_gradient_steps': {
        'LunarLanderContinuous-v2': 1,
        'Hopper-v2': 1
    },
    'num_env_steps': {
        'LunarLanderContinuous-v2': 1,
        'Hopper-v2': 1
    },
    'reward_scale': {
        'LunarLanderContinuous-v2': 5,
        'Hopper-v2': 5
    },
    'minibatch_size': {
        'LunarLanderContinuous-v2': 256,
        'Hopper-v2': 256
    },
    'state_dim': {
        'LunarLanderContinuous-v2': 8,
        'Hopper-v2': 11
    },
    'action_dim': {
        'LunarLanderContinuous-v2': 2,
        'Hopper-v2': 3
    }
}

num_iterations = parameters['num_iterations'][environment_name]
learning_rate = parameters['learning_rate'][environment_name]
discount_rate = parameters['discount_rate'][environment_name]
replay_buffer_max_size = parameters['replay_buffer_max_size'][environment_name]
target_smoothing_coefficient = parameters['target_smoothing_coefficient'][environment_name]
target_update_interval = parameters['target_update_interval'][environment_name]
num_gradient_steps = parameters['num_gradient_steps'][environment_name]
num_env_steps = parameters['num_env_steps'][environment_name]
temperature = 1/parameters['reward_scale'][environment_name]
minibatch_size = parameters['minibatch_size'][environment_name]
STATE_DIM = parameters['state_dim'][environment_name]
ACTION_DIM = parameters['action_dim'][environment_name]

writer = SummaryWriter(log_dir="./runs/v2-1mil-iter-256-node-hidden-layers-buffer-1mil")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


# define actor network
class SACActorNN(nn.Module):
    def __init__(self):
        super(SACActorNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, ACTION_DIM)
        self.log_stdev = nn.Linear(256, ACTION_DIM)
        self.normal_dist = normal.Normal(0, 1)

    def forward(self, x_state):
        # print(x_state.shape)
        x_state = F.relu(self.fc1(x_state))
        x_state = F.relu(self.fc2(x_state))
        mean = self.mean(x_state)
        log_stdev = self.log_stdev(x_state)
        unsquashed_action = mean + self.normal_dist.sample(sample_shape=log_stdev.shape).to(device) * torch.exp(log_stdev).to(device)
        squashed_action = torch.tanh(unsquashed_action)
        action_dist = normal.Normal(mean, torch.exp(log_stdev))
        log_prob_squashed_a = action_dist.log_prob(unsquashed_action).to(device) - torch.sum(torch.log(clamp(torch.ones(squashed_action.shape).to(device) - squashed_action**2, min=1e-8)), dim=1).view(-1, 1).repeat(1, ACTION_DIM)
        return squashed_action, log_prob_squashed_a


# define critic network
class SACCriticNN(nn.Module):
    def __init__(self):
        super(SACCriticNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, ACTION_DIM)

    def forward(self, x_state, x_action):
        x = cat((x_state, x_action), dim=1)  # concatenate inputs along 0th dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define soft state value network
class SACStateValueNN(nn.Module):
    def __init__(self):
        super(SACStateValueNN, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x_state):
        x = F.relu(self.fc1(x_state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize parameter vectors ψ, ψ¯, θ, φ.
state_value_net = SACStateValueNN().to(device)
state_value_target_net = SACStateValueNN().to(device)
critic_net_1 = SACCriticNN().to(device)
critic_net_2 = SACCriticNN().to(device)
actor_net = SACActorNN().to(device)

# make the state value target net parameters the same
state_value_target_net.load_state_dict(state_value_net.state_dict())

# initialize replay buffer D
replay_buffer = []

# initialize train and test environments
env = gym.make(environment_name)
curr_state = env.reset()
curr_state = tensor(curr_state).float().to(device)

test_env = gym.make(environment_name)
curr_test_state = test_env.reset()
greatest_avg_episode_rewards = -np.inf

# initialize optimizers for each network except target (parameters updated manually)
state_value_net_optimizer = optim.Adam(state_value_net.parameters(), lr=learning_rate)
critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=learning_rate)
critic_net_2_optimizer = optim.Adam(critic_net_2.parameters(), lr=learning_rate)
actor_net_optimizer = optim.Adam(actor_net.parameters(), lr=learning_rate)

print("{} MB".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

# for each iteration do
for t in range(num_iterations):
    # for each environment step do
    # (in practice, at most one env step per gradient step)
    # at ∼ πφ(at|st)
    if t % 1000 == 0:
        print("{} MB start of loop".format(torch.cuda.memory_allocated(device=device)*(1e-6)))
    action, log_prob = actor_net(curr_state.view(1, -1,).float().to(device))
    action = action.detach().to(cpu_device).numpy().squeeze()

    # st+1 ∼ p(st+1|st, at)
    next_state, reward, done, _ = env.step(action)

    # D ← D ∪ {(st, at, r(st, at), st+1)}
    replay_buffer.append((curr_state.to(cpu_device).view(1, -1, ).float(), tensor(action).to(cpu_device).float().view(1, -1, ), log_prob.float().to(cpu_device).view(1, -1, ),
                          tensor(reward).float().to(cpu_device).view(1, 1, ), tensor(next_state).float().to(cpu_device).view(1, -1, ),
                          tensor(done).to(cpu_device).view(1, 1, ).float()))
    if len(replay_buffer) > replay_buffer_max_size + 10:
        replay_buffer = replay_buffer[10:]

    if t % 1000 == 0:
        print("{} MB after adding to replay buffer".format(torch.cuda.memory_allocated(device=device)*(1e-6)))

    # for each gradient step do
    for gradient_step in range(num_gradient_steps):
        # Sample mini-batch of N transitions (s, a, r, s') from D
        transitions_minibatch = random.choices(replay_buffer, k=minibatch_size)
        minibatch_states, minibatch_actions, minibatch_action_log_probs, minibatch_rewards, minibatch_next_states, minibatch_dones = [cat(mb, dim=0).to(device) for mb in zip(*transitions_minibatch)]
        minibatch_states = minibatch_states.float()

        if t % 1000 == 0:
            print("{} MB after minibatch sampled".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

        # ψ ← ψ − λV ∇ˆψJV (ψ)
        state_value_net.zero_grad()
        state_value_net_loss = torch.mean(0.5 * (state_value_net(minibatch_states) - (torch.min(critic_net_1(minibatch_states, minibatch_actions), critic_net_2(minibatch_states, minibatch_actions)) - torch.mul(temperature, actor_net(minibatch_states)[1]))) ** 2)
        state_value_net_loss.backward()
        state_value_net_optimizer.step()
        writer.add_scalar('Loss/state_value_net', state_value_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

        if t % 1000 == 0:
            print("{} MB after state value net fit".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

        # θi ← θi − λQ∇ˆθiJQ(θi) for i ∈ {1, 2}
        critic_net_1.zero_grad()
        critic_net_1_loss = torch.mean(0.5 * (critic_net_1(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate*state_value_target_net(minibatch_next_states)*(-minibatch_dones + 1))) ** 2)
        critic_net_1_loss.backward()
        critic_net_1_optimizer.step()
        writer.add_scalar('Loss/critic_net_1', critic_net_1_loss.detach().to(cpu_device).numpy().squeeze(), t)

        if t % 1000 == 0:
            print("{} MB after critic net 1 fit".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

        critic_net_2.zero_grad()
        critic_net_2_loss = torch.mean(0.5 * (critic_net_2(minibatch_states, minibatch_actions) - (minibatch_rewards + discount_rate*state_value_target_net(minibatch_next_states)*(-minibatch_dones + 1))) ** 2)
        critic_net_2_loss.backward()
        critic_net_2_optimizer.step()
        writer.add_scalar('Loss/critic_net_2', critic_net_2_loss.detach().to(cpu_device).numpy().squeeze(), t)

        if t % 1000 == 0:
            print("{} MB after critic net 2 fit".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

        # φ ← φ − λπ∇ˆφJπ(φ)
        actor_net.zero_grad()
        minibatch_actions_new, minibatch_action_log_probs_new = actor_net(minibatch_states)
        actor_net_loss = torch.mean(torch.mul(minibatch_action_log_probs_new, temperature) - torch.min(critic_net_1(minibatch_states, minibatch_actions_new), critic_net_2(minibatch_states, minibatch_actions_new)))
        actor_net_loss.backward()
        actor_net_optimizer.step()
        writer.add_scalar('Loss/actor_net', actor_net_loss.detach().to(cpu_device).numpy().squeeze(), t)

        if t % 1000 == 0:
            print("{} MB after actor net fit".format(torch.cuda.memory_allocated(device=device) * (1e-6)))

        # ψ¯ ← τψ + (1 − τ )ψ¯
        for state_value_target_net_parameter, state_value_net_parameter in zip(state_value_target_net.parameters(), state_value_net.parameters()):
            state_value_target_net_parameter.data = target_smoothing_coefficient*state_value_net_parameter + (1 - target_smoothing_coefficient)*state_value_target_net_parameter
        # end for

    if t % 1000 == 0 or t == num_iterations - 1:
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

    if t % (num_iterations // 50) == 0 or t == num_iterations - 1:
        render = False
        num_eval_episodes = 10

        test_obs = test_env.reset()
        episode_rewards = []
        episode_reward = 0
        while len(episode_rewards) < num_eval_episodes:
            test_action, test_action_log_prob = actor_net(tensor(test_obs).view(1, -1, ).float().to(device))
            test_action = test_action.detach().to(cpu_device).numpy().squeeze()
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