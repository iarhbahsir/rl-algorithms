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

model_name = "TD3-Pendulum-v0"

num_iterations = 10000
replay_memory_max_size = 1000
sigma = 0.2
minibatch_size = 64
discount_rate = 0.99
steps_until_policy_update = 2
target_update_ratio = 0.0005
epsilon_limit = 0.5
min_action = -2
max_action = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO convert to gpu

# define actor network
class TD3CartpoleActorNN(nn.Module):
    def __init__(self):
        super(TD3CartpoleActorNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    # def load_state_dict(self, state_dict, strict=True):
    #     super(TD3CartpoleActorNN, self).load_state_dict(state_dict)
    #
    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return super(TD3CartpoleActorNN, self).state_dict()

# define critic network
class TD3CartpoleCriticNN(nn.Module):
    def __init__(self):
        super(TD3CartpoleCriticNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, state, action):
        print("critic forward shape", state.size(), action.size())
        x = cat((state, action), dim=0)  # concatenate inputs along 0th dimension
        print("critic x shape", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def load_state_dict(self, state_dict, strict=True):
    #     super(TD3CartpoleCriticNN, self).load_state_dict(state_dict)
    #
    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return super(TD3CartpoleCriticNN, self).state_dict()


# Initialize critic networks Qθ1, Qθ2, and actor network πφ with random parameters θ1, θ2, φ
critic_net_1 = TD3CartpoleCriticNN()
critic_net_2 = TD3CartpoleCriticNN()
actor_net = TD3CartpoleActorNN()

# Initialize target networks θ'1 ← θ1, 0'2 ← θ2, φ' ← φ
critic_target_net_1 = TD3CartpoleCriticNN()
critic_target_net_1.load_state_dict(critic_net_1.state_dict())
critic_target_net_2 = TD3CartpoleCriticNN()
critic_target_net_2.load_state_dict(critic_net_2.state_dict())
actor_target_net = TD3CartpoleActorNN()
actor_target_net.load_state_dict(actor_net.state_dict())

# # convert to float
# critic_net_1 = critic_net_1.float()
# critic_target_net_1 = critic_target_net_1.float()
# critic_net_2 = critic_net_2.float()
# critic_target_net_2 = critic_target_net_2.float()
# actor_net = actor_net.float()
# actor_target_net = actor_target_net.float()

# Initialize replay buffer B
replay_buffer = []

# initialize the environment
env = gym.make('Pendulum-v0')
curr_state = env.reset()

# initialize critic losses
critic_net_1_loss = nn.MSELoss(reduction='mean')
critic_net_2_loss = nn.MSELoss(reduction='mean')

# initialize optimizers
critic_net_1_optimizer = optim.Adam(critic_net_1.parameters(), lr=0.001)
critic_net_2_optimizer = optim.Adam(critic_net_2.parameters(), lr=0.001)
actor_net_optimizer = optim.Adam(actor_net.parameters(), lr=0.001)

# TODO replace 3 and 4 with -1?
# TODO remove empty comma?

# initialize normal distribution N
normal_dist = normal.Normal(0, sigma)

# for t = 1 to T do
for t in range(num_iterations):
    # Select action with exploration noise a ∼ πφ(s) + ϵ ,ϵ ∼ N (0, σ), and observe reward r and new state s'
    # action = actor_net(tensor(curr_state).view(1, 3,)) + np.random.normal(0, sigma, 1)  # TODO clip & sigma
    action = clamp(actor_net(tensor(curr_state).view(1, 3,).float()) + clamp(normal_dist.sample(), -epsilon_limit, epsilon_limit), min_action, max_action)
    action_no_grad = action.detach()
    print("action size", action.size())
    next_state, reward, done, _ = env.step(action_no_grad)

    # Store transition tuple (s, a, r, s') in B
    replay_buffer.append((tensor(curr_state).view(3,), tensor(action).view(1,), tensor(reward).view(1,), tensor(next_state).view(3,), tensor(done).view(1,)))
    if len(replay_buffer) > replay_memory_max_size + 10:
        replay_buffer = replay_buffer[10:]

    # Sample mini-batch of N transitions (s, a, r, s') from B
    transitions_minibatch = random.choices(replay_buffer, k=minibatch_size)
    minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_dones = zip(*transitions_minibatch)

    # a˜ ← πφ0 (s') + ϵ, ϵ ∼ clip(N (0, σ˜), −c, c)
    # next_actions = [actor_target_net(np.asarray(s_prime)).reshape(1, 3,) + np.random.normal(0, sigma, 1)
    #                 for _, _, _, s_prime in transitions_minibatch]  # TODO clip & sigma
    minibatch_next_actions = clamp(actor_target_net(tensor(minibatch_next_states)) + clamp(normal_dist.sample(sample_shape=(minibatch_size, 1,)), -epsilon_limit, epsilon_limit), min_action, max_action)
    print("minibatch next actions size", minibatch_next_actions.size())

    # TODO y should be r depending on terminal state
    # y ← r + γ mini=1,2 Qθ'i(s', a˜)
    minibatch_y = minibatch_rewards + discount_rate * torch.min(critic_target_net_1(minibatch_next_states, minibatch_next_actions), critic_target_net_2(minibatch_next_states, minibatch_next_actions)) * tensor(-minibatch_dones + 1)
    print("minibatch y size", minibatch_y.size())

    minibatch_x = tensor(minibatch_states, minibatch_actions)

    # transitions_minibatch_y = []
    # for (_, _, r, s_prime), a_prime in zip(transitions_minibatch, next_actions):
    #     critic_x = np.asarray(s_prime + a_prime).reshape(1, 4,) # TODO appending working?
    #     a_prime_value = (min(critic_target_net_1(critic_x), critic_target_net_2(critic_x)))
    #     transitions_minibatch_y.append(r + discount_rate*a_prime_value)

    # transitions_minibatch_y = np.asarray(transitions_minibatch_y).reshape(minibatch_size, 1,)
    # transitions_minibatch_x = np.asarray([np.asarray(s + a).reshape(1, 4,) for s, a, r, s_prime in transitions_minibatch]).reshape((minibatch_size, 1,))

    # Update critics θi ← argminθi (N^−1)*Σ(y−Qθi(s, a))^2
    critic_net_1.zero_grad()
    critic_net_1_loss_output = critic_net_1_loss(minibatch_x, minibatch_y)
    critic_net_1_loss.backward()
    critic_net_1_optimizer.step()

    critic_net_2.zero_grad()
    critic_net_2_loss_output = critic_net_2_loss(minibatch_x, minibatch_y)
    critic_net_2_loss.backward()
    critic_net_2_optimizer.step()

    # if t mod d then
    if t % steps_until_policy_update == 0:
        # Update φ by the deterministic policy gradient: ∇φJ(φ) = N −1 P∇aQθ1(s, a)|a=πφ(s)∇φπφ(s)
        actor_net.zero_grad()
        actor_net_loss = -1 * critic_net_1(minibatch_states, actor_net(minibatch_states)).mean()
        actor_net_loss.backward()
        actor_net_optimizer.step()

        # Update target networks:
        # θ'i ← τθi + (1 − τ )θ'i
        for critic_target_net_1_parameter, critic_net_1_parameter in zip(critic_target_net_1.parameters(), critic_net_1.parameters()):
            critic_target_net_1_parameter.data = target_update_ratio*critic_net_1_parameter + (1-target_update_ratio)*critic_target_net_1_parameter

        for critic_target_net_2_parameter, critic_net_2_parameter in zip(critic_target_net_2.parameters(), critic_net_2.parameters()):
            critic_target_net_2_parameter.data = target_update_ratio*critic_net_2_parameter + (1-target_update_ratio)*critic_target_net_2_parameter

        # φ' ← τφ + (1 − τ )φ'
        for actor_target_net_parameter, actor_net_parameter in zip(actor_target_net.parameters(), actor_net.parameters()):
            actor_target_net_parameter.data = target_update_ratio*actor_net_parameter + (1-target_update_ratio)*actor_target_net_parameter

    # end if
    if t % (num_iterations // 10) == 0 or t == num_iterations - 1:
        print("iter", t)
        torch.save(critic_net_1.state_dict(), 'td3/models/' + model_name + '-critic_net_1')
        torch.save(critic_target_net_1.state_dict(), 'td3/models/' + model_name + '-critic_target_net_1')
        torch.save(critic_net_2.state_dict(), 'td3/models/' + model_name + '-critic_net_2')
        torch.save(critic_target_net_2.state_dict(), 'td3/models/' + model_name + '-critic_target_net_2')
        torch.save(actor_net.state_dict(), 'td3/models/' + model_name + '-actor_net')
        torch.save(actor_target_net.state_dict(), 'td3/models/' + model_name + '-actor_target_net')

    if not done:
        curr_state = next_state
    else:
        curr_state = env.reset()
# end for