import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import adam
from tensorflow.python.keras.losses import mse

import gym

# some bookkeeping to keep things organized
model_name = 'DQN-CartPole-v1'


"""
Input: MDP (S, A, P, R, γ), replay memory M, number of iterations T, minibatch size n,
exploration probability ǫ ∈ (0, 1), a family of deep Q-networks Qθ : S × A → R, an integer Ttarget
for updating the target network, and a sequence of stepsizes {αt}t≥0.
"""

discount_rate = 0.99
num_iterations = 10000
minibatch_size = 32
num_steps_to_target_update = 10

# we will anneal (reduce) the exploration probability from 1 to 0.1 (as in the paper)
# our problem currently is not quite as complex, so we can't use the exact method in the paper
# instead we'll just use a heuristic of decreasing it linearly over the first 75% of the learning phase
start_exploration_prob = 1
end_exploration_prob = 0.1
exploration_annealment_portion = 0.75
exploration_annealment_rate = ((start_exploration_prob - end_exploration_prob) / (num_iterations * exploration_annealment_portion))
exploration_prob = start_exploration_prob

# we'll do the same thing with the stepsize, annealing it linearly from 0.001 to 0.0001 (just a guess)
# this will occur over the last 80% of the iterations
start_stepsize = 0.001
end_stepsize = 0.0001
stepsize_annealment_portion = 0.8
stepsize_annealment_rate = ((start_stepsize - end_stepsize) / (num_iterations * stepsize_annealment_portion))
stepsize = start_stepsize


# Initialize the replay memory M to be empty
replay_memory = []
replay_memory_max_size = 1000

# Initialize the Q-network with random weights θ
q_net = Sequential([
    Dense(10, input_shape=(None, 4,)),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(2),
    Activation('linear')
])

# we use the adam optimizer because it generally does well without needing much tuning
# the loss is mse because it has been defined as such in the Minh et al. paper
q_net.compile(optimizer=adam(lr=stepsize), loss=mse, metrics=['accuracy'])

# Initialize the weights of the target network with θ⋆ = θ
target_net = Sequential([
    Dense(10, input_shape=(None, 4,)),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(2),
    Activation('linear')
])
target_net.compile(optimizer=adam(lr=stepsize), loss=mse, metrics=['accuracy'])
target_net.set_weights(q_net.get_weights())

# Initialize the initial state S0
cp_env = gym.make('CartPole-v1')
curr_state = cp_env.reset()

# for t = 0, 1, ..., T do
for t in range(num_iterations):
    # With probability epsilon, choose At uniformly at random from A
    if random.uniform(0, 1) < exploration_prob:
        action = cp_env.action_space.sample()
    # and with probability 1 − epsilon, choose At such that Qθ(St, At) = maxa ∈ A Qθ(St, a)
    else:
        action = np.argmax(np.squeeze(q_net.predict(curr_state.reshape(1, 1, 4,))))

    # Execute At and observe reward Rt and the next state St+1
    next_state, reward, done, _ = cp_env.step(action)

    # Store transition (St, At, Rt, St+1) in M
    replay_memory.append((curr_state, action, reward, next_state, done))

    # # only fit if past minibatch_size to avoid biased samples
    # if len(replay_memory) > minibatch_size:
    # Experience replay: Sample random minibatch of transitions {(si, ai, ri, s′i)}i∈[n]from M
    transitions_minibatch_x = random.choices(replay_memory, k=minibatch_size)
    print(len(transitions_minibatch_x))

    # For each i ∈ [n], compute the target Yi = ri + γ · maxa∈A Qθ⋆(s′i, a)
    # special case of reward = returns if terminal state
    # I felt it was best to break this up into a full loop because it was monstrous as a list comp
    transitions_minibatch_Yi = []
    for _, _, r, s_prime, terminal_state in transitions_minibatch_x:
        if not terminal_state:
            # reshape to fit, and squeeze to get rid of extra dimension that would trivialize the max function
            transitions_minibatch_Yi.append(r + discount_rate * max(np.squeeze(target_net.predict(s_prime.reshape(1, 1, 4,)))))
        else:
            transitions_minibatch_Yi.append(r)

    # we will change only the Q value of the best action, and keep others the same
    # this will allow us to easily fit the model
    transitions_minibatch_y = [q_net.predict(s.reshape(1, 1, 4,)) for s, _, _, _, _ in transitions_minibatch_x]
    for y, Yi, (_, a, _, _, _) in zip(transitions_minibatch_y, transitions_minibatch_Yi, transitions_minibatch_x):
        y[0, 0, a] = Yi

    # Update the Q-network: Perform a gradient descent step
    q_net.fit(x=np.array([s for s, _, _, _, _ in transitions_minibatch_x]).reshape(len(transitions_minibatch_x), 1, 4,), y=np.array(transitions_minibatch_y).reshape(32, 1, 2), batch_size=minibatch_size)

    # Update the target network: Update θ⋆ ← θ every Ttarget steps
    if t % num_steps_to_target_update == 0:
        target_net.set_weights(q_net.get_weights())

    # annealing the exploration rate from 1 to 0.1 over the first 75% of the steps
    if t <= num_iterations * exploration_annealment_portion:
        exploration_prob -= exploration_annealment_rate

    # annealing stepsize linearly from 0.001 to 0.0001 over last 80% of the steps
    if t >= num_iterations * (1 - stepsize_annealment_portion):
        stepsize -= stepsize_annealment_rate

    # adjust replay memory size
    if len(replay_memory) > replay_memory_max_size+10:
        replay_memory = replay_memory[10:]

    # save our model every ~10% of training completion
    if t % (num_iterations//10) == 0 or t == num_iterations-1:
        print("iter", t)
        q_net.save('dqn/models/' + model_name + '-qnet')
        target_net.save('dqn/models/' + model_name + '-targetnet')

    if not done:
        curr_state = next_state
    else:
        curr_state = cp_env.reset()


#end for

# Define policy π as the greedy policy with respect to Qθ
# Output: Action-value function Qθ and policy π.


# evaluate model over 25000 steps or 100 episodes, whichever comes first
# a successful model will result in an average of at least 195.0 steps per episode over 100 consecutive episodes
# it looks like each episode will end after 500 steps max
render = True
num_eval_steps = 25000
num_eval_episodes = 100

obs = cp_env.reset()
episode_durations = []
current_duration = 0
for i in range(num_eval_steps):
    if len(episode_durations) == num_eval_episodes:
        break
    action = np.argmax(np.squeeze(q_net.predict(obs.reshape(1, 1, 4))))
    obs, reward, done, _ = cp_env.step(action)
    current_duration += reward
    if done == 1:
        episode_durations.append(current_duration)
        current_duration = 0
        obs = cp_env.reset()
    if render:
        cp_env.render()

episode_durations = np.asarray(episode_durations)
episode_length_histogram = plt.hist(episode_durations)
plt.title("Episode Lengths")
plt.xlabel("Number of Steps")
plt.ylabel("Frequency")
plt.savefig("dqn/episode_length_hist.png")
print("Mean episode length:", np.mean(episode_durations))