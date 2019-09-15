# rl-algorithms
An implementation-focused survey of popular reinforcement learning algorithms

### Goal:

The goal of this project is to implement some algorithms that I have come across repeatedly so that I can understand their workings, nuances, and tuning processes. 

### Implementation:
The implementations are derived directly from their respective publications along with any supplementary papers that shed light on how the algorithm works. This project is focused on understanding, so (as much as it pains me) the algorithms will be in a script format rather than object-oriented. I find that this format is easier to follow when trying to learn their workings, even though it isn't the prettiest.

### Benchmarking:
For benchmarking I'm using OpenAI Gym's CartPole-v1, Pong, and Demon Attack environments. I am also using the implementations in Stable Baselines (a fork of OpenAI's baselines) for comparison.

### Algorithms:
A tentative list of algorithms to be implemented, linked to their respective papers, is below:

- [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) ([and supplementary paper](https://arxiv.org/pdf/1901.00137.pdf))
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Twin Delayed Deep Deterministic policy gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Actor Critic with Experience Replay (ACER)](https://arxiv.org/pdf/1611.01224.pdf)
- [Advantage Actor Critic (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1801.01290.pdf)