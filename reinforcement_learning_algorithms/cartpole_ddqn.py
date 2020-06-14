import math
import random

import gym
import numpy as np
import torch
from gym import wrappers
from quicktracer import trace

from torchvision.transforms.functional import to_tensor

from reinforcement_learning_algorithms.agents.ddqn import DDQNAgent
from reinforcement_learning_algorithms.envs.wrappers.discrete_pendulum import DiscretePendulum4, PendulumNoVel, \
    DiscretePendulum2
from reinforcement_learning_algorithms.gym_stateful_lstm.model import LSTMCartPoleModel

### Define environment
# env = gym.make('CartPole-v0')
env = gym.make('Pendulum-v0')
env = DiscretePendulum4(env)
env = PendulumNoVel(env)


def get_obs(state):
    return state


def get_tensor_from_obs(obs):
    return agent.np_to_device(obs).view((-1, 1, *obs_shape))


obs_shape = env.observation_space.shape
n_actions = env.action_space.n

print(env.observation_space)
print(env.action_space)

device = torch.device("cuda")

agent = DDQNAgent(
    generate_model=lambda: LSTMCartPoleModel(*obs_shape, n_actions, device).float().to(device),
    lr=3e-4,
    device=device,
)

WARMUP = 24
BATCH_SIZE = 6
GAMMA = 0.99
EPSILON_START = 0.8
EPSILON_END = 0.05
EPSILON_DECAY = 50

EPISODES = 0


def select_action(state, hidden):
    global EPISODES

    sample = random.random()
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * EPISODES / EPSILON_DECAY)
    # trace(epsilon)
    # trace(EPISODES)
    # print(state.shape, "oipetreioreuio")
    # print(state)
    out, hidden = agent.model(get_tensor_from_obs(state), hidden)

    if sample > epsilon:
        with torch.no_grad():
            action = out.max(2)[1].item()
            # trace(action)
    else:
        action = random.getrandbits(2)
    return action, hidden


num_episodes = 3000

warmed_up = False

episode_rewards = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = get_obs(env.reset())

    step = 0
    episode_reward = 0
    hidden = agent.model.initial_hidden(1)
    while True:
        # env.render()

        action, hidden = select_action(state, hidden)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -1
        used_action = action
        trace(used_action)
        next_state = get_obs(next_state)
        trace(reward)

        agent.memory.add(state, action, reward, done)

        state = next_state

        episode_reward += reward
        step += 1
        if warmed_up:
            if step % 10 == 0:
                agent.optimize_model(batch_size=BATCH_SIZE, gamma=GAMMA, get_tensor_from_obs=get_tensor_from_obs)
                agent.update_target(model=agent.model, target_model=agent.target_model, tau=0.03)

        if done:
            EPISODES += 1
            episode_rewards.append(episode_reward)
            print(
                f"Ep: {len(episode_rewards): 3d}, \tstep: {step: 4d}, \treward: {int(episode_reward): 2d}, \taverage_reward: {np.mean(episode_rewards[-20:]):.2f}")
            # print(step)
            # episode_duration = step
            # trace(episode_duration)
            trace(episode_reward)

            if len(agent.memory) >= WARMUP:
                if len(agent.memory) == WARMUP:
                    print("Replay memory warmed up.")
                    warmed_up = True
            break
