import math
import random

import gym
import numpy as np
import torch
from gym.wrappers import AtariPreprocessing
from quicktracer import trace
from torchvision.transforms.functional import to_tensor

from reinforcement_learning_algorithms.agents.ddqn import DDQNAgent
from reinforcement_learning_algorithms.gym_stateful_lstm.model import LSTMAtariModel

### Define environment
env = gym.make('CartPole-v0')
# env = gym.make('Breakout-v0')
# env = WarpFrame(env)
AtariPreprocessing

def get_obs(state):
    return state


def get_tensor_from_obs(obs):
    return to_tensor(obs).unsqueeze(0).to(device)


obs_shape = env.observation_space.shape
n_actions = env.action_space.n

print(env.observation_space)
print(env.action_space)

device = torch.device("cuda")

agent = DDQNAgent(
    generate_model=lambda: LSTMAtariModel(*obs_shape, n_actions, device).float().to(device),
    device=device
)

WARMUP = 24
BATCH_SIZE = 3
GAMMA = 0.995
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 10000

steps_done = 0


def select_action(state, hidden):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    trace(eps_threshold)
    trace(steps_done)
    steps_done += 1
    # print(state.shape, "oipetreioreuio")
    out, hidden = agent.model(get_tensor_from_obs(state), hidden)

    if sample > eps_threshold:
        with torch.no_grad():
            action = out.max(2)[1].item()
            trace(action)
    else:
        action = random.getrandbits(2)
    return action, hidden


num_episodes = 3000

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

        if done:
            episode_rewards.append(episode_reward)
            print(
                f"Ep: {len(episode_rewards): 3d}, \tstep: {step: 4d}, \treward: {int(episode_reward): 2d}, \taverage_reward: {np.mean(episode_rewards[-20:]):.2f}")
            # print(step)
            episode_duration = step
            trace(episode_duration)
            break
