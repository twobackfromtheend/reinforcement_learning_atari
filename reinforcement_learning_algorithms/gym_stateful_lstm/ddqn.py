import math
import random

import gym
import numpy as np
import torch
from quicktracer import trace
from torch import optim
from torchvision.transforms.functional import to_tensor

from reinforcement_learning_algorithms.gym_stateful_lstm.model import LSTMAtariModel
from reinforcement_learning_algorithms.utils.episodic_memory.episodic_memory import EpisodicExperienceReplayBuffer

# env = gym.make('CartPole-v0')
from reinforcement_learning_algorithms.envs.wrappers.wrappers import WarpFrame

env = gym.make('Breakout-v0')
env = WarpFrame(env)


def get_obs(state):
    return state


def get_tensor_from_obs(obs):
    return to_tensor(obs).unsqueeze(0).to(device)


obs_shape = env.observation_space.shape
n_actions = env.action_space.n

print(env.observation_space)
print(env.action_space)

device = torch.device("cuda")


def np_to_device(array):
    return torch.from_numpy(array).float().to(device)


model = LSTMAtariModel(*obs_shape, n_actions, device).float().to(device)
opt = optim.Adam(model.parameters(), lr=3e-4)
memory = EpisodicExperienceReplayBuffer(size=1000)


def optimize_model():
    if len(memory) < WARMUP:
        return
    if len(memory) == WARMUP:
        print("Memory warmed up")

    for i in range(5):
        episodes = memory.sample(BATCH_SIZE)
        opt.zero_grad()
        batch_errors = []

        for states, actions, rewards, next_states, dones in episodes:
            actions_tensor = torch.tensor(actions, device=device).long().unsqueeze(1)
            rewards_tensor = np_to_device(rewards)

            states_tensor = torch.from_numpy(np.transpose(states, axes=[0, 3, 1, 2])).to(device).float()
            # print(states_tensor.shape)
            state_values = model(states_tensor, model.initial_hidden(1))[0].squeeze(1)
            state_action_values = state_values.gather(1, actions_tensor)
            # print(state_action_values)

            next_state_values = torch.zeros(len(states), device=device)
            next_state_values[:-1] = state_values[1:].max(1)[0].detach()
            expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor

            # import numpy as np
            # print(np.array([dones, state_action_values.squeeze().detach().cpu().numpy(), rewards_tensor.cpu().numpy(), expected_state_action_values.cpu().numpy()]).T)
            errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
            batch_errors.append(errors.flatten())

        batch_errors = torch.cat(batch_errors)
        loss = torch.mean(torch.where(batch_errors < 1, 0.5 * batch_errors ** 2, batch_errors - 0.5))
        print(loss)
        trace(loss.cpu().item())
        loss.backward()
        # for param in model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        opt.step()


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
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    trace(eps_threshold)
    trace(steps_done)
    steps_done += 1
    # print(state.shape, "oipetreioreuio")
    out, hidden = model(get_tensor_from_obs(state), hidden)

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
    hidden = model.initial_hidden(1)
    while True:
        # env.render()

        action, hidden = select_action(state, hidden)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -1
        used_action = action
        trace(used_action)
        next_state = get_obs(next_state)
        trace(reward)

        memory.add(state, action, reward, done)

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
    optimize_model()
