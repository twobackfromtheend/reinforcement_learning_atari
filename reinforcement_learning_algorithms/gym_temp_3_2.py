import math
import random

import gym
import torch
from quicktracer import trace
from torch import nn, optim

from reinforcement_learning_algorithms.utils.episodic_memory.episodic_memory import EpisodicExperienceReplayBuffer

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
print(env.observation_space)
print(env.action_space)

device = torch.device("cuda")


class Model(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.lstm_layers = 1
        self.lstm_hidden = 64
        self.lstm_1 = nn.LSTM(inputs, self.lstm_hidden, num_layers=self.lstm_layers)

        self.lin_1 = nn.Linear(self.lstm_hidden, 64)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(64, outputs)
        # self.relu_2 = nn.LeakyReLU()

    def initial_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device),
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device))

    def forward(self, x, hidden):
        # print(x.size())
        x, hidden = self.lstm_1(x, hidden)
        x = self.lin_1(x)
        x = self.relu_1(x)
        x = self.lin_2(x)
        # x = self.relu_2(x)

        return x, hidden


def np_to_device(array):
    return torch.from_numpy(array).float().to(device)


model = Model(2, 2).float().to(device)
memory = EpisodicExperienceReplayBuffer(size=1000)

# opt = optim.Adam(model.parameters(), lr=3e-4)
opt = optim.Adam(model.parameters(), lr=1e-2)
# opt = optim.SGD(model.parameters(), lr=1e-3)
# opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    episodes = memory.sample(BATCH_SIZE)
    opt.zero_grad()
    batch_errors = []

    for states, actions, rewards, next_states, dones in episodes:
        # opt.zero_grad()
        actions_tensor = torch.tensor(actions, device=device).long().unsqueeze(1)
        rewards_tensor = np_to_device(rewards)

        states_tensor = np_to_device(states).view((-1, 1, 2))
        # state_values, hidden = model(states_tensor, model.initial_hidden(1))
        # print(model(states_tensor, model.initial_hidden(1))[0].shape, actions_tensor.shape)
        # print(model(states_tensor, model.initial_hidden(1))[0].squeeze(1).shape)
        state_values = model(states_tensor, model.initial_hidden(1))[0].squeeze(1)
        state_action_values = state_values.gather(1, actions_tensor)
        # print(state_action_values)

        next_state_values = torch.zeros(len(states), device=device)
        # print(state_values.shape)
        # print(state_values[1:].max(1)[0].shape)
        next_state_values[:-1] = state_values[1:].max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor

        # import numpy as np
        # print(np.array([dones, state_action_values.squeeze().detach().cpu().numpy(), rewards_tensor.cpu().numpy(), expected_state_action_values.cpu().numpy()]).T)
        # print(expected_state_action_values)
        # print(state_action_values)
        errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
        batch_errors.append(errors.flatten())
        # batch_errors.append(torch.mean(errors).flatten())

        # Compute Huber loss
        # loss = torch.mean(torch.where(errors < 1, 0.5 * errors ** 2, errors - 0.5))
        # loss = torch.mean(errors ** 2)
        # print(loss)
        # trace(loss.cpu().item())
        # loss.backward()
        # for param in model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # opt.step()
    # print(batch_errors)
    # print([_x.shape for _x in batch_errors])
    batch_errors = torch.cat(batch_errors)
    loss = torch.mean(torch.where(batch_errors < 1, 0.5 * batch_errors ** 2, batch_errors - 0.5))
    print(loss)
    trace(loss.cpu().item())
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    opt.step()


BATCH_SIZE = 24
GAMMA = 0.76
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 3000
TARGET_UPDATE = 10

steps_done = 0


def select_action(state, hidden):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    trace(eps_threshold)
    trace(steps_done)
    steps_done += 1
    out, hidden = model(np_to_device(state).view((1, 1, -1)), hidden)

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(0) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print(out)
            action = out.max(2)[1].item()
            trace(action)
    else:
        action = random.getrandbits(1)
    return action, hidden

def get_obs(state):
    return state[[0, 2]]


num_episodes = 3000

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = get_obs(env.reset())
    step = 0
    hidden = model.initial_hidden(1)
    while True:
        # env.render()

        # Select and perform an action
        action, hidden = select_action(state, hidden)
        # print(f"a: {action}")
        next_state, reward, done, _ = env.step(action)
        next_state = get_obs(next_state)
        # print(reward, done)
        # reward = 0 if done else reward
        # Store the transition in memory
        memory.add(state, action, reward, done)

        # Move to the next state
        state = next_state

        if done:
            print(step)
            episode_duration = step
            trace(episode_duration)
            break
        step += 1
    # Perform one step of the optimization
    optimize_model()
    # Update the target network, copying all weights and biases in DQN
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())
