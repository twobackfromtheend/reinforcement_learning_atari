import math
import random

import gym
import torch
from torch import nn, optim

from reinforcement_learning_algorithms.utils.gym_dqn_memory.replay_buffer import ReplayBuffer

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
print(env.observation_space)
print(env.action_space)

device = torch.device("cuda")


class Model(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.lin_1 = nn.Linear(inputs, 64)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(64, 64)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(64, outputs)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu_1(self.lin_1(x))
        x = self.relu_2(self.lin_2(x))
        return self.lin_3(x)
        # return self.sig(self.lin_3(x))


def get_model(inputs, outputs):
    model = nn.Sequential(
        nn.Linear(inputs, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),

        nn.Linear(64, outputs),
        nn.Sigmoid(),
    )
    return model.to(device)


# model = nn.Sequential(
#     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.AvgPool2d(4),
# )


def np_to_device(array):
    return torch.from_numpy(array).float().to(device)


model = Model(4, 2).float().to(device)
memory = ReplayBuffer(size=10000)

opt = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)


# opt = optim.RMSprop(model.parameters())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(~dones)
    non_final_next_states = np_to_device(next_states[~dones])
    states_tensor = np_to_device(states)
    actions_tensor = torch.tensor(actions, device=device).long().unsqueeze(1)
    rewards_tensor = np_to_device(rewards)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(model(states_tensor), model(states_tensor).shape)
    # print(actions_tensor, actions_tensor.shape)
    state_action_values = model(states_tensor).gather(1, actions_tensor)
    # print(state_action_values)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + rewards_tensor
    # print('a', flush=True)
    # import numpy as np
    # print(np.array([dones, state_action_values.squeeze().detach().cpu().numpy(), rewards_tensor.cpu().numpy(), expected_state_action_values.cpu().numpy()]).T)
    # print(dones)
    # print(state_action_values.squeeze())
    # print(rewards_tensor)
    # print(expected_state_action_values)

    # Compute Huber loss
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss)
    # Optimize the model
    opt.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    opt.step()


BATCH_SIZE = 128
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0


def select_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(0) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(model(np_to_device(state)))
            return (model(np_to_device(state)).max(0)[1]).item()
    else:
        return random.getrandbits(1)
        # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


num_episodes = 3000

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    step = 0
    while True:
        env.render()

        # Select and perform an action
        action = select_action(state)
        # print(f"a: {action}")
        next_state, reward, done, _ = env.step(action)
        reward = 0 if done else reward
        # Store the transition in memory
        memory.add(state, action, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization
        if steps_done % 5 == 0:
            optimize_model()
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            print(step)
            break
        step += 1
    # Update the target network, copying all weights and biases in DQN
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())
