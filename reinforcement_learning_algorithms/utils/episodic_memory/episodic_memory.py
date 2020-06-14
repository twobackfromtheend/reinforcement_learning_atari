import random
from collections import deque
from typing import Deque, Sequence, NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    done: bool


class Episode(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class EpisodicExperienceReplayBuffer:
    """
    N.B. keras-rl uses their own memory class (as of writing), claiming deque is non-performant.
    But https://github.com/keras-rl/keras-rl/issues/165 - benchmarks show deque is faster.
    """

    def __init__(self, size: int):
        self.size = size
        self.memory: Deque[Sequence[Experience]] = deque(maxlen=size)
        self.episode_buffer = []

    def __len__(self):
        return len(self.memory)

    def add(self, state: np.ndarray, action: int, reward: float, done: bool = None):
        # Save step data for episode in episode_buffer
        self.episode_buffer.append(Experience(state, action, reward, done))
        # print(type(state), isinstance(state, np.ndarray))
        assert isinstance(state, np.ndarray), f"Added state to memory is not of type np.ndarray (found: {type(state)})"
        # print(action)
        assert isinstance(action, int), f"Added action to memory is not of type int (found: {type(action)})"
        assert isinstance(reward, (int, float)), f"Added reward to memory is not of type float (found: {type(reward)})"
        assert isinstance(done, bool), f"Added done to memory is not of type bool (found: {type(done)})"

        if done:
            # Append full episode to memory
            episode_states, episode_actions, episode_rewards, episode_dones = zip(*self.episode_buffer)
            # print(np.array(episode_states).shape)
            # print(np.array(episode_states))
            episode_next_states = np.roll(episode_states, 1, axis=0)
            episode_next_states[0, :] = 0
            self.memory.append(
                Episode(
                    np.array(episode_states),
                    np.array(episode_actions),
                    np.array(episode_rewards),
                    np.array(episode_next_states),
                    np.array(episode_dones)
                )
            )
            self.episode_buffer = []

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
