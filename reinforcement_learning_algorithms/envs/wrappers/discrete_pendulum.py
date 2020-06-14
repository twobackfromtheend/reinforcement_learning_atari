from gym import ActionWrapper, ObservationWrapper
from gym.spaces import Box, Discrete


class DiscretePendulum2(ActionWrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        super().__init__(env)
        self.action_space = Discrete(2)

    def action(self, action):
        return [-1] if action == 0 else [1]
        # return self.env.action_space.low if action == 0 else self.env.action_space.high


class DiscretePendulum4(ActionWrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        super().__init__(env)
        self.action_space = Discrete(4)

    def action(self, action):
        if action == 0:
            return self.env.action_space.low
        elif action == 1:
            return self.env.action_space.low * 0.3
        elif action == 2:
            return self.env.action_space.high * 0.3
        elif action == 3:
            return self.env.action_space.high
        else:
            raise ValueError(f"Unknown action: {action}")


class PendulumNoVel(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=env.observation_space.low[:-1],
            high=env.observation_space.high[:-1],
        )

    def observation(self, observation):
        return observation[:-1]
