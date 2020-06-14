import numpy as np
import torch
from quicktracer import trace
from torch import optim

from reinforcement_learning_algorithms.utils.episodic_memory.episodic_memory import EpisodicExperienceReplayBuffer


class DDQNAgent:
    def __init__(self, generate_model, lr, device):
        self.device = device

        self.model = generate_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_model = generate_model()

        self.memory = EpisodicExperienceReplayBuffer(size=1000)

    def optimize_model(self, batch_size, gamma, get_tensor_from_obs):
        for i in range(1):
            episodes = self.memory.sample(batch_size)
            self.optimizer.zero_grad()
            batch_errors = []

            for states, actions, rewards, next_states, dones in episodes:
                actions_tensor = torch.tensor(actions, device=self.device).long().unsqueeze(1)
                rewards_tensor = self.np_to_device(rewards)

                states_tensor = get_tensor_from_obs(states)
                # print(states_tensor.shape)
                state_values = self.model(states_tensor, self.model.initial_hidden(1))[0].squeeze(1)
                state_action_values = state_values.gather(1, actions_tensor)
                # print(state_action_values)

                target_next_state_values = torch.zeros(len(states), device=self.device)
                target_state_values = self.target_model(states_tensor, self.model.initial_hidden(1))[0].squeeze(1)
                target_next_state_values[:-1] = target_state_values[1:].max(1)[0].detach()
                expected_state_action_values = (target_next_state_values * gamma) + rewards_tensor

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
            self.optimizer.step()

    def update_target(self, model, target_model, tau=1):
        if tau == 1:
            target_model.load_state_dict(model.state_dict())
        else:
            for target_param, model_param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * model_param.data + (1.0 - tau) * target_param.data)

    def np_to_device(self, array):
        return torch.from_numpy(array).float().to(self.device)
