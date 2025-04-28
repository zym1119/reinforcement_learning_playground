import torch

from models.dqn import DQNTrainer
from utils import TRAINER


@TRAINER.register('DoubleDQN')
class DoubleDQNTrainer(DQNTrainer):
    def train_one_batch(self):
        gamma = 0.99  # a constant

        # sample batch data
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)
        # to tensor
        state = torch.tensor(
            state, dtype=torch.float32)  # [B, state_dim]
        action = torch.tensor(
            action, dtype=torch.int64).unsqueeze(1)  # [B, action_dim]
        reward = torch.tensor(
            reward, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        next_state = torch.tensor(
            next_state, dtype=torch.float32)  # [B, state_dim]
        done = torch.tensor(
            done, dtype=torch.float32).unsqueeze(1)  # [B, 1]

        q_values = self.policy_network(
            state).gather(1, action)
        # double DQN implementation
        next_action = self.policy_network(
            next_state).argmax(dim=1).unsqueeze(1)  # [B, 1]
        next_q_values = self.target_network(next_state).gather(1, next_action)
        # origin DQN implementation
        # next_q_values = self.target_network(
        #     next_state).max(dim=1).values.unsqueeze(1)

        expected_q_values = reward + \
            gamma * next_q_values * (1 - done)

        loss = self.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
