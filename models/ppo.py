import logging

import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_critic import A2CTrainer, A2CInferer
from utils import TRAINER, INFERER
from modules.replay_buffer import BatchReplayBuffer


logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, state):
        return self.model(state)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, state):
        return self.model(state)
 

@TRAINER.register('PPO')
class PPOTrainer(A2CTrainer):
    def __init__(self, num_epochs=10, buffer_size=1000, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self.num_epoches = num_epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = BatchReplayBuffer(capacity=buffer_size)
        # fixed hyper parameter
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2
        
    def train_one_episode(self):
        state, _ = self.env.reset()
        done = False
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = self.actor(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # take action
            next_state, reward, done, _, _ = self.env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            dones.append(done)
            
            state = next_state
            total_reward += reward
        
        # compute values, no need to compute next_values for efficiency 
        states = torch.tensor(states, dtype=torch.float32)
        values = self.critic(states)
        
        # compute GAE
        returns = self.compute_gae(rewards, values, dones)
        
        # NOTE: advantage is used to calculate actor loss, gradient should
        # not passed into critic, so detach() is used to detach the gradient
        # from the graph
        advantages = returns - values.detach()
        
        # save into replay buffer
        self.replay_buffer.push(states, actions, returns, log_probs, advantages)
        
        if len(self.replay_buffer) < self.batch_size:
            return {'ciritic_loss': 0, 'actor_loss': 0, 'total_loss': 0}
        # train for k epoches
        for _ in range(self.num_epoches):
            # all data are in batch mode
            states, actions, returns, old_log_probs, advantages = self.replay_buffer.sample(self.batch_size)
            # convert to tensor
            states = torch.stack(states)  # [B, state_dim]
            actions = torch.stack(actions)  # [B]
            returns = torch.stack(returns).unsqueeze(-1)  # [B, 1]
            old_log_probs = torch.stack(old_log_probs)  # [B]
            advantages = torch.stack(advantages)  # [B]
            # predict values and new log probs on actor whose parameter is updating
            logits = self.actor(states)
            values = self.critic(states)
            dists = torch.distributions.Categorical(logits=logits)
            new_log_probs = dists.log_prob(actions)
            
            # why exp?
            ratio = (new_log_probs - old_log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (values - returns).pow(2).mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dists.entropy().mean()
            
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()
        
        return {
            'ciritic_loss': critic_loss.item(), 
            'actor_loss': actor_loss.item(), 
            'total_loss': total_loss.item(), 
            'total_reward': total_reward,
        }
        
    def compute_gae(self, rewards, values, dones):
        """
        计算 GAE （Generalized Advantage Estimation）。
        参数:
            rewards (list): 每个时间步的奖励。
            values (list): 每个时间步的价值估计。
            dones (list): 每个时间步是否完成。
        返回:
            list: 每个时间步的 GAE 值。
        """
        gae = 0
        next_value = 0.0
        returns = []
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = td_error + self.gamma * self.lamda * gae
            returns.insert(0, gae)
            next_value = values[i]
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)
        return returns


@INFERER.register('PPO')
class PPOInferer(A2CInferer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)