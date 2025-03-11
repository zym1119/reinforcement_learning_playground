import logging

import torch
import torch.nn as nn
import torch.optim as optim

from runner import BaseTrainer, BaseInferer


logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
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
    
    def forward(self, state):
        return self.model(state)
    

class A2CTrainer(BaseTrainer):
    """A2C训练器，包含训练逻辑"""
    
    def init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
    
    def before_train_hook(self):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor.train()
        self.critic.train()
        logger.info('start training...')
        
    def train_one_episode(self):
        state, _ = self.env.reset()
        done = False
        log_probs = []
        rewards = []
        states = []
        next_states = []
        dones = []
        total_reward = 0

        # forward 1 episode
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
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            total_reward += reward
        
        # to tensor
        log_probs = torch.stack(log_probs)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        values = self.critic(states).squeeze()
        # with torch.no_grad():
        next_values =  self.critic(next_states).squeeze() * (1 - dones)
        target_values = rewards + 0.99 * next_values
        
        # caculate critic loss
        td_error = target_values - values
        critic_loss = torch.nn.functional.mse_loss(values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # calculate actor loss
        actor_loss = (-log_probs * td_error.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'reward': sum(rewards)}
    
    
    def save_model(self, is_last=False):
        if is_last:
            save_path = self.run_dir / 'model_last.pth'
        else:
            save_path = self.run_dir / f'model_episode{self.episode}.pth'
        # 保存模型的状态字典
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, save_path)
        logger.info(f'save ckpt {save_path}')
    
    
class A2CInferer(BaseInferer):
    def init_model(self, ckpt_path):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = PolicyNetwork(state_dim, action_dim)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['actor'])
    
    
def get_a2c_trainer(env, run_dir, **kwargs):
    return A2CTrainer(env=env, run_dir=run_dir, **kwargs)


def get_a2c_inferer(env, ckpt_path, **kwargs):
    return A2CInferer(env=env, ckpt_path=ckpt_path, **kwargs)