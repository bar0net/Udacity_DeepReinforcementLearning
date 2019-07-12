import random
import numpy as np
import os

import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod

from Source.Network import Network, DDPG_Actor, DDPG_Critic
from Source.Utils import OUNoise

class BaseAgent(ABC):
    def __init__(self, state_size, action_size, seed=0, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.seed = random.seed(seed)
        
    def episode_start(self):
        pass
        
    def episode_end(self):
        pass
    
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def act(self, state):
       pass
   
    @abstractmethod
    def learn(self, experiences):
        pass
    
    
class DDPG_Agent(BaseAgent):
    def __init__(self, state_size, action_size, actor_network, critic_network,
                 replay_buffer, seed=0, device="cpu"):
        super().__init__(state_size, action_size, seed, device)
        
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.memory = replay_buffer
        self.noise  = OUNoise(action_size, seed)
        self.use_noise = True
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        
        self.actor_network.local.eval()
        with torch.no_grad():
            action = self.actor_network.local(state).cpu().data.numpy()
        self.actor_network.local.train()
        
        if self.use_noise:
            action += self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def episode_start(self):
        self.noise.reset()
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
            
    def learn(self):
        if not self.memory.active():
            return
        
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        
        # transform numpy arrays to torch tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # Update Critic Network
        next_actions = self.actor_network.target(states)
        Q_targets_next = self.critic_network.target(next_states, next_actions)
        Q_targets  = rewards + (self.critic_network.gamma * Q_targets_next * (1-dones))
        Q_expected = self.critic_network.local(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_network.update(critic_loss)
        #self.critic_network.optimizer.zero_grad()
        #critic_loss.backward()
        #self.critic_network.optimizer.step()
        #Network.soft_update(self.critic_network.local, self.critic_network.target, 1e-3)
        
        # Update Actor Network
        predicted_actions = self.actor_network.local(states)
        actor_loss = -self.critic_network.local(states, predicted_actions).mean()
        self.actor_network.update(actor_loss)
        #self.actor_network.optimizer.zero_grad()
        #actor_loss.backward()
        #self.actor_network.optimizer.step()
        #Network.soft_update(self.actor_network.local, self.actor_network.target, 1e-3)
        
    def save(self, path, pre):
        if (path[-1] != "/"):
            path += '/'
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        torch.save(self.actor_network.local.state_dict(), '{}/{}_actor.pth'.format(path, pre))
        torch.save(self.actor_network.local.state_dict(), '{}/{}_critic.pth'.format(path, pre))
            
    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        