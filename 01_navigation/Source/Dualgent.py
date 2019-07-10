from Source.BaseAgent import BaseAgent

import random
import numpy as np

import torch
import torch.nn.functional as F

class DualAgent(BaseAgent):
    def __init__(self, state_size, action_size, replay_buffer,
                 qnetwork_local, qnetwork_target, optimizer,
                 gamma=0.99, alpha=0, beta_initial=0, beta_rate=0,
                 tau=1e-3, update_every=4, device="cpu", seed=0):
        
        super().__init__(state_size, action_size, replay_buffer,
                 qnetwork_local, qnetwork_target, optimizer,
                 gamma, alpha, beta_initial, beta_rate,
                 tau, update_every, device, seed)
        
    def learn(self, experiences):
        # learn from experiences
        
        # unroll the experiences tuple (depending on buffer type)
        if self.memory.type == "replay":
            states, actions, rewards, next_states, dones = experiences
            indices, weights = None, None
            
        elif self.memory.type == "priority":
            states, actions, rewards, next_states, dones, indices, weights = experiences
            weights = torch.from_numpy(weights).float().to(self.device).unsqueeze(1)
            
        else:
            raise TypeError("Unrecognized Buffer Type")
            return
        
        # Set data as tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(), 1, max_actions)
            
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        # and update priorities if necessary
        if self.memory.type == "replay":
            loss = F.mse_loss(Q_expected, Q_targets)
            
        elif self.memory.type == "priority":
            loss = (Q_expected - Q_targets).pow(2) * weights
            self.memory.set_priorities(indices, loss.squeeze().cpu().data.numpy())
            loss = loss.mean()
            
        # perform a learning step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        # update network parameters
        self.soft_update()
        
            