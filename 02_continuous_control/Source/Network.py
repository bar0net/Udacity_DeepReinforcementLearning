import Source.Models as Models

import torch.optim as optim
import torch.nn.functional as F

from abc import ABC, abstractmethod

class Network(ABC):
    """ Wrapper for leraning models """
    def __init__(self, model, learning_rate=1e-4, weight_decay=0, tau=1e-3):
        self.local = model
        self.optimizer = optim.Adam(self.local.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.tau = tau
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        Network.soft_update(self.local, self.target, self.tau)
    
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
    
    
class DDPG_Actor(Network):
    def __init__(self, state_size, action_size, fc1_units=200, fc2_units=200, 
                 device="cpu", seed=0, learning_rate=1e-4, weight_decay=0, tau=1e-3):
        """ Initialize Class """        
        model = Models.Actor(state_size, action_size, fc1_units, fc2_units, device, seed)
        super().__init__(model, learning_rate, weight_decay, tau)
        self.target = Models.Actor(state_size, action_size, fc1_units, fc2_units, device, seed)
        
        
class DDPG_Critic(Network):
    def __init__(self, state_size, action_size, fc1_units=200, fc2_units=200, 
                 device="cpu", seed=0, learning_rate=1e-4, weight_decay=0, tau=1e-3, gamma=0.99):
        """ Initialize Class """
        model = Models.Critic(state_size, action_size, fc1_units, fc2_units, device, seed)
        super().__init__(model, learning_rate, weight_decay, tau)
        self.target = Models.Critic(state_size, action_size, fc1_units, fc2_units, device, seed)
        self.gamma = gamma
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    