import random
import numpy as np

import torch
import torch.nn.functional as F

class BaseAgent:
    def __init__(self, state_size, action_size, replay_buffer,
                 qnetwork_local, qnetwork_target, optimizer,
                 gamma=0.99, alpha=0, beta_initial=0, beta_rate=0,
                 tau=1e-3, update_every=4, device="cpu", seed=0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.device = device
        self.seed = random.seed(seed)
        
        self.alpha = alpha
        self.beta = beta_initial
        self.beta_rate = beta_rate
                
        self.memory = replay_buffer
        self.t_step = 0
        
        self.qnetwork_local = qnetwork_local
        self.qnetwork_target = qnetwork_target
        self.optim = optimizer
        
    def episode_end(self):
        self.beta = min(1, self.beta + self.beta_rate)

    def step(self, state, action, reward, next_state, done):
        # Perform an action step
        self.memory.add(state, action, reward, next_state, np.int(done))
        
        # check if we are in a learning state
        self.t_step = (self.t_step+1) % self.update_every
        if self.t_step > 0:
            return
        
        # learn if there are enough samples in memory
        if self.memory.active():
           experiences = self.memory.sample(self.alpha, self.beta) 
           self.learn(experiences)
           
    def act(self, state, epsilon):
       # Returns a set of actions using epsilon-greedy policy
       if random.random() < epsilon:
           return random.randint(0, self.action_size-1)
       
        # Transform state to torch tensor
       torch_state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
       
       # Compute actions from QNetwork (in Evaluation Mode)
       self.qnetwork_local.eval()
       with torch.no_grad():
           action_values = self.qnetwork_local(torch_state)
       self.qnetwork_local.train()
       
       # Selection
       return np.argmax(action_values.cpu().numpy())
   
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
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
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
        
    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
            