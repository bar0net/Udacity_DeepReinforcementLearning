import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=200, fc2_units=200, device="cpu", seed=0):
        super(Actor,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, action_size).to(device)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=200, fc2_units=200, device="cpu", seed=0):
        super(Critic,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, 1).to(device)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        x_state = F.relu(self.fc1(state))
        x = torch.cat((x_state, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class A3C(nn.Module):
        def __init__(self, state_size, action_size, fc1_units=200, fc2_units=200, device="cpu", seed=0):
        super(Critic,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(device)
        self.fc3A = nn.Linear(fc2_units, action_size).to(device)
        self.fc3V = nn.Linear(fc2_units, 1).to(device)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3A.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3V.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3A(x), self.fc3V(x)