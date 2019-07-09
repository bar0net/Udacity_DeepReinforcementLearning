from Source.BaseBuffer import BaseBuffer

from collections import namedtuple
import numpy as np
import random

class ReplayBuffer(BaseBuffer):
    def __init__(self, buffer_size, batch_size, seed):
        super().__init__(buffer_size, batch_size, seed)
        self.type = "replay"
        
        field_names = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=field_names)

    def add(self, state, action, reward, next_state, done):
        # Add new experience to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        
    def sample(self, alpha=0, beta=0):
        # Returns a collection of random items in memory
        selection = random.sample(self.memory, k=self.batch_size)
        
        states      = np.vstack([x.state      for x in selection if x is not None])
        actions     = np.vstack([x.action     for x in selection if x is not None])
        rewards     = np.vstack([x.reward     for x in selection if x is not None])
        next_states = np.vstack([x.next_state for x in selection if x is not None])
        dones       = np.vstack([x.done       for x in selection if x is not None])
        
        return states, actions, rewards, next_states, dones