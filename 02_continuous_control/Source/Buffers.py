import numpy as np
import random
from collections import deque, namedtuple

class Buffer:
    def __init__(self, buffer_size=1e5, batch_size=128, seed=0):
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", ["state", "action", "reward",
                                                    "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        
        states = np.vstack([x.state for x in samples])
        actions = np.vstack([x.action for x in samples])
        rewards = np.vstack([x.reward for x in samples])
        next_states = np.vstack([x.next_state for x in samples])
        dones = np.vstack([x.done for x in samples]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def active(self):
        return len(self.memory) >= self.batch_size