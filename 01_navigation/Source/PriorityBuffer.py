from Source.BaseBuffer import BaseBuffer

from collections import namedtuple
import numpy as np

class PriorityBuffer(BaseBuffer):
    def __init__(self, buffer_size, batch_size, seed):
        super().__init__(buffer_size, batch_size, seed)
        self.type = "priority"
        
        field_names = ["state", "action", "reward", "next_state", "done", "priority"]
        self.experience = namedtuple("Experience", field_names=field_names)

    def add(self, state, action, reward, next_state, done):
        # Add new experience to memory
        priority = max([x.priority for x in self.memory]) if len(self.memory) > 0 else 1.0
        
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
        
    def set_priorities(self, indices, weights):
        # Set the priority parameter for items in memory
        for i, idx in enumerate(indices):
            self.memory[idx] = self.memory[idx]._replace(priority=(1e-5 + weights[i]))
        
    def sample(self, alpha=0, beta=0):
        # Returns a collection of random items in memory
        
        # Define probabilities associated to each entry
        probs = np.array([x.priority**alpha for x in self.memory])
        probs /= sum(probs)
        
        # Generate a random selection
        indices = np.random.choice(len(self.memory), self.batch_size, replace = False, p=probs)
        selection = [self.memory[i] for i in indices]
        
        # Generate weight array
        N = len(self.memory)
        weights  = np.float32((N * probs[indices]) ** (-beta))
        weights /= weights.max()
        
        states      = np.vstack([x.state      for x in selection if x is not None])
        actions     = np.vstack([x.action     for x in selection if x is not None])
        rewards     = np.vstack([x.reward     for x in selection if x is not None])
        next_states = np.vstack([x.next_state for x in selection if x is not None])
        dones       = np.vstack([x.done       for x in selection if x is not None])
        
        return states, actions, rewards, next_states, dones, indices, weights