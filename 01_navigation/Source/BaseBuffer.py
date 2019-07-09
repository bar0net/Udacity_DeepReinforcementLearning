from collections import deque
from abc import ABC, abstractmethod
import random

class BaseBuffer(ABC):
    # Fixed-size container to store experience tuples
    
    def __init__(self, buffer_size, batch_size, seed):
        # Initialize class
        #
        # Params:
        # @ buffer_size: max number of items in container
        # @ batch_size: number of items returned by sample
        
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.type = None
        
    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        # Add new experience to memory
        pass
        
    @abstractmethod
    def sample(self, alpha=0, beta=0):
        # Returns a collection of random items in memory
        pass
    
    def active(self):
        # Returns if the container have enough items to sample
        return self.batch_size <= len(self.memory)
    
    def __len__(self):
        # Returns the current size of the container
        return len(self.memory)