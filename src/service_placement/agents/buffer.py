"""Experience replay buffer for RL"""
import random
from collections import deque, namedtuple
from typing import List, Optional
import numpy as np


Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add transition to buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch of transitions"""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)
    
    def clear(self) -> None:
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        if not self.buffer:
            return {}
        
        rewards = [t.reward for t in self.buffer]
        dones = [t.done for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'fullness': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'min_reward': min(rewards) if rewards else 0,
            'done_ratio': sum(dones) / len(dones) if dones else 0
        }