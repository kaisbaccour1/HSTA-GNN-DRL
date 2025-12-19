"""Actor and Critic networks for RL agent"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, action_dim),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities"""
        x = self.network(state)
        x = torch.clamp(x, min=-10, max=10)
        action_probs = F.softmax(x, dim=-1)
        action_probs = (action_probs + 1e-8) / (1.0 + 1e-8 * action_probs.size(-1))
        return action_probs
    
    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw logits (before softmax)"""
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)