"""Dummy agent for non-RL strategies"""
import numpy as np
import random
from typing import Optional
import torch
from ..config import ServicePlacementConfig


class DummyAgent:
    def __init__(self, config: ServicePlacementConfig, device, reward_type: str = 'dummy'):
        self.config = config
        self.device = device
        self.action_dim = 2 ** config.n_services
        self.reward_type = reward_type
        self.n_services = config.n_services
    
    def action_to_services(self, action: int, edge_predictions: Optional[torch.Tensor] = None) -> list:
        """Convert action to list of service indices"""
        services = []
        for i in range(self.n_services):
            if action & (1 << i):
                if edge_predictions is None or edge_predictions[i] == 1:
                    services.append(i)
        return services
    
    def calculate_reward(self, energy: float, latency: float) -> float:
        """Dummy reward calculation for compatibility"""
        return 0.0
    
    def get_random_action(self) -> int:
        """Generate random action"""
        return random.randint(0, self.action_dim - 1)
    
    def get_zero_action(self) -> int:
        """Zero deployment action"""
        return 0
    
    def get_prediction_follower_action(self, edge_predictions: torch.Tensor) -> int:
        """Action based on predictions"""
        if edge_predictions is None:
            return 0
        
        action = 0
        for i in range(min(self.n_services, len(edge_predictions))):
            if edge_predictions[i] > 0.5:
                action |= (1 << i)
        return action