"""Configuration for service placement system"""
from dataclasses import dataclass
from typing import List

@dataclass
class ServicePlacementConfig:
    # GNN Prediction
    n_hid: int = 128
    time_window: int = 5
    hidden_dim: int = 512
    epochs: int = 10
    lr: float = 1e-4
    n_services: int = 5
    batch_size: int = 32
    
    # RL Config
    actor_lr: float = 2e-5
    critic_lr: float = 1e-4
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 1.0
    gamma: float = 0.95
    tau: float = 0.02
    buffer_size: int = 100000
    epsilon: float = 0.9
    epsilon_decay: float = 0.97
    min_epsilon: float = 0.1
    
    # Experiment
    n_repetitions: int = 30
    n_episodes: int = 100
    reward_types: List[str] = None
    
    def __post_init__(self):
        if self.reward_types is None:
            self.reward_types = ['energy_only', 'latency_only', 'energy_latency_combined']

# Service specifications
SERVICE_SPECS = {
    'cooperative_perception': {'cpu': 15, 'ram': 10, 'data_size': 2, 'default_ttl': 5},
    'platooning_control': {'cpu': 7, 'ram': 5, 'data_size': 1.6, 'default_ttl': 4},
    'edge_object_recognition': {'cpu': 5, 'ram': 3, 'data_size': 1, 'default_ttl': 3},
    'predictive_collision_avoidance': {'cpu': 10, 'ram': 7, 'data_size': 1.8, 'default_ttl': 3},
    'infrastructure_vision': {'cpu': 3, 'ram': 2, 'data_size': 0.6, 'default_ttl': 3}
}

SERVICE_NAMES = list(SERVICE_SPECS.keys())