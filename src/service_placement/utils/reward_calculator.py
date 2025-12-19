"""Reward calculation for different strategies"""
import numpy as np
from typing import Dict


class RewardCalculator:
    """Calculates different types of rewards"""
    
    # Energy bounds (from your configuration)
    ENERGY_MIN = 0
    ENERGY_MAX = 2753.2
    
    # Latency bounds (from your configuration)
    LATENCY_MIN = 0
    LATENCY_MAX = 0.8
    
    @classmethod
    def energy_only_reward(cls, energy: float, latency: float) -> float:
        """Reward based only on energy (minimization)"""
        energy_norm = cls._normalize_energy(energy)
        return -energy_norm  # Negative because lower energy is better
    
    @classmethod
    def latency_only_reward(cls, energy: float, latency: float) -> float:
        """Reward based only on latency (minimization)"""
        latency_norm = cls._normalize_latency(latency)
        return -latency_norm  # Negative because lower latency is better
    
    @classmethod
    def energy_latency_combined_reward(cls, energy: float, latency: float, 
                                      energy_weight: float = 0.6, 
                                      latency_weight: float = 0.4) -> float:
        """Combined reward with weights"""
        energy_norm = cls._normalize_energy(energy)
        latency_norm = cls._normalize_latency(latency)
        
        # Weighted combination (both minimized)
        combined = (energy_weight * (-energy_norm)) + (latency_weight * (-latency_norm))
        return combined
    
    @classmethod
    def _normalize_energy(cls, energy: float) -> float:
        """Normalize energy to [0, 1] range"""
        if cls.ENERGY_MAX == cls.ENERGY_MIN:
            return 0.0
        return (energy - cls.ENERGY_MIN) / (cls.ENERGY_MAX - cls.ENERGY_MIN)
    
    @classmethod
    def _normalize_latency(cls, latency: float) -> float:
        """Normalize latency to [0, 1] range"""
        if cls.LATENCY_MAX == cls.LATENCY_MIN:
            return 0.0
        return (latency - cls.LATENCY_MIN) / (cls.LATENCY_MAX - cls.LATENCY_MIN)
    
    @classmethod
    def calculate_all_rewards(cls, energy: float, latency: float) -> Dict[str, float]:
        """Calculate all reward types"""
        return {
            'energy_only': cls.energy_only_reward(energy, latency),
            'latency_only': cls.latency_only_reward(energy, latency),
            'combined': cls.energy_latency_combined_reward(energy, latency)
        }
    
    @classmethod
    def update_bounds(cls, energy_samples: list, latency_samples: list):
        """Update bounds based on observed data"""
        if energy_samples:
            cls.ENERGY_MAX = max(energy_samples)
            cls.ENERGY_MIN = min(energy_samples)
        
        if latency_samples:
            cls.LATENCY_MAX = max(latency_samples)
            cls.LATENCY_MIN = min(latency_samples)