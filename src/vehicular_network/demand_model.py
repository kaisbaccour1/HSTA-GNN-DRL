"""Markov model for service demand generation"""
import numpy as np
import random
from typing import List, Dict
from .config import SERVICES, VEHICLE_TYPES


class MarkovServiceDemand:
    def __init__(self, services: List[str] = None):
        self.services = services or SERVICES
        self.services_with_none = self.services + ['none']
        self.transition_matrix = self._create_transition_matrix()
        self.current_services = {}  # Current state per vehicle
        self.poisson_triggered = set()  # Vehicles that already triggered Poisson
        
    def _create_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Transition matrix between services (including 'none')"""
        return {
            'none': {
                'cooperative_perception': 0.15, 
                'platooning_control': 0.25, 
                'edge_object_recognition': 0.08, 
                'predictive_collision_avoidance': 0.07, 
                'infrastructure_vision': 0.05, 
                'none': 0.4
            },
            # ... (le reste de la matrice comme dans votre code)
        }
    
    def initialize_vehicle(self, vid: str, vehicle_data: Dict) -> str:
        """Initialize a vehicle with Poisson distribution (only once)"""
        if vid not in self.poisson_triggered:
            self.poisson_triggered.add(vid)
            
            # Poisson distribution to decide initial service
            k = np.random.poisson(0.5)
            if k >= 1:
                initial_service = self._get_initial_service(vehicle_data)
                self.current_services[vid] = initial_service
                return initial_service
            else:
                self.current_services[vid] = 'none'
                return 'none'
        return self.current_services.get(vid, 'none')
    
    def _get_initial_service(self, vehicle_data: Dict) -> str:
        """Determine initial service based on vehicle type"""
        vehicle_type = vehicle_data.get('vehicle_type', 'car')
        
        initial_probs = {
            'car': {
                'cooperative_perception': 0.3, 
                'platooning_control': 0.4, 
                'edge_object_recognition': 0.15, 
                'predictive_collision_avoidance': 0.1, 
                'infrastructure_vision': 0.05
            },
            # ... (autres types)
        }
        
        probs = initial_probs.get(vehicle_type, initial_probs['car'])
        return random.choices(list(probs.keys()), weights=list(probs.values()))[0]
    
    def get_next_service(self, vid: str, vehicle_data: Dict) -> str:
        """Markov transition to next service"""
        current = self.current_services.get(vid, 'none')
        next_probs = self.transition_matrix[current].copy()
        next_probs = self._adjust_for_context(next_probs, vehicle_data)
        
        next_service = random.choices(
            list(next_probs.keys()), 
            weights=list(next_probs.values())
        )[0]
        
        self.current_services[vid] = next_service
        return next_service
    
    def _adjust_for_context(self, probabilities: Dict[str, float], vehicle_data: Dict) -> Dict[str, float]:
        """Adjust probabilities based on vehicle context"""
        adjusted = probabilities.copy()
        vehicle_type = vehicle_data.get('vehicle_type', 'car')
        speed = vehicle_data.get('speed', 0)
        
        # Adjustments based on vehicle type
        if vehicle_type == 'emergency':
            adjusted['infrastructure_vision'] *= 2.5
            adjusted['predictive_collision_avoidance'] *= 1.5
        elif vehicle_type in ['truck', 'bus']:
            adjusted['predictive_collision_avoidance'] *= 2.0
            adjusted['platooning_control'] *= 1.3
            
        # Adjustments based on speed
        if speed < 10:  # Traffic jam
            adjusted['cooperative_perception'] *= 1.8
            adjusted['platooning_control'] *= 0.7
        elif speed > 60:  # Highway
            adjusted['platooning_control'] *= 1.5
            adjusted['predictive_collision_avoidance'] *= 1.3
            
        # Renormalize
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}
    
    def get_service_duration(self, service: str) -> int:
        """Service lifetime duration"""
        base_durations = {
            'cooperative_perception': np.random.exponential(1/0.08),
            'platooning_control': np.random.exponential(1/0.12),
            'edge_object_recognition': np.random.exponential(1/0.05),
            'predictive_collision_avoidance': np.random.exponential(1/0.1),
            'infrastructure_vision': np.random.exponential(1/0.15),
            'none': np.random.exponential(1/0.05)
        }
        return max(1, int(base_durations.get(service, 10)))