"""Edge node with service deployment and TTL management"""
import torch
import numpy as np
from typing import Dict, List
from ..config import ServicePlacementConfig
from ..service_specs import ServiceSpecs


class EdgeNode:
    def __init__(self, edge_id: int, initial_state: Dict, service_specs: ServiceSpecs):
        self.edge_id = edge_id
        self.service_specs = service_specs
        self.n_services = 5
        
        # Initialize state from initial_state dict
        self.cpu_capacity = float(initial_state['cpu_capacity'])
        self.ram_capacity = float(initial_state['ram_capacity'])
        self.available_cpu = float(initial_state['cpu_available'])
        self.available_ram = float(initial_state['ram_available'])
        self.services_hosted = initial_state['services_hosted'].clone()
        self.service_ttl = initial_state['TTL_services_hosted'].clone()
        self.position = initial_state['position']
        
        # Utilization
        self.utilization_rate = 0.0
        self._update_utilization()
        
        # Tracking
        self.total_deployed_services = 0
        self.total_expired_services = 0
    
    def _update_utilization(self) -> None:
        """Update resource utilization rate"""
        cpu_used = self.cpu_capacity - self.available_cpu
        ram_used = self.ram_capacity - self.available_ram
        
        if self.cpu_capacity > 0 and self.ram_capacity > 0:
            cpu_util = cpu_used / self.cpu_capacity
            ram_util = ram_used / self.ram_capacity
            self.utilization_rate = (cpu_util + ram_util) / 2
        else:
            self.utilization_rate = 0.0
    
    def update_ttl(self, timestep: int) -> List[int]:
        """Decrement TTL and free expired services"""
        expired_mask = (self.service_ttl <= 1) & (self.services_hosted == 1)
        
        services_freed = []
        for service_idx in range(self.n_services):
            if expired_mask[service_idx]:
                self._free_service(service_idx)
                services_freed.append(service_idx)
                self.total_expired_services += 1
        
        # Decrement TTL for active services
        active_mask = (self.services_hosted == 1)
        self.service_ttl[active_mask] -= 1
        
        # Force utilization update
        self._update_utilization()
        
        # Validation check
        if self.utilization_rate > 0 and torch.sum(self.services_hosted) == 0:
            print(f"🚨 BUG DETECTED @ t={timestep}: Utilization {self.utilization_rate:.1%} but NO service!")
            print(f"   CPU: {self.available_cpu}/{self.cpu_capacity}")
            print(f"   RAM: {self.available_ram}/{self.ram_capacity}")
        
        return services_freed
    
    def _free_service(self, service_idx: int) -> None:
        """Free resources of an expired service"""
        service_req = self.service_specs.get_requirements(service_idx)
        
        # Free resources
        self.available_cpu += service_req['cpu']
        self.available_ram += service_req['ram']
        
        # Ensure we don't exceed capacity
        self.available_cpu = min(self.available_cpu, self.cpu_capacity)
        self.available_ram = min(self.available_ram, self.ram_capacity)
        
        # Reset service state
        self.services_hosted[service_idx] = 0
        self.service_ttl[service_idx] = 0
        
        # Immediate utilization update
        self._update_utilization()
    
    def deploy_service(self, service_idx: int, timestep: int) -> bool:
        """Deploy a service if resources are available"""
        if self.services_hosted[service_idx] == 1:
            return False  # Service already deployed
        
        service_req = self.service_specs.get_requirements(service_idx)
        
        # Check resource availability
        if (self.available_cpu >= service_req['cpu'] and 
            self.available_ram >= service_req['ram']):
            
            # Allocate resources
            self.available_cpu -= service_req['cpu']
            self.available_ram -= service_req['ram']
            
            # Update service state
            self.services_hosted[service_idx] = 1
            self.service_ttl[service_idx] = service_req['default_ttl']
            
            # Update utilization
            self._update_utilization()
            
            # Update tracking
            self.total_deployed_services += 1
            
            return True
        else:
            return False
    
    def can_deploy_service(self, service_idx: int) -> bool:
        """Check if service can be deployed"""
        if self.services_hosted[service_idx] == 1:
            return False
        
        service_req = self.service_specs.get_requirements(service_idx)
        return (self.available_cpu >= service_req['cpu'] and 
                self.available_ram >= service_req['ram'])
    
    def get_state_dict(self) -> Dict:
        """Get current state as dictionary"""
        return {
            'id': self.edge_id,
            'cpu_capacity': self.cpu_capacity,
            'ram_capacity': self.ram_capacity,
            'available_cpu': self.available_cpu,
            'available_ram': self.available_ram,
            'services_hosted': self.services_hosted.clone(),
            'TTL_services_hosted': self.service_ttl.clone(),
            'position': self.position,
            'utilization_rate': self.utilization_rate,
            'total_deployed': self.total_deployed_services,
            'total_expired': self.total_expired_services
        }
    
    def get_utilization_stats(self) -> Dict:
        """Get detailed utilization statistics"""
        return {
            'cpu_used': self.cpu_capacity - self.available_cpu,
            'cpu_capacity': self.cpu_capacity,
            'cpu_utilization': (self.cpu_capacity - self.available_cpu) / self.cpu_capacity if self.cpu_capacity > 0 else 0,
            'ram_used': self.ram_capacity - self.available_ram,
            'ram_capacity': self.ram_capacity,
            'ram_utilization': (self.ram_capacity - self.available_ram) / self.ram_capacity if self.ram_capacity > 0 else 0,
            'services_active': int(torch.sum(self.services_hosted)),
            'services_ttl': self.service_ttl.tolist()
        }