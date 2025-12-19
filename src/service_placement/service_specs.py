"""Service specifications and requirements"""
from typing import Dict
from .config import SERVICE_SPECS, SERVICE_NAMES


class ServiceSpecs:
    def __init__(self, specs: Dict = None):
        self.specs = specs or SERVICE_SPECS
        self.service_names = SERVICE_NAMES
    
    def get_requirements(self, service_idx: int) -> Dict:
        """Get requirements for a service by index"""
        if service_idx < 0 or service_idx >= len(self.service_names):
            raise ValueError(f"Invalid service index: {service_idx}")
        
        service_name = self.service_names[service_idx]
        return self.specs[service_name]
    
    def get_service_name(self, service_idx: int) -> str:
        """Get service name by index"""
        if service_idx < 0 or service_idx >= len(self.service_names):
            raise ValueError(f"Invalid service index: {service_idx}")
        return self.service_names[service_idx]
    
    @property
    def n_services(self) -> int:
        return len(self.service_names)