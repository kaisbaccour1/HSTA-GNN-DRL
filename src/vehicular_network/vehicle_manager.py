"""Vehicle management with SUMO integration"""
import torch
import random
import traci
from typing import Dict, List
from .demand_model import MarkovServiceDemand
from .config import SERVICE_SPECS, VEHICLE_TYPES, SERVICES


class VehicleManager:
    def __init__(self, services: List[str], service_specs: Dict[str, Dict], 
                 target_vehicles: int, demand_model: MarkovServiceDemand,
                 sumo_interface):
        
        self.services = services
        self.service_specs = service_specs
        self.target_vehicles = target_vehicles
        self.demand_model = demand_model
        self.sumo = sumo_interface.sumo
        self.net = sumo_interface.net
        self.current_vehicles = {}
        self.vehicle_counter = 0
        self.first_appearance_timestamps = {}
        self.service_durations = {}
        self.vehicle_types = ["car", "truck", "bus", "motorcycle", "emergency"]
    
    def update(self, t: int) -> Dict[str, torch.Tensor]:
        """Update simulation and return vehicle features"""
        if self.sumo is None:
            return self._empty_tensor_dict()
        
        try:
            self.sumo.simulationStep()
            
            # Vehicle count control
            self._control_vehicle_count()
            
            # Update vehicles with realistic services
            updated_vehicles = self._update_vehicles(t)
            
            self.current_vehicles = updated_vehicles
            return self._to_tensor_format()
            
        except Exception as e:
            print(f"❌ Error during SUMO update: {e}")
            return self._empty_tensor_dict()
    
    def _control_vehicle_count(self):
        """Control number of vehicles"""
        current_count = len(self.sumo.vehicle.getIDList())
        if current_count < self.target_vehicles:
            self._add_vehicles(self.target_vehicles - current_count)
        elif current_count > self.target_vehicles:
            self._remove_vehicles(current_count - self.target_vehicles)
    
    def _update_vehicles(self, t: int) -> Dict[str, Dict]:
        """Update all vehicles with service management"""
        updated_vehicles = {}
        
        for vid in self.sumo.vehicle.getIDList():
            try:
                vehicle_data = self._get_vehicle_data(vid)
                
                if vid in self.current_vehicles:
                    # Existing vehicle
                    vehicle_info = self._update_existing_vehicle(vid, t, vehicle_data)
                else:
                    # New vehicle
                    vehicle_info = self._initialize_new_vehicle(vid, t, vehicle_data)
                
                updated_vehicles[vid] = vehicle_info
                
            except traci.TraCIException:
                continue
        
        return updated_vehicles
    
    def _get_vehicle_data(self, vid: str) -> Dict:
        """Extract vehicle data from SUMO"""
        pos = self.sumo.vehicle.getPosition(vid)
        speed = self.sumo.vehicle.getSpeed(vid)
        
        return {
            'position': pos,
            'speed': speed,
            'road_id': self.sumo.vehicle.getRoadID(vid),
            'vehicle_type': self.sumo.vehicle.getTypeID(vid)
        }
    
    def _update_existing_vehicle(self, vid: str, t: int, vehicle_data: Dict) -> Dict:
        """Update existing vehicle's service state"""
        current_service = self.current_vehicles[vid]['service']
        
        if vid in self.service_durations and self.service_durations[vid] > 0:
            # Active service, decrement duration
            self.service_durations[vid] -= 1
            service = current_service
        else:
            # Service expired, Markov transition
            service = self.demand_model.get_next_service(vid, vehicle_data)
            if service != 'none':
                self.service_durations[vid] = self.demand_model.get_service_duration(service)
            else:
                self.service_durations[vid] = 0
        
        return self._create_vehicle_info(vid, service, t, vehicle_data)
    
    def _initialize_new_vehicle(self, vid: str, t: int, vehicle_data: Dict) -> Dict:
        """Initialize new vehicle with Markov model"""
        self.first_appearance_timestamps[vid] = t
        service = self.demand_model.initialize_vehicle(vid, vehicle_data)
        
        if service != 'none':
            self.service_durations[vid] = self.demand_model.get_service_duration(service)
        else:
            self.service_durations[vid] = 0
        
        return self._create_vehicle_info(vid, service, t, vehicle_data)
    
    def _create_vehicle_info(self, vid: str, service: str, t: int, vehicle_data: Dict) -> Dict:
        """Create vehicle information dictionary"""
        x, y = vehicle_data['position']
        
        return {
            'id': hash(vid) % 1000000,
            'sumo_id': vid,
            'position': torch.tensor([x, y], dtype=torch.float),
            'service': service,
            'cpu_demand': self.service_specs[service]['cpu'] if service != 'none' else 0,
            'ram_demand': self.service_specs[service]['ram'] if service != 'none' else 0,
            'data_size': self.service_specs[service]['data_size'] if service != 'none' else 0,
            'speed': vehicle_data['speed'],
            'vehicle_type': vehicle_data['vehicle_type'],
            'road_id': vehicle_data['road_id'],
            'timestamp': t,
            'timestamp_apparition': self.first_appearance_timestamps.get(vid, t),
            'service_remaining_duration': self.service_durations.get(vid, 0)
        }
    
    def _add_vehicles(self, num_to_add: int):
        """Add vehicles to simulation"""
        if self.net is None:
            return
            
        edges = [edge.getID() for edge in self.net.getEdges() 
                if not edge.getID().startswith(':')]
        if not edges:
            return
            
        for i in range(num_to_add):
            try:
                start_edge = random.choice(edges)
                end_edge = random.choice(edges)
                
                route_id = f"dynamic_route_{self.vehicle_counter}"
                veh_id = f"dynamic_veh_{self.vehicle_counter}"
                chosen_type = random.choice(self.vehicle_types)
                
                # Create simple route
                self.sumo.route.add(route_id, [start_edge, end_edge])
                self.sumo.vehicle.add(veh_id, route_id, typeID=chosen_type, depart="now")
                
                self.vehicle_counter += 1
                
            except Exception as e:
                print(f"⚠️ Error adding vehicle {i}: {e}")
                continue
    
    def _remove_vehicles(self, num_to_remove: int):
        """Remove vehicles from simulation"""
        vehicles = self.sumo.vehicle.getIDList()
        if vehicles:
            vehicles_to_remove = random.sample(vehicles, min(num_to_remove, len(vehicles)))
            for vid in vehicles_to_remove:
                try:
                    self.sumo.vehicle.remove(vid)
                    if vid in self.current_vehicles:
                        del self.current_vehicles[vid]
                    if vid in self.service_durations:
                        del self.service_durations[vid]
                except:
                    continue
    
    def _to_tensor_format(self) -> Dict[str, torch.Tensor]:
        """Convert vehicles to tensor format"""
        if not self.current_vehicles:
            return self._empty_tensor_dict()
        
        all_vehicle_types = ["car", "truck", "bus", "motorcycle", "emergency"]
        all_services = self.services + ['none']
        
        return {
            'id': torch.tensor([v['id'] for v in self.current_vehicles.values()]),
            'position': torch.stack([v['position'] for v in self.current_vehicles.values()]),
            'type_service': torch.tensor([
                all_services.index(v['service']) for v in self.current_vehicles.values()
            ]),
            'vehicle_type': torch.tensor([
                all_vehicle_types.index(v['vehicle_type']) for v in self.current_vehicles.values()
            ]), 
            'cpu_demand': torch.tensor([v['cpu_demand'] for v in self.current_vehicles.values()]),
            'ram_demand': torch.tensor([v['ram_demand'] for v in self.current_vehicles.values()]),
            'data_size': torch.tensor([v['data_size'] for v in self.current_vehicles.values()]),
            'speed': torch.tensor([v['speed'] for v in self.current_vehicles.values()]),
            'timestamp_apparition': torch.tensor([
                v['timestamp_apparition'] for v in self.current_vehicles.values()
            ]),
            'timestamp': torch.tensor([v['timestamp'] for v in self.current_vehicles.values()]),
            'service_remaining_duration': torch.tensor([
                v['service_remaining_duration'] for v in self.current_vehicles.values()
            ])
        }
    
    def _empty_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """Return empty tensor dictionary"""
        return {
            'id': torch.tensor([], dtype=torch.long),
            'position': torch.tensor([], dtype=torch.float).reshape(0, 2),
            'type_service': torch.tensor([], dtype=torch.long),
            'vehicle_type': torch.tensor([], dtype=torch.long),
            'cpu_demand': torch.tensor([], dtype=torch.float),
            'ram_demand': torch.tensor([], dtype=torch.float),
            'data_size': torch.tensor([], dtype=torch.float),
            'speed': torch.tensor([], dtype=torch.float),
            'timestamp_apparition': torch.tensor([], dtype=torch.long),
            'timestamp': torch.tensor([], dtype=torch.long),
            'service_remaining_duration': torch.tensor([], dtype=torch.long)
        }