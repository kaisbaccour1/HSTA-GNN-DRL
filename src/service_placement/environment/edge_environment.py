"""Environment for RL training with edge nodes"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..config import ServicePlacementConfig
from ..models.comparator import ModelComparator
from ..models.gnn_predictor import TemporalAttentionPredictor
from .edge_node import EdgeNode
from ..service_specs import ServiceSpecs
import dgl


class EdgeEnvironment:
    def __init__(self, comparator: ModelComparator, prediction_model: TemporalAttentionPredictor, 
                 config: ServicePlacementConfig):
        self.comparator = comparator
        self.prediction_model = prediction_model
        self.prediction_model.eval()
        self.config = config
        self.device = comparator.device
        self.service_specs = ServiceSpecs()
        self.n_services = config.n_services
        
        # Energy coefficients
        self.energy_coeffs = {'alpha': 0.2, 'beta': 0.3}
        self.startup_energy_multiplier = 1.5
        
        # State tracking
        self.current_step = 0
        self.episode_length = 0
        self.edge_nodes: Dict[int, EdgeNode] = {}
        self.cloud_edge_id = 0
        
        # Metrics
        self.episode_energy = 0
        self.episode_latency = 0
        
        # Cache for analysis results
        self._last_edge_analysis = {}
        self._last_cloud_analysis = None
        self._last_total_energy = 0
        
        # Snapshot generation
        self.snapshot_generator = None
        self.last_episode_analysis = []
        
        # Previous state
        self.previous_services_state = {}
        
        # Agent
        self.agent = None
    
    def set_agent(self, agent):
        self.agent = agent
        if hasattr(agent, 'reward_type'):
            self.reward_type = agent.reward_type
        else:
            self.reward_type = 'combined'
    
    def enable_snapshot_generation(self, n_edges: int, service_names: List[str], 
                                 snapshot_dir: str = "episode_snapshots"):
        """Enable snapshot generation with custom directory"""
        from ..utils.snapshot_generator import StepSnapshot
        self.snapshot_generator = StepSnapshot(n_edges, service_names, snapshot_dir)
        self.last_episode_analysis = []
    
    def take_step_snapshot(self) -> Dict:
        """Take snapshot of current step"""
        analysis = self.get_detailed_step_analysis()
        self.last_episode_analysis.append(analysis)
        
        if self.snapshot_generator:
            filename = self.snapshot_generator.create_step_snapshot(self.current_step, analysis)
        
        return analysis
    
    def reset(self, scenario_idx: int = 0, use_eval_set: bool = False) -> Optional[Dict]:
        """Reset environment for new episode"""
        self.current_step = 0
        
        # Select dataset
        if use_eval_set:
            scenario_data = self.comparator.eval_scenarios[scenario_idx]
        else:
            scenario_data = self.comparator.train_scenarios[scenario_idx]
        
        self.episode_length = len(scenario_data) - self.config.time_window - 1
        self.edge_nodes = {}
        
        if scenario_data:
            initial_graph = scenario_data[0]
            self._initialize_edge_nodes(initial_graph)
        
        state = self.get_state()
        
        # Reset previous state
        self.previous_services_state = {}
        for edge_id, edge_node in self.edge_nodes.items():
            self.previous_services_state[edge_id] = edge_node.services_hosted.clone()
        
        # Reset cache
        self._last_edge_analysis = {}
        self._last_cloud_analysis = None
        self._last_total_energy = 0
        
        # Reset metrics
        self.episode_energy = 0
        self.episode_latency = 0
        
        return state
    
    def _initialize_edge_nodes(self, graph: dgl.DGLGraph):
        """Initialize edge nodes from graph"""
        edge_data = graph.nodes['edge'].data
        
        for i in range(graph.number_of_nodes('edge')):
            edge_id = int(edge_data['id'][i].item())
            
            initial_state = {
                'cpu_capacity': edge_data['cpu_capacity'][i].float(),
                'ram_capacity': edge_data['ram_capacity'][i].float(),
                'cpu_available': edge_data['cpu_capacity'][i].float(),
                'ram_available': edge_data['ram_capacity'][i].float(),
                'services_hosted': torch.zeros(self.n_services, dtype=torch.int),
                'TTL_services_hosted': torch.zeros(self.n_services, dtype=torch.int),
                'position': edge_data['position'][i].float()
            }
            
            self.edge_nodes[edge_id] = EdgeNode(
                edge_id, initial_state, self.service_specs
            )
        
        # Last edge is the cloud
        self.cloud_edge_id = len(self.edge_nodes) - 1
    
    def get_state(self) -> Optional[Dict]:
        """Get current state of the environment"""
        if self.current_step >= len(self.comparator.train_scenarios[0]) - self.config.time_window:
            return None
        
        if not self.edge_nodes:
            current_graph = self.comparator.train_scenarios[0][self.current_step]
            self._initialize_edge_nodes(current_graph)
        
        # Get time window of graphs
        start_idx = self.current_step
        end_idx = start_idx + self.config.time_window
        graphs_window = self.comparator.train_scenarios[0][start_idx:end_idx]
        
        # Prepare graphs
        prepared_graphs = [self.comparator.prepare_features(g) for g in graphs_window]
        
        # Get predictions from GNN
        with torch.no_grad():
            service_logits, state_embeddings = self.prediction_model(prepared_graphs)
            predicted_services = (torch.sigmoid(service_logits) > 0.8).float()
        
        # Collect current resource information
        current_resources = []
        for edge_id, edge_node in self.edge_nodes.items():
            state_dict = edge_node.get_state_dict()
            resources = [
                state_dict['available_cpu'] / max(state_dict['cpu_capacity'], 1.0),
                state_dict['available_ram'] / max(state_dict['ram_capacity'], 1.0),
                state_dict['utilization_rate']
            ]
            current_resources.extend(resources)
        
        current_resources = np.array(current_resources, dtype=np.float32)
        predicted_services_np = predicted_services.cpu().numpy().flatten()
        
        # Combine into state vector
        state_vector = np.concatenate([predicted_services_np, current_resources])
        
        return {
            'state_vector': state_vector,
            'predicted_services': predicted_services,
            'edge_states': {eid: node.get_state_dict() for eid, node in self.edge_nodes.items()}
        }
    
    def step(self, actions: Dict[int, int]) -> Tuple[Optional[Dict], float, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        if self.current_step >= self.episode_length:
            return None, 0, True, {}
        
        # Update TTL for all edges
        for edge_node in self.edge_nodes.values():
            edge_node.update_ttl(self.current_step)
        
        # Get current state
        current_state = self.get_state()
        if current_state is None:
            return None, 0, True, {}
        
        predicted_services = current_state['predicted_services']
        
        # Apply actions
        deployment_results = self._apply_actions_with_predictions(actions, predicted_services)
        
        # Calculate metrics
        total_latency = self._calculate_total_latency()
        total_energy, energy_breakdown = self._get_total_energy_from_analysis()
        
        # Calculate reward
        reward = self._calculate_reward(total_energy, total_latency)
        
        # Update episode metrics
        self.episode_energy += total_energy
        self.episode_latency += total_latency
        
        # Get next state
        next_state = self.get_state()
        
        return next_state, reward, False, {
            'latency': total_latency,
            'total_energy': total_energy,
            'energy_breakdown': energy_breakdown,
            'deployment_results': deployment_results
        }
    
    def _apply_actions_with_predictions(self, actions: Dict[int, int], 
                                       predicted_services: torch.Tensor) -> Dict[int, List[int]]:
        """Apply actions considering predicted services"""
        results = {}
        
        for edge_id, action in actions.items():
            if edge_id not in self.edge_nodes:
                continue
                
            edge_node = self.edge_nodes[edge_id]
            
            # Find edge index in predictions
            edge_indices = [i for i, (eid, _) in enumerate(self.get_state()['edge_states'].items()) 
                          if eid == edge_id]
            
            if edge_indices:
                edge_pred_idx = edge_indices[0]
                if edge_pred_idx < predicted_services.shape[0]:
                    edge_predictions = predicted_services[edge_pred_idx]
                    services_to_deploy = self.agent.action_to_services(action, edge_predictions)
                else:
                    services_to_deploy = []
            else:
                services_to_deploy = []
            
            # Deploy services
            deployed_services = []
            for service_idx in services_to_deploy:
                if edge_node.deploy_service(service_idx, self.current_step):
                    deployed_services.append(service_idx)
            
            results[edge_id] = deployed_services
        
        return results
    
    def _calculate_total_latency(self) -> float:
        """Calculate total latency for current step"""
        current_graph_idx = self.current_step + self.config.time_window
        if current_graph_idx >= len(self.comparator.train_scenarios[0]):
            return 0
        
        current_graph = self.comparator.train_scenarios[0][current_graph_idx]
        total_latency = 0
        n_vehicles = 0
        
        src, dst = current_graph.edges(etype=('vehicle','connects','edge'))
        
        for v_idx in range(current_graph.number_of_nodes('vehicle')):
            service_type = int(current_graph.nodes['vehicle'].data['type_service'][v_idx])
            if service_type >= self.n_services:
                continue

            n_vehicles += 1
            served = False
            connected_edges = dst[src == v_idx]
            
            best_edge_idx = None
            min_distance = float('inf')
            
            for edge_idx in connected_edges:
                edge_id = int(current_graph.nodes['edge'].data['id'][edge_idx].item())
                
                if edge_id in self.edge_nodes:
                    edge_node = self.edge_nodes[edge_id]
                    if (edge_node.services_hosted[service_type] == 1 and
                        edge_node.available_cpu >= 0 and edge_node.available_ram >= 0):
                        
                        distance = current_graph.edges['connects'].data['distance'][edge_idx].item()
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_edge_idx = edge_idx
            
            if best_edge_idx is not None:
                distance = current_graph.edges['connects'].data['distance'][best_edge_idx]
                data_size = self.service_specs.get_requirements(service_type)['data_size']
                bandwidth = current_graph.edges['connects'].data['bandwidth'][best_edge_idx]
                
                latency = self.calculate_transmission_latency(distance, data_size, bandwidth)
                total_latency += latency
                served = True
            
            if not served:
                data_size = self.service_specs.get_requirements(service_type)['data_size']
                cloud_distance = 500
                cloud_bandwidth = 15
                
                cloud_latency = self.calculate_transmission_latency(
                    cloud_distance, data_size, cloud_bandwidth
                )
                total_latency += cloud_latency
        
        return total_latency / n_vehicles if n_vehicles > 0 else 0
    
    def _get_total_energy_from_analysis(self) -> Tuple[float, Dict]:
        """Get total energy from analysis results"""
        total_edge_energy = 0
        edge_breakdown = {}
        
        # Analyze each edge
        for edge_id, edge_node in self.edge_nodes.items():
            if edge_id == self.cloud_edge_id:
                continue
                
            state_dict = edge_node.get_state_dict()
            edge_analysis = self._analyze_edge_state(edge_id, state_dict)
            self._last_edge_analysis[edge_id] = edge_analysis
            
            total_edge_energy += edge_analysis['energy_consumed']
            edge_breakdown[edge_id] = {
                'energy': edge_analysis['energy_consumed'],
                'useful_cpu': edge_analysis['energy_breakdown']['useful_cpu'],
                'idle_cpu': edge_analysis['energy_breakdown']['idle_cpu'],
                'startup_penalty': edge_analysis['energy_breakdown']['startup_penalty']
            }
        
        # Analyze cloud
        cloud_analysis = self._analyze_cloud_state()
        self._last_cloud_analysis = cloud_analysis
        
        # Calculate total energy
        total_energy = total_edge_energy + cloud_analysis['energy_consumed']
        self._last_total_energy = total_energy
        
        # Update previous state
        for edge_id, edge_node in self.edge_nodes.items():
            self.previous_services_state[edge_id] = edge_node.services_hosted.clone()
        
        # Return breakdown
        energy_breakdown = {
            'edges': edge_breakdown,
            'cloud': {
                'energy': cloud_analysis['energy_consumed'],
                'useful_cpu': cloud_analysis['useful_cpu'],
                'wasted_cpu': cloud_analysis['wasted_cpu']
            },
            'total': total_energy,
            'edge_total': total_edge_energy,
            'cloud_total': cloud_analysis['energy_consumed']
        }
        
        return total_energy, energy_breakdown
    
    def calculate_transmission_latency(self, distance: float, data_size: float, 
                                      bandwidth: float) -> float:
        """Calculate transmission latency"""
        propagation_speed = 200000  # m/s
        propagation_latency = distance / propagation_speed
        
        data_size_bits = data_size * 8 * 1e6  # Convert MB to bits
        transmission_latency = data_size_bits / (bandwidth * 1000000)  # bandwidth in Mbps
        
        total_latency = propagation_latency + transmission_latency
        return total_latency
    
    def _calculate_reward(self, energy: float, latency: float) -> float:
        """Calculate reward based on agent type"""
        if self.agent is None:
            return self._default_reward(energy, latency)
        return self.agent.calculate_reward(energy, latency)
    
    def _default_reward(self, energy: float, latency: float) -> float:
        """Default combined reward"""
        energy_min, energy_max = 0, 2753.2
        latency_min, latency_max = 0, 0.8
        
        energy_norm = (energy - energy_min) / (energy_max - energy_min)
        latency_norm = (latency - latency_min) / (latency_max - latency_min)
        
        energy_component = -energy_norm
        latency_component = -latency_norm
        
        return 0.6 * energy_component + 0.4 * latency_component
    
    def _analyze_edge_state(self, edge_id: int, state_dict: Dict) -> Dict:
        """Analyze edge state for energy calculation"""
        edge_node = self.edge_nodes[edge_id]
        cpu_capacity = state_dict['cpu_capacity']
        
        # Maximum dynamic energy
        max_dynamic_energy = (
            self.energy_coeffs['alpha'] * cpu_capacity + 
            self.energy_coeffs['beta'] * cpu_capacity ** 2
        )
        
        # Static energy = 50% of max
        energy_static = 0.5 * max_dynamic_energy
        
        # Calculate used vs idle CPU + startup energy
        useful_cpu = 0
        idle_cpu = 0
        startup_energy_penalty = 0
        used_services_list = []
        unused_services_list = []
        
        current_services = state_dict['services_hosted']
        previous_services = self.previous_services_state.get(
            edge_id, 
            torch.zeros_like(current_services)
        )
        
        for service_idx in range(self.n_services):
            service_name = self.service_specs.service_names[service_idx]
            if current_services[service_idx] == 1:
                service_req = self.service_specs.get_requirements(service_idx)
                service_cpu = service_req['cpu']
                
                is_new_service = (
                    current_services[service_idx] == 1 and 
                    previous_services[service_idx] == 0
                )
                
                service_info = {
                    'name': service_name,
                    'cpu': service_cpu,
                    'ram': service_req['ram'],
                    'ttl': state_dict['TTL_services_hosted'][service_idx].item(),
                    'is_new': is_new_service
                }
                
                if self._is_service_used(edge_id, service_idx):
                    used_services_list.append(service_info)
                    useful_cpu += service_cpu
                else:
                    unused_services_list.append(service_info)
                    idle_cpu += service_cpu
                
                if is_new_service:
                    startup_energy = self.energy_coeffs['alpha'] * service_cpu
                    startup_energy_penalty += startup_energy * self.startup_energy_multiplier
        
        useful_energy = (
            self.energy_coeffs['alpha'] * useful_cpu + 
            self.energy_coeffs['beta'] * useful_cpu ** 2
        )
        
        idle_penalty = 0.6 * (
            self.energy_coeffs['alpha'] * idle_cpu + 
            self.energy_coeffs['beta'] * idle_cpu ** 2
        )
        
        energy_dynamic_with_idle = useful_energy + idle_penalty
        edge_energy_with_startup = energy_static + energy_dynamic_with_idle + startup_energy_penalty
        
        total_cpu_used = useful_cpu + idle_cpu
        utilization_rate = total_cpu_used / cpu_capacity if cpu_capacity > 0 else 0
        
        return {
            'edge_id': edge_id,
            'utilization_rate': utilization_rate,
            'energy_consumed': edge_energy_with_startup,
            'energy_breakdown': {
                'static': energy_static,
                'dynamic': energy_dynamic_with_idle,
                'dynamic_useful': useful_energy,
                'dynamic_idle': idle_penalty,
                'startup_penalty': startup_energy_penalty,
                'useful_cpu': useful_cpu,
                'idle_cpu': idle_cpu,
                'max_dynamic_energy': max_dynamic_energy,
            },
            'cpu_available': state_dict['available_cpu'],
            'cpu_capacity': cpu_capacity,
            'ram_available': state_dict['available_ram'],
            'ram_capacity': state_dict['ram_capacity'],
            'used_services': used_services_list,
            'unused_services': unused_services_list,
            'total_services': len(used_services_list) + len(unused_services_list),
            'used_services_count': len(used_services_list),
            'unused_services_count': len(unused_services_list),
            'new_services_count': sum(1 for s in used_services_list + unused_services_list if s['is_new']),
            'position': state_dict['position']
        }
    
    def _analyze_cloud_state(self) -> Dict:
        """Analyze cloud state for energy calculation"""
        current_graph_idx = self.current_step + self.config.time_window
        cloud_services_used = set()
        cloud_served_vehicles = 0
        
        if current_graph_idx < len(self.comparator.train_scenarios[0]):
            current_graph = self.comparator.train_scenarios[0][current_graph_idx]
            
            for v_idx in range(current_graph.number_of_nodes('vehicle')):
                service_type = int(current_graph.nodes['vehicle'].data['type_service'][v_idx])
                if service_type >= self.n_services:
                    continue
                
                served_by_edge = False
                src, dst = current_graph.edges(etype=('vehicle','connects','edge'))
                connected_edges = dst[src == v_idx]
                
                for edge_idx in connected_edges:
                    edge_id = int(current_graph.nodes['edge'].data['id'][edge_idx].item())
                    if edge_id in self.edge_nodes:
                        edge_node = self.edge_nodes[edge_id]
                        if (edge_node.services_hosted[service_type] == 1 and
                            edge_node.available_cpu >= 0 and edge_node.available_ram >= 0):
                            served_by_edge = True
                            break

                if not served_by_edge:
                    service_name = self.service_specs.service_names[service_type]
                    cloud_served_vehicles += 1
                    cloud_services_used.add(service_name)
        
        # Cloud energy calculation
        total_cloud_cpu = sum([
            specs['cpu'] for specs in self.service_specs.specs.values()
        ])
        
        max_dynamic_energy_cloud = (
            self.energy_coeffs['alpha'] * total_cloud_cpu + 
            self.energy_coeffs['beta'] * total_cloud_cpu ** 2
        )
        
        cloud_static_energy = 0.5 * max_dynamic_energy_cloud
        
        useful_cpu, wasted_cpu = 0.0, 0.0
        
        for service_idx in range(self.n_services):
            service_name = self.service_specs.service_names[service_idx]
            cpu = self.service_specs.get_requirements(service_idx)['cpu']
            
            if service_name in cloud_services_used:
                useful_cpu += cpu
            else:
                wasted_cpu += cpu
        
        useful_energy = (
            self.energy_coeffs['alpha'] * useful_cpu + 
            self.energy_coeffs['beta'] * useful_cpu ** 2
        )
        
        waste_penalty = 0.6 * (
            self.energy_coeffs['alpha'] * wasted_cpu + 
            self.energy_coeffs['beta'] * wasted_cpu ** 2
        )
        
        cloud_energy = cloud_static_energy + useful_energy + waste_penalty
        
        return {
            'cloud_id': self.cloud_edge_id,
            'energy_consumed': cloud_energy,
            'vehicles_served': cloud_served_vehicles,
            'services_used': list(cloud_services_used),
            'services_unused': [s for s in self.service_specs.service_names 
                              if s not in cloud_services_used],
            'useful_cpu': useful_cpu,
            'wasted_cpu': wasted_cpu,
            'useful_energy': useful_energy,
            'waste_penalty': waste_penalty,
            'base_energy': cloud_static_energy,
            'max_dynamic_energy': max_dynamic_energy_cloud,
        }
    
    def _is_service_used(self, edge_id: int, service_idx: int) -> bool:
        """Check if a service is used"""
        current_graph_idx = self.current_step + self.config.time_window
        if current_graph_idx >= len(self.comparator.train_scenarios[0]):
            return False
        
        current_graph = self.comparator.train_scenarios[0][current_graph_idx]
        
        edge_indices = torch.where(current_graph.nodes['edge'].data['id'] == edge_id)[0]
        if len(edge_indices) == 0:
            return False
        
        edge_idx = edge_indices[0]
        
        src, dst = current_graph.edges(etype=('vehicle','connects','edge'))
        connected_vehicles = src[dst == edge_idx]
        
        for v_idx in connected_vehicles:
            vehicle_service = int(current_graph.nodes['vehicle'].data['type_service'][v_idx])
            if vehicle_service == service_idx:
                return True
        
        return False
    
    def get_detailed_step_analysis(self) -> Dict:
        """Get detailed analysis of current step"""
        analysis = {
            'step': self.current_step,
            'edges': {},
            'cloud': {},
            'summary': {}
        }
        
        total_edge_energy = 0
        total_edge_utilization = 0
        edge_count = 0
        
        # Use cache if available, otherwise calculate
        if not self._last_edge_analysis:
            for edge_id, edge_node in self.edge_nodes.items():
                if edge_id == self.cloud_edge_id:
                    continue
                    
                state_dict = edge_node.get_state_dict()
                edge_analysis = self._analyze_edge_state(edge_id, state_dict)
                self._last_edge_analysis[edge_id] = edge_analysis
        
        # Retrieve from cache
        for edge_id, edge_analysis in self._last_edge_analysis.items():
            analysis['edges'][edge_id] = edge_analysis
            total_edge_energy += edge_analysis['energy_consumed']
            total_edge_utilization += edge_analysis['utilization_rate']
            edge_count += 1
        
        # Use cloud cache if available
        if self._last_cloud_analysis is None:
            self._last_cloud_analysis = self._analyze_cloud_state()
        
        analysis['cloud'] = self._last_cloud_analysis
        
        # Calculate total energy from cache
        if self._last_total_energy == 0:
            self._last_total_energy = total_edge_energy + self._last_cloud_analysis['energy_consumed']
        
        analysis['summary'] = {
            'total_energy': self._last_total_energy,
            'edge_energy': total_edge_energy,
            'cloud_energy': self._last_cloud_analysis['energy_consumed'],
            'avg_edge_utilization': total_edge_utilization / max(edge_count, 1),
            'total_edges': edge_count,
            'timestamp': self.current_step
        }
        
        return analysis
    
    def get_episode_metrics(self, step_count: int) -> Dict:
        """Get episode metrics"""
        if step_count == 0:
            return {}
        
        return {
            'avg_energy_per_step': self.episode_energy / step_count,
            'avg_latency_per_step': self.episode_latency / step_count,
            'total_energy': self.episode_energy,
            'total_latency': self.episode_latency,
            'total_steps': step_count
        }
    
    def calculate_energy_bounds(self) -> float:
        """Calculate maximum possible energy"""
        max_energy = 0
        
        for edge_id, edge_node in self.edge_nodes.items():
            if edge_id == self.cloud_edge_id:
                continue
                
            cpu_capacity = edge_node.cpu_capacity
            
            max_dynamic_energy = (
                self.energy_coeffs['alpha'] * cpu_capacity + 
                self.energy_coeffs['beta'] * cpu_capacity ** 2
            )
            
            energy_static = 0.5 * max_dynamic_energy
            edge_energy = energy_static + max_dynamic_energy
            max_energy += edge_energy

        total_cloud_cpu = sum([
            specs['cpu'] for specs in self.service_specs.specs.values()
        ])
        
        max_dynamic_energy_cloud = (
            self.energy_coeffs['alpha'] * total_cloud_cpu + 
            self.energy_coeffs['beta'] * total_cloud_cpu ** 2
        )
        
        cloud_static_energy = 0.5 * max_dynamic_energy_cloud
        
        useful_cpu = total_cloud_cpu
        useful_energy = (
            self.energy_coeffs['alpha'] * useful_cpu + 
            self.energy_coeffs['beta'] * useful_cpu ** 2
        )
        
        cloud_energy = cloud_static_energy + useful_energy
        total_max_energy = max_energy + cloud_energy
        
        return total_max_energy
    
    def calculate_latency_bounds(self) -> float:
        """Calculate average latency bound"""
        total_latency = 0
        snapshot_count = 0
        
        for scenario in self.comparator.train_scenarios:
            for graph in scenario:
                snapshot_latency = 0
                n_vehicles = graph.number_of_nodes('vehicle')
                vehicles_counted = 0
                
                if n_vehicles == 0:
                    continue
                    
                for v_idx in range(n_vehicles):
                    service_type = int(graph.nodes['vehicle'].data['type_service'][v_idx])
                    if service_type >= self.n_services:
                        continue
                    
                    data_size = self.service_specs.get_requirements(service_type)['data_size']
                    vehicle_latency = self.calculate_transmission_latency(
                        distance=500, 
                        data_size=data_size, 
                        bandwidth=15
                    )
                    snapshot_latency += vehicle_latency
                    vehicles_counted += 1
                
                if vehicles_counted > 0:
                    total_latency += (snapshot_latency / vehicles_counted)
                    snapshot_count += 1
        
        if snapshot_count == 0:
            return 0
        
        average_latency = total_latency / snapshot_count
        return average_latency