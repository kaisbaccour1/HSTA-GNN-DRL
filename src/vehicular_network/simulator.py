"""Main simulator class orchestrating everything"""
import torch
import dgl
import copy
import numpy as np
from pathlib import Path
from typing import List, Dict
from dgl.data.utils import save_graphs

from .demand_model import MarkovServiceDemand
from .sumo_interface import SUMOInterface
from .vehicle_manager import VehicleManager
from .graph_builder import GraphBuilder
from .config import SERVICES, SERVICE_SPECS, DEFAULT_NUM_EDGES
from .utils import calculate_edge_positions


class VehicularNetworkSimulator:
    def __init__(self, num_edges: int = DEFAULT_NUM_EDGES, 
                 target_vehicles: int = 20, time_steps: int = 200,
                 network_file: str = "data/network/manhattan.net.xml", 
                 use_gui: bool = True,
                 save_snapshots: bool = True, 
                 output_file: str = "data/graphs/manhattan_snapshots.bin"):
        """
        Initialize the vehicular network simulator.
        
        Args:
            num_edges: Number of edge nodes
            target_vehicles: Target number of vehicles in simulation
            time_steps: Number of simulation steps
            network_file: Path to SUMO network file (.net.xml)
            use_gui: Whether to use SUMO GUI
            save_snapshots: Whether to save snapshots to file
            output_file: Path to output DGL graphs file
        """
        self.num_edges = num_edges
        self.time_steps = time_steps
        self.save_snapshots = save_snapshots
        self.output_file = output_file
        self.saved_snapshots = []
        
        # Create output directories if they don't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(network_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if network file exists
        if not Path(network_file).exists():
            raise FileNotFoundError(
                f"Network file not found: {network_file}\n"
                f"Please place your manhattan.net.xml file in data/network/"
            )
        
        # Generate route file path (same directory as network file)
        network_dir = Path(network_file).parent
        route_file = network_dir / "manhattan.rou.xml"
        
        print(f"📁 Network file: {network_file}")
        print(f"📁 Route file: {route_file}")
        print(f"📁 Output file: {output_file}")
        
        # Initialize components
        self.demand_model = MarkovServiceDemand(SERVICES)
        self.sumo_interface = SUMOInterface(str(network_file), str(route_file), use_gui)
        
        if not self.sumo_interface.load_network():
            raise ValueError(f"Could not load network file: {network_file}")
        
        # Generate routes if needed
        if not route_file.exists():
            print("🔄 Generating routes for Manhattan network...")
            self.sumo_interface.generate_routes(
                target_vehicles,
                ["car", "truck", "bus", "motorcycle", "emergency"],
                [0.65, 0.15, 0.08, 0.10, 0.02]
            )
        else:
            print(f"✅ Using existing route file: {route_file}")
        
        if not self.sumo_interface.start_simulation():
            raise RuntimeError("Failed to start SUMO simulation")
        
        self.vehicle_manager = VehicleManager(
            services=SERVICES,
            service_specs=SERVICE_SPECS,
            target_vehicles=target_vehicles,
            demand_model=self.demand_model,
            sumo_interface=self.sumo_interface
        )
        
        self.graph_builder = GraphBuilder(num_edges, SERVICES)
        
        # Initialize edge state
        self.current_edge_state = self._init_edges()
        self.sumo_net = self.sumo_interface.net
    
    def _init_edges(self) -> Dict[str, torch.Tensor]:
        """Initialize edge features"""
        cpu_capacity = torch.full((self.num_edges + 1,), 25)
        ram_capacity = torch.full((self.num_edges + 1,), 17)
    
        # Cloud with more capacity
        cpu_capacity[-1] = 150
        ram_capacity[-1] = 100
        
        # Get optimal positions for edges
        edge_positions = calculate_edge_positions(self.sumo_net, self.num_edges)
        
        # Get cloud position
        cloud_position = self._get_cloud_position(edge_positions)
        
        # Combine positions
        positions = torch.cat([edge_positions, cloud_position], dim=0)
    
        # Initialize services_hosted and TTL_services_hosted
        services_hosted = torch.zeros(self.num_edges + 1, len(SERVICES))
        TTL_services_hosted = torch.zeros(self.num_edges + 1, len(SERVICES))
        
        # For last edge (cloud), set all services to 1 and TTL to 1000
        services_hosted[-1, :] = torch.ones(len(SERVICES))
        TTL_services_hosted[-1, :] = torch.full((len(SERVICES),), 1000)
    
        return {
            'id': torch.arange(self.num_edges + 1),
            'position': positions,
            'cpu_capacity': cpu_capacity,
            'ram_capacity': ram_capacity,
            'cpu_available': cpu_capacity.clone(),
            'ram_available': ram_capacity.clone(),
            'services_hosted': services_hosted,
            'TTL_services_hosted': TTL_services_hosted
        }
    
    def _get_cloud_position(self, edge_positions: torch.Tensor) -> torch.Tensor:
        """Position cloud at distance to recover uncovered vehicles"""
        # Calculate edge center of gravity
        center_x = torch.mean(edge_positions[:, 0])
        center_y = torch.mean(edge_positions[:, 1])
        
        # Calculate average edge radius
        distances_from_center = torch.norm(
            edge_positions - torch.tensor([center_x, center_y]), dim=1
        )
        mean_radius = torch.mean(distances_from_center)
        
        # Position cloud at 2x average radius
        cloud_distance = mean_radius * 2.0 + 300
        
        # Arbitrary direction (e.g., northeast)
        cloud_x = center_x + cloud_distance * 0.7
        cloud_y = center_y + cloud_distance * 0.7
        
        return torch.tensor([[cloud_x, cloud_y]])
    
    def generate_snapshot(self, t: int) -> dgl.DGLGraph:
        """Generate a single snapshot at time step t"""
        is_first_snapshot = (t == 0)
        
        if self.current_edge_state is None:
            self.current_edge_state = self._init_edges()
        
        # Get vehicle features from SUMO
        vehicle_feats = self.vehicle_manager.update(t)
        num_vehicles = len(vehicle_feats['id'])
        
        if num_vehicles == 0 and t == 0:
            print(f"⚠️  No vehicles at time {t}, skipping...")
            return None
        
        # Update edge services
        updated_edge_feats = self._update_services(
            copy.deepcopy(self.current_edge_state), 
            vehicle_feats, 
            is_first_snapshot
        )
        self.current_edge_state = updated_edge_feats
        
        # Build graph
        g = self.graph_builder.build_graph(vehicle_feats, updated_edge_feats)
        
        if self.save_snapshots and g is not None:
            self.saved_snapshots.append(g)
        
        return g
    
    def _update_services(self, edge_feats: Dict[str, torch.Tensor], 
                        vehicle_feats: Dict[str, torch.Tensor], 
                        is_first_snapshot: bool) -> Dict[str, torch.Tensor]:
        """Update service states on edges"""
        if is_first_snapshot:
            edge_feats['cpu_available'] = edge_feats['cpu_capacity'].clone()
            edge_feats['ram_available'] = edge_feats['ram_capacity'].clone()
        
        # Decrement TTL for services in cloud
        cloud_index = -1
        services_deployed = edge_feats['services_hosted'][cloud_index] > 0
        edge_feats['TTL_services_hosted'][cloud_index, services_deployed] -= 1
        
        return edge_feats
    
    def generate_all_snapshots(self) -> List[dgl.DGLGraph]:
        """Generate all snapshots for the simulation"""
        print(f"⏳ Generating {self.time_steps} snapshots...")
        
        graphs = []
        for t in range(self.time_steps):
            if t % 10 == 0:
                print(f"  Step {t}/{self.time_steps}")
            
            g = self.generate_snapshot(t)
            
            # Skip if no graph returned
            if g is None:
                continue
                
            # Skip first snapshot if no vehicles
            if t == 0 and g.number_of_nodes('vehicle') == 0:
                print("⏭️  Snapshot 0 ignored (no vehicles)")
                continue
                
            graphs.append(g)
        
        print(f"✅ Generated {len(graphs)} valid snapshots")
        return graphs
    
    def save_snapshots(self, custom_output_path: str = None):
        """Save snapshots to file"""
        output_path = custom_output_path or self.output_file
        
        if not self.save_snapshots or not self.saved_snapshots:
            print("❌ No snapshots to save")
            return
        
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            save_graphs(output_path, self.saved_snapshots)
            print(f"✅ {len(self.saved_snapshots)} snapshots saved in {output_path}")
            
            # Statistics
            total_vehicles = sum(g.number_of_nodes('vehicle') for g in self.saved_snapshots)
            total_edges = sum(g.number_of_edges() for g in self.saved_snapshots)
            
            print(f"📊 Statistics:")
            print(f"  • Snapshots: {len(self.saved_snapshots)}")
            print(f"  • Total vehicles: {total_vehicles}")
            print(f"  • Total connections: {total_edges}")
            print(f"  • Average vehicles/snapshot: {total_vehicles/len(self.saved_snapshots):.1f}")
            
            # Save metadata
            self._save_metadata(output_path)
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            raise
    
    def _save_metadata(self, output_path: str):
        """Save metadata about the generated graphs"""
        import json
        from datetime import datetime
        
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'num_snapshots': len(self.saved_snapshots),
            'simulation_params': {
                'num_edges': self.num_edges,
                'time_steps': self.time_steps,
                'services': SERVICES
            },
            'file_info': {
                'output_path': str(output_path),
                'graph_format': 'DGL binary',
                'size_mb': Path(output_path).stat().st_size / (1024 * 1024) if Path(output_path).exists() else 0
            }
        }
        
        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata saved: {metadata_path}")
    
    def get_snapshot_statistics(self, snapshot_idx: int = None) -> Dict:
        """Get statistics about the generated snapshots"""
        if not self.saved_snapshots:
            return {}
        
        if snapshot_idx is not None:
            g = self.saved_snapshots[snapshot_idx]
            return {
                'snapshot_id': snapshot_idx,
                'num_vehicles': g.number_of_nodes('vehicle'),
                'num_edges': g.number_of_nodes('edge'),
                'num_connections': g.number_of_edges(),
            }
        
        # Overall statistics
        stats = {
            'total_snapshots': len(self.saved_snapshots),
            'vehicles_per_snapshot': [],
            'connections_per_snapshot': [],
        }
        
        for g in self.saved_snapshots:
            stats['vehicles_per_snapshot'].append(g.number_of_nodes('vehicle'))
            stats['connections_per_snapshot'].append(g.number_of_edges())
        
        stats['avg_vehicles'] = np.mean(stats['vehicles_per_snapshot'])
        stats['avg_connections'] = np.mean(stats['connections_per_snapshot'])
        
        return stats
    
    def visualize_snapshot(self, snapshot_idx: int = 0):
        """Visualize a specific snapshot"""
        if not self.saved_snapshots or snapshot_idx >= len(self.saved_snapshots):
            print(f"❌ Snapshot {snapshot_idx} not available")
            return
        
        from .visualization import visualize_snapshot
        g = self.saved_snapshots[snapshot_idx]
        visualize_snapshot(g, snapshot_idx)
    
    def close(self):
        """Clean up resources and close SUMO connection"""
        print("🔌 Closing simulator...")
        self.sumo_interface.close()
        print("✅ Simulator closed")
