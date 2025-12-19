"""DGL graph construction from vehicle and edge data"""
import torch
import dgl
import copy
from typing import Dict
from .config import SERVICES, CONNECTION_RADIUS


class GraphBuilder:
    def __init__(self, num_edges: int, services: List[str] = None):
        self.num_edges = num_edges
        self.services = services or SERVICES
    
    def build_graph(self, vehicle_features: Dict[str, torch.Tensor], 
                   edge_features: Dict[str, torch.Tensor]) -> dgl.DGLGraph:
        """Build heterograph from vehicle and edge features"""
        num_vehicles = len(vehicle_features['id'])
        cloud_idx = self.num_edges  # Cloud is last node
        
        # Calculate connections
        dist_matrix = torch.cdist(vehicle_features['position'], 
                                 edge_features['position'][:-1])
        connections = (dist_matrix < CONNECTION_RADIUS).nonzero(as_tuple=False)
        
        # Identify unconnected vehicles
        connected_vehicles = set(connections[:, 0].tolist())
        all_vehicles = set(range(num_vehicles))
        unconnected_vehicles = list(all_vehicles - connected_vehicles)
        
        # Create edge lists
        cloud_src = torch.tensor(unconnected_vehicles, dtype=torch.long)
        cloud_dst = torch.full((len(unconnected_vehicles),), cloud_idx, dtype=torch.long)
        
        if len(connections) > 0:
            src = torch.cat([connections[:, 0], cloud_src])
            dst = torch.cat([connections[:, 1], cloud_dst])
        else:
            src = cloud_src
            dst = cloud_dst
        
        # Create heterograph
        g = dgl.heterograph({
            ('vehicle', 'connects', 'edge'): (src, dst)
        }, num_nodes_dict={
            'vehicle': num_vehicles,
            'edge': self.num_edges + 1
        })
        
        # Add node features
        g.nodes['edge'].data.update(edge_features)
        g.nodes['vehicle'].data.update(vehicle_features)
        
        # Add edge features
        self._add_edge_features(g, connections, cloud_src, cloud_dst, dist_matrix)
        
        return g
    
    def _add_edge_features(self, g: dgl.DGLGraph, connections: torch.Tensor,
                          cloud_src: torch.Tensor, cloud_dst: torch.Tensor,
                          dist_matrix: torch.Tensor):
        """Add features to edges"""
        if g.number_of_edges('connects') == 0:
            return
        
        edge_features_list = []
        
        # Normal connections
        if len(connections) > 0:
            normal_src = connections[:, 0]
            normal_dst = connections[:, 1]
            normal_distances = dist_matrix[normal_src, normal_dst]
            
            edge_features_list.append({
                'src': normal_src,
                'dst': normal_dst,
                'bandwidth': torch.full((len(connections),), 100.0),
                'distance': normal_distances,
                'type': torch.zeros(len(connections))  # 0 for normal
            })
        
        # Cloud connections
        if len(cloud_src) > 0:
            edge_features_list.append({
                'src': cloud_src,
                'dst': cloud_dst,
                'bandwidth': torch.full((len(cloud_src),), 15.0),
                'distance': torch.full((len(cloud_src),), 1500.0),
                'type': torch.ones(len(cloud_src))  # 1 for cloud
            })
        
        # Combine all features
        all_src = torch.cat([feat['src'] for feat in edge_features_list])
        all_dst = torch.cat([feat['dst'] for feat in edge_features_list])
        all_bandwidth = torch.cat([feat['bandwidth'] for feat in edge_features_list])
        all_distance = torch.cat([feat['distance'] for feat in edge_features_list])
        all_type = torch.cat([feat['type'] for feat in edge_features_list])
        
        # Assign edge features
        g.edges['connects'].data.update({
            'bandwidth': all_bandwidth,
            'distance': all_distance,
            'type': all_type
        })