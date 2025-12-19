"""Visualization functions for graphs and statistics"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict
import dgl


def visualize_snapshot(g: dgl.DGLGraph, snapshot_id: int):
    """Visualize a snapshot with NetworkX"""
    plt.figure(figsize=(16, 12))
    nx_g = nx.Graph()

    vehicle_type_colors = {
        0: 'red',      # Car
        1: 'blue',     # Truck
        2: 'green',    # Bus
        3: 'orange',   # Motorcycle
        4: 'purple',   # Emergency vehicle
        'edge': 'gray',
        'cloud': 'black'
    }
    
    service_colors = {
        0: 'lightcoral',    # cooperative_perception
        1: 'lightblue',     # platooning_control  
        2: 'lightgreen',    # edge_object_recognition
        3: 'gold',          # predictive_collision_avoidance
        4: 'violet',        # infrastructure_vision
        5: 'white'          # none
    }

    # Add vehicle nodes
    if 'vehicle' in g.ntypes and 'vehicle_type' in g.nodes['vehicle'].data:
        positions = g.nodes['vehicle'].data['position'].cpu().numpy()
        vehicle_types = g.nodes['vehicle'].data['vehicle_type'].cpu().numpy()
        service_types = g.nodes['vehicle'].data['type_service'].cpu().numpy()
        vehicle_ids = g.nodes['vehicle'].data['id'].cpu().numpy()
        
        for i in range(g.number_of_nodes('vehicle')):
            node_id = f"V{int(vehicle_ids[i])}"
            vehicle_type = int(vehicle_types[i])
            service_type = int(service_types[i])
            
            # Color based on service
            color = service_colors.get(service_type, 'white')
            edge_color = vehicle_type_colors.get(vehicle_type, 'pink')
            
            nx_g.add_node(node_id, 
                        pos=positions[i], 
                        ntype='vehicle',
                        vehicle_type=vehicle_type,
                        service_type=service_type,
                        color=color,
                        edge_color=edge_color,
                        size=200 if service_type == 5 else 250)
    
    # Add edge nodes
    if 'edge' in g.ntypes:
        positions = g.nodes['edge'].data['position'].cpu().numpy()
        edge_ids = g.nodes['edge'].data['id'].cpu().numpy()
        
        for i in range(g.number_of_nodes('edge')):
            if i == g.number_of_nodes('edge') - 1:
                node_id = "Cloud"
                color = vehicle_type_colors['cloud']
                size = 500
            else:
                node_id = f"E{int(edge_ids[i])}"
                color = vehicle_type_colors['edge']
                size = 350
            
            nx_g.add_node(node_id, 
                        pos=positions[i], 
                        ntype='edge',
                        color=color,
                        size=size)

    # Add edges
    for etype in g.etypes:
        if 'vehicle' in g.ntypes:
            vehicle_ids = g.nodes['vehicle'].data['id'].cpu().numpy()
        if 'edge' in g.ntypes:
            edge_ids = g.nodes['edge'].data['id'].cpu().numpy()

        src, dst = g.edges(etype=etype)
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()

        for i in range(len(src)):
            src_type, _, dst_type = g.to_canonical_etype(etype)

            if src_type == 'vehicle':
                src_id = f"V{vehicle_ids[src[i]]}"
            else:
                if src[i] == g.number_of_nodes('edge') - 1:
                    src_id = "Cloud"
                else:
                    src_id = f"E{edge_ids[src[i]]}"

            if dst_type == 'vehicle':
                dst_id = f"V{vehicle_ids[dst[i]]}"
            else:
                if dst[i] == g.number_of_nodes('edge') - 1:
                    dst_id = "Cloud"
                else:
                    dst_id = f"E{edge_ids[dst[i]]}"

            if src_id in nx_g and dst_id in nx_g:
                nx_g.add_edge(src_id, dst_id, etype=etype)

    # Draw graph
    if len(nx_g.nodes()) == 0:
        print("No nodes to display")
        plt.close()
        return
    
    pos = {node: data.get('pos', (0, 0)) for node, data in nx_g.nodes(data=True)}
    colors = [data.get('color', 'pink') for node, data in nx_g.nodes(data=True)]
    edge_colors = [data.get('edge_color', 'black') for node, data in nx_g.nodes(data=True)]
    sizes = [data.get('size', 300) for node, data in nx_g.nodes(data=True)]

    # Draw nodes with colored border
    nx.draw_networkx_nodes(nx_g, pos, node_color=colors, node_size=sizes, 
                          edgecolors=edge_colors, linewidths=2, alpha=0.8)
    
    if len(nx_g.nodes()) < 50:
        labels = {node: node for node in nx_g.nodes()}
        nx.draw_networkx_labels(nx_g, pos, labels, font_size=8, font_weight='bold')
    
    nx.draw_networkx_edges(nx_g, pos, alpha=0.3, edge_color='gray', width=1.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markeredgecolor='red', markersize=10, label='Cooperative Perception'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markeredgecolor='blue', markersize=10, label='Platooning Control'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markeredgecolor='green', markersize=10, label='Edge-Assisted Object Recognition'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                  markeredgecolor='orange', markersize=10, label='Predictive Collision Avoidance'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='violet', 
                  markeredgecolor='purple', markersize=10, label='Infrastructure-Assisted Vision'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                  markeredgecolor='black', markersize=10, label='No service'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label='Edge'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                  markersize=10, label='Cloud')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.title(f"Snapshot {snapshot_id} - Manhattan with Services\n(V=Vehicle, E=Edge, Color=interior=service, border=vehicle type)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Statistics
    num_with_service = sum(1 for node, data in nx_g.nodes(data=True) 
                          if data.get('ntype') == 'vehicle' and data.get('service_type', 5) != 5)
    
    print(f"\nSnapshot {snapshot_id}:")
    print(f"Nodes: {len(nx_g.nodes())}, Edges: {len(nx_g.edges())}")
    print(f"Vehicles: {g.number_of_nodes('vehicle')}, Edges: {g.number_of_nodes('edge')}")
    print(f"Vehicles with service: {num_with_service}/{g.number_of_nodes('vehicle')}")


def visualize_features(glist: List[dgl.DGLGraph]):
    """Visualize features for all snapshots"""
    for t, g in enumerate(glist):
        print(f"\n{'='*50}")
        print(f"=== SNAPSHOT {t} ===")
        print(f"{'='*50}")
        
        # Edge features
        if 'edge' in g.ntypes and g.number_of_nodes('edge') > 0:
            edge_data = _extract_features(g.nodes['edge'].data)
            edge_df = pd.DataFrame(edge_data)
            edge_df.index.name = 'Edge_ID'
            print("\n📡 EDGE NODES FEATURES:")
            print(edge_df)
            
            # Additional statistics for edges
            print(f"\n📈 Edge Statistics - Snapshot {t}:")
            print(f"Number of edges: {g.number_of_nodes('edge')}")
            if 'cpu_available' in g.nodes['edge'].data:
                cpu_avail = g.nodes['edge'].data['cpu_available'].numpy()
                ram_avail = g.nodes['edge'].data['ram_available'].numpy()
                print(f"Average available CPU: {cpu_avail.mean():.2f}")
                print(f"Average available RAM: {ram_avail.mean():.2f}")
        
        # Vehicle features
        if 'vehicle' in g.ntypes and g.number_of_nodes('vehicle') > 0:
            vehicle_data = _extract_features(g.nodes['vehicle'].data)
            vehicle_df = pd.DataFrame(vehicle_data)
            vehicle_df.index.name = 'Vehicle_ID'
            print("\n🚗 VEHICLE NODES FEATURES:")
            print(vehicle_df)
            
            # Service distribution
            _print_service_distribution(g)
        
        # Connection statistics
        _print_connection_statistics(g, t)


def _extract_features(data_dict: Dict) -> Dict:
    """Extract features from data dictionary"""
    extracted = {}
    for k, v in data_dict.items():
        if v.ndim > 1:
            for i in range(v.shape[1]):
                extracted[f"{k}_{i}"] = v[:, i].numpy()
        else:
            extracted[k] = v.numpy()
    return extracted


def _print_service_distribution(g: dgl.DGLGraph):
    """Print service distribution statistics"""
    services = g.nodes['vehicle'].data['type_service'].numpy()
    unique, counts = np.unique(services, return_counts=True)
    
    service_names = [
        'cooperative_perception', 
        'platooning_control', 
        'edge_object_recognition', 
        'predictive_collision_avoidance', 
        'infrastructure_vision', 
        'none'
    ]
    
    print("\n📊 Vehicle Statistics:")
    print(f"Number of vehicles: {g.number_of_nodes('vehicle')}")
    print("Service distribution:")
    for service_id, count in zip(unique, counts):
        service_name = service_names[service_id] if service_id < len(service_names) else 'unknown'
        print(f"  {service_name}: {count} vehicles")


def _print_connection_statistics(g: dgl.DGLGraph, snapshot_id: int):
    """Print connection statistics"""
    if g.number_of_edges() == 0:
        return
    
    print(f"\n🔗 CONNECTIONS - Snapshot {snapshot_id}:")
    print(f"Total connections: {g.number_of_edges()}")
    
    # Count connections to cloud vs local edges
    if 'edge' in g.ntypes and 'vehicle' in g.ntypes:
        cloud_idx = g.number_of_nodes('edge') - 1
        edge_connections = 0
        cloud_connections = 0
        
        for etype in g.etypes:
            src, dst = g.edges(etype=etype)
            for d in dst.numpy():
                if d == cloud_idx:
                    cloud_connections += 1
                else:
                    edge_connections += 1
        
        print(f"Connections to local edges: {edge_connections}")
        print(f"Connections to cloud: {cloud_connections}")
        if edge_connections + cloud_connections > 0:
            print(f"Local connection rate: {edge_connections/(edge_connections+cloud_connections)*100:.1f}%")