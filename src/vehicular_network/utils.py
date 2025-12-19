"""Utility functions"""
import torch
import random
import numpy as np
from typing import List, Tuple
import sumolib


def calculate_edge_positions(sumo_net, num_edges: int) -> torch.Tensor:
    """Calculate optimal edge positions from SUMO network"""
    # Get all edges from SUMO network
    sumo_edges = list(sumo_net.getEdges())
    
    # Extract center positions of SUMO edges
    edge_positions = []
    for edge in sumo_edges:
        # Get edge shape (list of points)
        shape = edge.getShape()
        if len(shape) > 0:
            # Calculate edge center
            x_coords = [point[0] for point in shape]
            y_coords = [point[1] for point in shape]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            edge_positions.append((center_x, center_y))
    
    # If not enough edges found, use node coordinates
    if len(edge_positions) < num_edges:
        nodes = list(sumo_net.getNodes())
        for node in nodes:
            coord = node.getCoord()
            if coord:
                edge_positions.append((coord[0], coord[1]))
    
    # Use simplified k-means to find optimal positions
    if len(edge_positions) >= num_edges:
        # Select most spaced positions
        selected_positions = select_dispersed_positions(edge_positions, num_edges)
    else:
        # Fallback: circular positions around network center
        selected_positions = get_circular_positions(sumo_net, num_edges)
    
    return torch.tensor(selected_positions, dtype=torch.float)


def select_dispersed_positions(positions: List[Tuple[float, float]], 
                              num_to_select: int) -> List[Tuple[float, float]]:
    """Select most dispersed positions"""
    if len(positions) <= num_to_select:
        return positions
    
    # Simple method: select most distant positions
    selected = [random.choice(positions)]
    
    for _ in range(1, num_to_select):
        max_min_distance = -1
        best_candidate = None
        
        for candidate in positions:
            if candidate in selected:
                continue
            
            # Calculate minimum distance to already selected positions
            min_dist = min(distance(candidate, sel) for sel in selected)
            
            if min_dist > max_min_distance:
                max_min_distance = min_dist
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
    
    return selected


def get_circular_positions(sumo_net, num_positions: int) -> List[Tuple[float, float]]:
    """Circular positions around network center (fallback)"""
    # Calculate network center
    nodes = list(sumo_net.getNodes())
    if nodes:
        coords = [node.getCoord() for node in nodes if node.getCoord()]
        if coords:
            center_x = sum(coord[0] for coord in coords) / len(coords)
            center_y = sum(coord[1] for coord in coords) / len(coords)
        else:
            center_x, center_y = 500, 500
    else:
        center_x, center_y = 500, 500
        
    radius = 400  # Larger radius for Manhattan
    
    positions = []
    for i in range(num_positions):
        angle = 2 * np.pi * i / num_positions
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions.append((x, y))
    
    return positions


def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_local_demand(vehicle_feats: Dict[str, torch.Tensor], 
                          edge_pos: torch.Tensor, 
                          services: List[str]) -> torch.Tensor:
    """Calculate local demand around an edge (only for vehicles with service)"""
    distances = torch.norm(vehicle_feats['position'] - edge_pos, dim=1)
    nearby_vehicles = distances < 400  # Larger radius for Manhattan
    
    if not nearby_vehicles.any():
        return torch.zeros(len(services))
    
    # Filter only vehicles with service (exclude 'none')
    service_vehicles = vehicle_feats['type_service'] < len(services)
    valid_nearby = nearby_vehicles & service_vehicles
    
    if not valid_nearby.any():
        return torch.zeros(len(services))
    
    local_service_counts = torch.bincount(
        vehicle_feats['type_service'][valid_nearby],
        minlength=len(services)
    )
    
    return local_service_counts.float() / max(1, valid_nearby.sum())