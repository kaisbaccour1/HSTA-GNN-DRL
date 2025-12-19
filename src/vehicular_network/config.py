"""Configuration constants for vehicular network simulation"""

# Service specifications
SERVICES = [
    'cooperative_perception', 
    'platooning_control', 
    'edge_object_recognition',
    'predictive_collision_avoidance',
    'infrastructure_vision'
]

SERVICE_SPECS = {
    'cooperative_perception': {'cpu': 15, 'ram': 10, 'data_size': 2},
    'platooning_control': {'cpu': 7, 'ram': 5, 'data_size': 1.6},
    'edge_object_recognition': {'cpu': 5, 'ram': 3, 'data_size': 1},
    'predictive_collision_avoidance': {'cpu': 10, 'ram': 7, 'data_size': 1.8},
    'infrastructure_vision': {'cpu': 3, 'ram': 2, 'data_size': 0.6},
    'none': {'cpu': 0, 'ram': 0, 'data_size': 0}
}

# Vehicle types
VEHICLE_TYPES = {
    "car": {"length": 4.3, "max_speed": 50, "accel": 2.6, "decel": 4.5},
    "truck": {"length": 12.0, "max_speed": 40, "accel": 1.3, "decel": 3.5},
    "bus": {"length": 14.0, "max_speed": 36, "accel": 1.2, "decel": 3.0},
    "motorcycle": {"length": 2.5, "max_speed": 60, "accel": 3.0, "decel": 6.0},
    "emergency": {"length": 6.0, "max_speed": 70, "accel": 3.5, "decel": 6.5}
}

# Simulation parameters
DEFAULT_NUM_EDGES = 7
DEFAULT_TARGET_VEHICLES = 20
DEFAULT_TIME_STEPS = 200
CONNECTION_RADIUS = 150 
CLOUD_DISTANCE = 1500  