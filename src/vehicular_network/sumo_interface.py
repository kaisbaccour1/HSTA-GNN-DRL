"""SUMΟ interface for network and route management"""
import os
import random
import subprocess
from typing import List
import sumolib
import traci


class SUMOInterface:
    def __init__(self, network_file: str, route_file: str, use_gui: bool = True):
        self.network_file = network_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.net = None
        self.sumo = None
        
    def load_network(self) -> bool:
        """Load existing network file"""
        if not os.path.exists(self.network_file):
            print(f"❌ Network file {self.network_file} not found")
            return False
        
        try:
            self.net = sumolib.net.readNet(self.network_file)
            print(f"✅ Network loaded: {len(self.net.getNodes())} nodes, {len(self.net.getEdges())} edges")
            return True
        except Exception as e:
            print(f"❌ Error loading network: {e}")
            return False
    
    def generate_routes(self, num_vehicles: int, vehicle_types: List[str], 
                       type_probabilities: List[float]) -> bool:
        """Generate routes for existing network"""
        if self.net is None:
            print("❌ Network not loaded")
            return False
        
        routes_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" length="4.3" maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.5" color="yellow"/>
    <vType id="truck" length="12.0" maxSpeed="11.11" accel="1.3" decel="3.5" sigma="0.7" color="red"/>
    <vType id="bus" length="14.0" maxSpeed="10.0" accel="1.2" decel="3.0" sigma="0.8" color="blue"/>
    <vType id="motorcycle" length="2.5" maxSpeed="16.67" accel="3.0" decel="6.0" sigma="0.3" color="green"/>
    <vType id="emergency" length="6.0" maxSpeed="19.44" accel="3.5" decel="6.5" sigma="0.2" color="white"/>
"""
        
        # Get edges from network
        edges = [edge.getID() for edge in self.net.getEdges() 
                if not edge.getID().startswith(':')]
        
        print(f"🔍 Network has {len(edges)} available edges")
        
        # Generate routes
        departure_times = sorted([random.uniform(0, 500) for _ in range(num_vehicles)])
        
        for i, depart in enumerate(departure_times):
            route_edges = self._create_manhattan_route(edges)
            vtype = random.choices(vehicle_types, weights=type_probabilities, k=1)[0]
            
            routes_content += f"""
    <route id="route_{i}" edges="{' '.join(route_edges)}"/>
    <vehicle id="veh{i}" type="{vtype}" depart="{depart:.1f}" route="route_{i}"/>"""
        
        routes_content += "\n</routes>"
        
        with open(self.route_file, 'w', encoding='utf-8') as f:
            f.write(routes_content)
        
        print(f"✅ Routes generated: {self.route_file}")
        return True
    
    def _create_manhattan_route(self, edges: List[str]) -> List[str]:
        """Create a typical Manhattan route"""
        try:
            horizontal_edges = []
            vertical_edges = []
            
            for edge_id in edges:
                edge_obj = self.net.getEdge(edge_id)
                from_node = edge_obj.getFromNode()
                to_node = edge_obj.getToNode()
                
                if abs(from_node.getCoord()[0] - to_node.getCoord()[0]) > abs(from_node.getCoord()[1] - to_node.getCoord()[1]):
                    horizontal_edges.append(edge_id)
                else:
                    vertical_edges.append(edge_id)
            
            route = []
            if horizontal_edges and vertical_edges:
                route.append(random.choice(horizontal_edges))
                route.append(random.choice(vertical_edges))
                route.append(random.choice(horizontal_edges))
            else:
                route = random.sample(edges, min(3, len(edges)))
            
            return route
        except Exception as e:
            print(f"⚠️ Route creation error: {e}")
            return random.sample(edges, min(3, len(edges)))
    
    def start_simulation(self) -> bool:
        """Start SUMO simulation"""
        config_file = self._create_sumo_config()
        
        try:
            if self.use_gui:
                print("🚗 Starting SUMO-GUI...")
                traci.start(['sumo-gui', '-c', config_file, '--start', '--delay', '100'])
            else:
                print("🚗 Starting SUMO (console mode)...")
                traci.start(['sumo', '-c', config_file])
            
            self.sumo = traci
            print("✅ SUMO started successfully")
            return True
            
        except Exception as e:
            print(f"❌ SUMO error: {e}")
            return False
    
    def _create_sumo_config(self) -> str:
        """Create SUMO configuration file"""
        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{self.network_file}"/>
        <route-files value="{self.route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
    <processing>
        <step-length value="1.0"/>
        <no-step-log value="true"/>
        <ignore-route-errors value="true"/>
        <no-internal-links value="false"/>
    </processing>
    <report>
        <no-warnings value="false"/>
        <verbose value="true"/>
    </report>
</configuration>"""
        
        config_file = "simulation_config.sumocfg"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return config_file
    
    def close(self):
        """Close SUMO connection"""
        try:
            traci.close()
        except:
            pass