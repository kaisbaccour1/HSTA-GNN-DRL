"""Model comparator for feature preparation"""
import torch
import torch.nn.functional as F
import dgl
import numpy as np
from typing import List, Tuple, Generator
from sklearn.metrics import f1_score
from ..config import ServicePlacementConfig


class ModelComparator:
    def __init__(self, scenarios: List[List[dgl.DGLGraph]], config: ServicePlacementConfig):
        self.scenarios = scenarios
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.n_services = 6  # 5 services + 'none'
        self.vehicle_dim = 10 + self.n_services
        self.edge_dim = 5
        
        # Split scenarios into train and eval sets
        self.train_scenarios = []
        self.eval_scenarios = []
        for scenario in scenarios:
            train, eval = self.split_scenario(scenario)
            self.train_scenarios.append(train)
            self.eval_scenarios.append(eval)
    
    def split_scenario(self, graphs: List[dgl.DGLGraph], train_ratio: float = 0.7) -> Tuple[List, List]:
        """Split scenario into train and eval sets"""
        n_total = len(graphs)
        n_train = int(n_total * train_ratio)
        return graphs[:n_train], graphs[n_train:]
    
    def prepare_features(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """Prepare features for GNN"""
        def standardize(tensor):
            mean = tensor.mean()
            std = tensor.std()
            if std > 1e-8:
                return (tensor - mean) / std
            else:
                return tensor * 0
        
        # VEHICLES
        if g.number_of_nodes('vehicle') > 0:
            v_data = g.nodes['vehicle'].data
            features_list = []
            
            # Vehicle ID
            vehicle_id = standardize(v_data['id'].float().unsqueeze(1))
            features_list.append(vehicle_id)
            
            # Position
            pos = v_data['position'].float()
            pos_x = standardize(pos[:, 0:1])
            pos_y = standardize(pos[:, 1:2])
            features_list.extend([pos_x, pos_y])
            
            # Resource demands
            cpu = standardize(v_data['cpu_demand'].float().unsqueeze(1))
            ram = standardize(v_data['ram_demand'].float().unsqueeze(1))
            data_size = standardize(v_data['data_size'].float().unsqueeze(1))
            vehicle_type = standardize(v_data['vehicle_type'].float().unsqueeze(1))
            speed = standardize(v_data['speed'].float().unsqueeze(1))
            duration = standardize(v_data['service_remaining_duration'].float().unsqueeze(1))
            features_list.extend([cpu, ram, data_size, speed, duration, vehicle_type])
            
            # Time features
            time_since = standardize(
                (v_data['timestamp'] - v_data['timestamp_apparition']).float().unsqueeze(1)
            )
            features_list.append(time_since)
            
            # Service one-hot encoding
            service_onehot = F.one_hot(v_data['type_service'].long(), self.n_services).float()
            features_list.append(service_onehot)
            
            # Concatenate all vehicle features
            v_feats = torch.cat(features_list, dim=1)
        else:
            v_feats = torch.zeros(0, 10 + self.n_services, device=self.device)
        
        # EDGES
        e_data = g.nodes['edge'].data
        e_features_list = []
        
        # Edge ID
        edge_id = standardize(e_data['id'].float().unsqueeze(1))
        e_features_list.append(edge_id)
        
        # Position and resource capacities
        for key in ['position', 'cpu_capacity', 'ram_capacity']:
            if key in e_data:
                feat = e_data[key].float()
                if feat.dim() == 1:
                    feat = feat.unsqueeze(1)
                    e_features_list.append(standardize(feat))
                elif feat.dim() == 2 and feat.shape[1] == 2:  # Position
                    pos_x = standardize(feat[:, 0:1])
                    pos_y = standardize(feat[:, 1:2])
                    e_features_list.extend([pos_x, pos_y])
                else:
                    e_features_list.append(standardize(feat))
        
        # Concatenate all edge features
        e_feats = torch.cat(e_features_list, dim=1)
        
        # Edge connection features
        if 'bandwidth' in g.edges['connects'].data:
            edge_feat_list = []
            for key in ['bandwidth', 'distance']:
                if key in g.edges['connects'].data:
                    feat = g.edges['connects'].data[key].float().unsqueeze(1)
                    edge_feat_list.append(standardize(feat))
            
            if edge_feat_list:
                edge_features = torch.cat(edge_feat_list, dim=1)
                g.edges['connects'].data['features'] = edge_features.to(self.device)
        
        # Store features in graph
        g.nodes['vehicle'].data['features'] = v_feats
        g.nodes['edge'].data['features'] = e_feats
        
        return g
    
    def get_true_services(self, g_next: dgl.DGLGraph) -> torch.Tensor:
        """Get true service requirements from graph"""
        num_edges = g_next.number_of_nodes('edge')
        y = torch.zeros(num_edges, self.n_services)
        
        src, dst = g_next.edges(etype=('vehicle','connects','edge'))
        
        # For each edge, check which services are needed by connected vehicles
        for e in range(num_edges):
            connected = src[dst == e]
            for v in connected:
                s = int(g_next.nodes['vehicle'].data['type_service'][v])
                if s < 5:  # Exclude 'none' service (index 5)
                    y[e, s] = 1
        
        return y.float().to(self.device)
    
    def windows_from_scenario(self, graphs: List[dgl.DGLGraph]) -> Generator:
        """Generate sliding windows from scenario"""
        T = len(graphs)
        W = self.cfg.time_window
        
        for start in range(0, T - (W+1) + 1):
            # Get sequence of graphs for time window
            seq = graphs[start : start+W+1]
            
            # Prepare features for each graph
            prepared_seq = [self.prepare_features(g) for g in seq]
            
            # Last graph is the target for prediction
            y_true = self.get_true_services(prepared_seq[-1])
            
            yield prepared_seq[:-1], y_true
    
    def get_all_windows(self, scenario_idx: int = 0) -> List[Tuple]:
        """Get all windows for training"""
        windows = []
        graphs = self.train_scenarios[scenario_idx]
        
        for window_graphs, y_true in self.windows_from_scenario(graphs):
            windows.append((window_graphs, y_true))
        
        return windows
    
    def compute_metrics(self, logits: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float]:
        """Compute F1 metrics for predictions"""
        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_pred = (y_prob >= 0.8).astype(np.int32)
        y_true_np = y_true.detach().cpu().numpy()
        
        # F1 micro/macro multi-label
        f1_micro = f1_score(y_true_np, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true_np, y_pred, average='macro', zero_division=0)
        
        return f1_micro, f1_macro
    