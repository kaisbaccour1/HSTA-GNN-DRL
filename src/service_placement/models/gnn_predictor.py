"""GNN models for service prediction"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv, HeteroGraphConv
from typing import List, Tuple, Optional
import dgl


class SimpleHeteroGNN(nn.Module):
    def __init__(self, vehicle_dim: int, edge_dim: int, hid: int, dropout: float = 0.2):
        super().__init__()
        self.hid = hid
        
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid)
        )
        
        self.conv = HeteroGraphConv({
            'connects': GATv2Conv(hid, hid, num_heads=4)
        }, aggregate='mean')
        
        self.feature_proj = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Forward pass for heterogeneous graph"""
        # Encode node features
        v_feats = self.vehicle_encoder(g.nodes['vehicle'].data['features'])
        e_feats = self.edge_encoder(g.nodes['edge'].data['features'])
        
        # Apply heterogeneous convolution
        out = self.conv(g, {'vehicle': v_feats, 'edge': e_feats})
        edge_out = out['edge']
        
        # Handle multi-head attention output
        if edge_out.dim() == 3:  # [num_nodes, num_heads, hid]
            edge_out = edge_out.mean(dim=1)  # Average over heads
        
        # Project features
        features = self.feature_proj(edge_out)
        return features


class TemporalAttentionPredictor(nn.Module):
    def __init__(self, vehicle_dim: int, edge_dim: int, hid: int, 
                 n_services: int, time_window: int, dropout: float = 0.2):
        super().__init__()
        self.time_window = time_window
        self.hid = hid
        self.n_services = n_services
        
        # Spatial GNN
        self.gnn = SimpleHeteroGNN(vehicle_dim, edge_dim, hid, dropout)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hid, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Positional encoding for temporal sequence
        self.positional_encoding = nn.Parameter(torch.randn(1, time_window, hid))
        
        # Normalization layers
        self.layer_norm1 = nn.LayerNorm(hid)
        self.layer_norm2 = nn.LayerNorm(hid)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier for service prediction
        self.classifier = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, n_services)
        )
        
        # State projector for RL state representation
        self.state_projector = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh()
        )
        
    def forward(self, graphs: List[dgl.DGLGraph], 
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal attention predictor
        
        Args:
            graphs: List of DGL graphs for time window
            return_attention: Whether to return attention weights
        
        Returns:
            service_logits: Logits for service prediction [num_edges, n_services]
            state_embeddings: State embeddings for RL [num_edges, hid]
            attention_weights: Optional attention weights
        """
        # Extract spatial features for each time step
        spatial_features = []
        for g in graphs:
            features = self.gnn(g)  # [num_edges, hid]
            spatial_features.append(features.unsqueeze(1))  # [num_edges, 1, hid]
        
        # Create sequence [num_edges, time_window, hid]
        sequence = torch.cat(spatial_features, dim=1)
        
        # Add positional encoding
        sequence = sequence + self.positional_encoding[:, :sequence.size(1), :]
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(
            sequence, sequence, sequence
        )
        
        # Residual connection and layer norm
        attended = self.layer_norm1(sequence + self.dropout(attended))
        
        # Take the last time step as temporal embedding
        temporal_embedding = attended[:, -1, :]  # [num_edges, hid]
        temporal_embedding = self.layer_norm2(temporal_embedding)
        
        # Predict services
        service_logits = self.classifier(temporal_embedding)  # [num_edges, n_services]
        
        # Generate state embeddings for RL
        state_embeddings = self.state_projector(temporal_embedding)  # [num_edges, hid]
        
        if return_attention:
            return service_logits, state_embeddings, attention_weights
        
        return service_logits, state_embeddings
    
    def predict_services(self, graphs: List[dgl.DGLGraph], 
                        threshold: float = 0.8) -> torch.Tensor:
        """
        Predict services with thresholding
        
        Args:
            graphs: List of DGL graphs
            threshold: Probability threshold for binary prediction
        
        Returns:
            Binary service predictions [num_edges, n_services]
        """
        self.eval()
        with torch.no_grad():
            service_logits, _ = self.forward(graphs)
            service_probs = torch.sigmoid(service_logits)
            predictions = (service_probs > threshold).float()
        return predictions
    
    def get_attention_visualization(self, graphs: List[dgl.DGLGraph]) -> torch.Tensor:
        """
        Get attention weights for visualization
        
        Args:
            graphs: List of DGL graphs
        
        Returns:
            Attention weights [time_window, time_window]
        """
        self.eval()
        with torch.no_grad():
            _, _, attention_weights = self.forward(graphs, return_attention=True)
            # Average over heads and batch
            attention_weights = attention_weights.mean(dim=0).mean(dim=0)
        return attention_weights