"""Trainer for GNN prediction model"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from typing import List, Tuple
from ..models.comparator import ModelComparator
from ..models.gnn_predictor import TemporalAttentionPredictor
from ..config import ServicePlacementConfig


class PredictionTrainer:
    def __init__(self, comparator: ModelComparator, config: ServicePlacementConfig):
        self.comparator = comparator
        self.cfg = config
        self.device = comparator.device
        
        self.model = TemporalAttentionPredictor(
            vehicle_dim=comparator.vehicle_dim,
            edge_dim=comparator.edge_dim,
            hid=config.n_hid,
            n_services=comparator.n_services,
            time_window=config.time_window
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_losses = []
        self.f1_micro = []
    
    def train_model(self, graphs: List[dgl.DGLGraph]) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for graphs_win, y_true in self.comparator.windows_from_scenario(graphs):
            self.optimizer.zero_grad()
            
            prepared_graphs = [self.comparator.prepare_features(g) for g in graphs_win]
            logits, emb = self.model(prepared_graphs)
            loss = self.criterion(logits, y_true)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def evaluate_model(self, graphs: List[dgl.DGLGraph]) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for graphs_win, y_true in self.comparator.windows_from_scenario(graphs):
                prepared_graphs = [self.comparator.prepare_features(g) for g in graphs_win]
                logits, emb = self.model(prepared_graphs)
                all_logits.append(logits)
                all_targets.append(y_true)
        
        if all_logits:
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            return self._compute_metrics(all_logits, all_targets)
        
        return 0, 0
    
    def _compute_metrics(self, logits: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float]:
        """Compute F1 metrics"""
        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_bin = (y_prob >= 0.8).astype(np.int32)
        y_true = y_true.detach().cpu().numpy()
        
        f1_micro = f1_score(y_true, y_bin, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_bin, average='macro', zero_division=0)
        
        return f1_micro, f1_macro
    
    def train(self, num_epochs: int = 10):
        """Full training loop"""
        print("🔧 Starting prediction model training...")
        
        best_f1 = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_model(self.comparator.train_scenarios[0])
            f1_micro, f1_macro = self.evaluate_model(self.comparator.eval_scenarios[0])
            
            self.train_losses.append(train_loss)
            self.f1_micro.append(f1_micro)
            
            if f1_micro > best_f1:
                best_f1 = f1_micro
                torch.save(self.model.state_dict(), 'models/best_prediction_model.pth')
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | F1µ: {f1_micro:.4f} | F1M: {f1_macro:.4f}")
        
        print("✅ Prediction model training completed")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_prediction_model.pth'))
        return self.model
    
    def plot_training(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.f1_micro, label='F1µ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.show()
