"""Metrics computation for evaluation"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import Tuple, Dict


def compute_classification_metrics(logits: np.ndarray, y_true: np.ndarray, 
                                  threshold: float = 0.8) -> Dict[str, float]:
    """Compute multi-label classification metrics"""
    y_prob = sigmoid(logits)
    y_pred = (y_prob >= threshold).astype(np.int32)
    
    # Initialize metrics dict
    metrics = {}
    
    # F1 scores
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Precision and recall
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        if np.any(y_true[:, i]):  # Only if class exists in true labels
            metrics[f'f1_class_{i}'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'precision_class_{i}'] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics[f'recall_class_{i}'] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
    
    return metrics


def compute_rl_metrics(rewards: np.ndarray, energies: np.ndarray, 
                      latencies: np.ndarray) -> Dict[str, float]:
    """Compute RL performance metrics"""
    metrics = {}
    
    # Reward statistics
    metrics['avg_reward'] = np.mean(rewards)
    metrics['std_reward'] = np.std(rewards)
    metrics['min_reward'] = np.min(rewards)
    metrics['max_reward'] = np.max(rewards)
    metrics['total_reward'] = np.sum(rewards)
    
    # Energy statistics
    metrics['avg_energy'] = np.mean(energies)
    metrics['std_energy'] = np.std(energies)
    metrics['min_energy'] = np.min(energies)
    metrics['max_energy'] = np.max(energies)
    metrics['total_energy'] = np.sum(energies)
    
    # Latency statistics
    metrics['avg_latency'] = np.mean(latencies)
    metrics['std_latency'] = np.std(latencies)
    metrics['min_latency'] = np.min(latencies)
    metrics['max_latency'] = np.max(latencies)
    
    # Efficiency metrics
    if np.mean(energies) > 0:
        metrics['reward_per_energy'] = np.mean(rewards) / np.mean(energies)
    else:
        metrics['reward_per_energy'] = 0
    
    if np.mean(latencies) > 0:
        metrics['reward_per_latency'] = np.mean(rewards) / np.mean(latencies)
    else:
        metrics['reward_per_latency'] = 0
    
    # Improvement percentages (if baseline available)
    metrics['improvement'] = {}
    
    return metrics


def compute_service_utilization(services_hosted: np.ndarray, 
                               services_used: np.ndarray) -> Dict[str, float]:
    """Compute service utilization metrics"""
    n_edges, n_services = services_hosted.shape
    
    metrics = {}
    
    # Overall utilization
    total_hosted = np.sum(services_hosted)
    total_used = np.sum(services_used)
    
    if total_hosted > 0:
        metrics['overall_utilization'] = total_used / total_hosted
    else:
        metrics['overall_utilization'] = 0
    
    # Per-service utilization
    for i in range(n_services):
        hosted = np.sum(services_hosted[:, i])
        used = np.sum(services_used[:, i])
        
        if hosted > 0:
            metrics[f'service_{i}_utilization'] = used / hosted
        else:
            metrics[f'service_{i}_utilization'] = 0
        
        metrics[f'service_{i}_hosted'] = hosted
        metrics[f'service_{i}_used'] = used
    
    # Per-edge utilization
    for j in range(n_edges):
        hosted = np.sum(services_hosted[j, :])
        used = np.sum(services_used[j, :])
        
        if hosted > 0:
            metrics[f'edge_{j}_utilization'] = used / hosted
        else:
            metrics[f'edge_{j}_utilization'] = 0
    
    return metrics


def compute_resource_efficiency(cpu_available: np.ndarray, cpu_capacity: np.ndarray,
                              ram_available: np.ndarray, ram_capacity: np.ndarray) -> Dict[str, float]:
    """Compute resource efficiency metrics"""
    metrics = {}
    
    # CPU efficiency
    cpu_used = cpu_capacity - cpu_available
    if np.sum(cpu_capacity) > 0:
        metrics['cpu_efficiency'] = np.sum(cpu_used) / np.sum(cpu_capacity)
    else:
        metrics['cpu_efficiency'] = 0
    
    # RAM efficiency
    ram_used = ram_capacity - ram_available
    if np.sum(ram_capacity) > 0:
        metrics['ram_efficiency'] = np.sum(ram_used) / np.sum(ram_capacity)
    else:
        metrics['ram_efficiency'] = 0
    
    # Overall resource efficiency
    metrics['overall_efficiency'] = (metrics['cpu_efficiency'] + metrics['ram_efficiency']) / 2
    
    # Resource wastage
    metrics['cpu_wastage'] = 1 - metrics['cpu_efficiency']
    metrics['ram_wastage'] = 1 - metrics['ram_efficiency']
    
    return metrics


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))