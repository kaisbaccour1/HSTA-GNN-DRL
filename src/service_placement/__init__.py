"""Service Placement System with GNN prediction and RL optimization"""

from .config import ServicePlacementConfig
from .service_specs import ServiceSpecs
from .models.comparator import ModelComparator
from .models.gnn_predictor import TemporalAttentionPredictor
from .environment.edge_environment import EdgeEnvironment
from .environment.edge_node import EdgeNode
from .agents.rl_agent import RLAgent, RewardSpecificAgent
from .training.prediction_trainer import PredictionTrainer
from .training.rl_trainer import RLTrainer
from .training.experiment_manager import ExperimentManager
from .evaluation.strategy_evaluator import StrategyEvaluator
from .utils.reward_calculator import RewardCalculator


__version__ = "0.1.0"
__all__ = [
    'ServicePlacementConfig',
    'ServiceSpecs',
    'ModelComparator',
    'TemporalAttentionPredictor',
    'EdgeEnvironment',
    'EdgeNode',
    'RLAgent',
    'RewardSpecificAgent',
    'PredictionTrainer',
    'RLTrainer',
    'ExperimentManager',
    'StrategyEvaluator',
    'RewardCalculator'
]