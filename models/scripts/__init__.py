"""
Scripts for running simulation and training
"""

from .run_simulation import main as run_simulation
from .run_training_and_evaluation import main as run_training

__all__ = ['run_simulation', 'run_training']