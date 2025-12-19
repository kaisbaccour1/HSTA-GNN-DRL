#!/usr/bin/env python3
"""Main script to run the complete training pipeline"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import torch
import dgl

def main():
    parser = argparse.ArgumentParser(
        description='HSTA-GNN-DRL Training and Evaluation Pipeline'
    )
    
    parser.add_argument('--train-prediction', action='store_true',
                       help='Train GNN prediction model')
    parser.add_argument('--train-rl', action='store_true',
                       help='Train RL agents')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate all strategies')
    parser.add_argument('--data', type=str, default='data/graphs/manhattan_snapshots.bin',
                       help='Path to graph data')
    parser.add_argument('--repetitions', type=int, default=30,
                       help='Number of repetitions for RL training')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per repetition')
    
    args = parser.parse_args()
    
    if not any([args.train_prediction, args.train_rl, args.evaluate]):
        print("No action specified. Use --help for options.")
        return
    
    print("="*80)
    print("🚀 HSTA-GNN-DRL Pipeline")
    print("="*80)
    
    # Import modules
    from service_placement.utils.config import Config
    from service_placement.models.comparator import ModelComparator
    from service_placement.models.gnn_predictor import TemporalAttentionPredictor
    from service_placement.training.prediction_trainer import PredictionTrainer
    from service_placement.training.experiment_manager import ExperimentManager
    from service_placement.evaluation.strategy_evaluator import StrategyEvaluator
    
    # Configuration
    config = Config()
    config.n_repetitions = args.repetitions
    config.n_episodes = args.episodes
    
    # Load data
    print("\n📂 Loading scenarios...")
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("Please run simulation first: python -m models.scripts.run_simulation")
        return
    
    graphs, _ = dgl.load_graphs(args.data)
    scenarios = [graphs]
    
    # Initialize comparator
    comparator = ModelComparator(scenarios, config)
    
    if args.train_prediction:
        print("\n🧠 Training prediction model...")
        prediction_trainer = PredictionTrainer(comparator, config)
        prediction_model = prediction_trainer.train(num_epochs=config.epochs)
        prediction_trainer.plot_training()
    else:
        # Load existing prediction model
        print("\n📂 Loading existing prediction model...")
        prediction_model = TemporalAttentionPredictor(
            vehicle_dim=comparator.vehicle_dim,
            edge_dim=comparator.edge_dim,
            hid=config.n_hid,
            n_services=comparator.n_services,
            time_window=config.time_window
        ).to(comparator.device)
        
        model_path = 'models/best_prediction_model.pth'
        if os.path.exists(model_path):
            prediction_model.load_state_dict(torch.load(model_path))
        else:
            print(f"❌ Prediction model not found: {model_path}")
            print("Please train prediction model first with --train-prediction")
            return
    
    if args.train_rl:
        print("\n🤖 Training RL agents...")
        experiment_manager = ExperimentManager(config, comparator, prediction_model)
        all_results, best_agents_info = experiment_manager.run_experiments()
        best_rl_agents = {rt: info['agent'] for rt, info in best_agents_info.items()}
    else:
        # Load existing RL agents
        print("\n📂 Loading existing RL agents...")
        experiment_manager = ExperimentManager(config, comparator, prediction_model)
        if os.path.exists('models/best_agents'):
            best_agents_info = experiment_manager.load_best_agents()
            best_rl_agents = {rt: info['agent'] for rt, info in best_agents_info.items()}
        else:
            print("❌ No trained RL agents found")
            print("Please train RL agents first with --train-rl")
            return
    
    if args.evaluate:
        print("\n📊 Evaluating strategies...")
        strategy_evaluator = StrategyEvaluator(
            config, comparator, prediction_model, best_rl_agents
        )
        evaluation_results = strategy_evaluator.evaluate_all_strategies(
            test_scenario_idx=0, n_episodes=5
        )
    
    print("\n" + "="*80)
    print("✅ Pipeline completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()