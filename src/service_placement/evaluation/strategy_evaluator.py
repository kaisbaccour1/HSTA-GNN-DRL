"""Evaluator for comparing different strategies"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

from ..config import ServicePlacementConfig
from ..models.comparator import ModelComparator
from ..models.gnn_predictor import TemporalAttentionPredictor
from ..environment.edge_environment import EdgeEnvironment
from ..agents.rl_agent import RewardSpecificAgent
from ..agents.dummy_agent import DummyAgent
from ..utils.reward_calculator import RewardCalculator


class StrategyEvaluator:
    def __init__(self, config: ServicePlacementConfig, comparator: ModelComparator,
                 prediction_model: TemporalAttentionPredictor, rl_agents_dict: Dict):
        self.config = config
        self.comparator = comparator
        self.prediction_model = prediction_model
        self.device = comparator.device
        self.rl_agents = rl_agents_dict
        
        # Create dummy agents
        self.dummy_agents = {
            'random': DummyAgent(config, self.device, 'random'),
            'zero_deployment': DummyAgent(config, self.device, 'zero_deployment'),
            'prediction_follower': DummyAgent(config, self.device, 'prediction_follower')
        }
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Create directory for results
        os.makedirs('experiment_plots/strategy_evaluation_plots', exist_ok=True)
    
    def evaluate_all_strategies(self, test_scenario_idx: int = 0, n_episodes: int = 5) -> Dict:
        """Evaluate all strategies on test scenarios"""
        print("\n" + "="*80)
        print("🎯 STRATEGY EVALUATION ON TEST SCENARIOS")
        print("="*80)
        
        all_results = {
            'random': self._evaluate_random_strategy(test_scenario_idx, n_episodes, use_eval_set=True),
            'zero_deployment': self._evaluate_zero_deployment_strategy(test_scenario_idx, n_episodes, use_eval_set=True),
            'prediction_follower': self._evaluate_prediction_follower_strategy(test_scenario_idx, n_episodes, use_eval_set=True),
            'rl_energy': self._evaluate_rl_strategy('energy_only', test_scenario_idx, n_episodes, use_eval_set=True),
            'rl_latency': self._evaluate_rl_strategy('latency_only', test_scenario_idx, n_episodes, use_eval_set=True),
            'rl_combined': self._evaluate_rl_strategy('energy_latency_combined', test_scenario_idx, n_episodes, use_eval_set=True)
        }
        
        self._generate_all_evaluation_plots(all_results)
        return all_results
    
    def _evaluate_random_strategy(self, scenario_idx: int, n_episodes: int, 
                                 use_eval_set: bool = False) -> Dict:
        """Evaluate random strategy"""
        print(f"\n🔀 Evaluating Random Strategy ({'EVAL' if use_eval_set else 'TRAIN'} set)...")
        
        env = EdgeEnvironment(
            self.comparator, 
            self.prediction_model, 
            self.config
        )
        
        env.set_agent(self.dummy_agents['random'])
        
        results = {
            'rewards': {'energy': [], 'latency': [], 'combined': []},
            'energy': [],
            'latency': [],
        }
        
        for episode in range(n_episodes):
            state_dict = env.reset(scenario_idx, use_eval_set=use_eval_set)
            if state_dict is None:
                print(f"   ⚠️ Episode {episode}: state_dict is None, skipping")
                continue
                
            episode_rewards = {'energy': 0, 'latency': 0, 'combined': 0}
            episode_energy = 0
            episode_latency = 0
            steps = 0
            
            while True:
                # Random action
                actions = {}
                for edge_id in env.edge_nodes.keys():
                    if edge_id == env.cloud_edge_id:
                        continue
                    actions[edge_id] = np.random.randint(0, 2**self.config.n_services)
                
                # Execute step
                next_state_dict, reward, done, info = env.step(actions)
                
                # Extract metrics
                energy = info.get('total_energy', 0)
                latency = info.get('latency', 0)
                
                # Calculate rewards for all types
                energy_reward = self.reward_calculator.energy_only_reward(energy, latency)
                latency_reward = self.reward_calculator.latency_only_reward(energy, latency)
                combined_reward = self.reward_calculator.energy_latency_combined_reward(energy, latency)
                
                episode_rewards['energy'] += energy_reward
                episode_rewards['latency'] += latency_reward
                episode_rewards['combined'] += combined_reward
                
                episode_energy += energy
                episode_latency += latency
                
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state_dict = next_state_dict
            
            # Normalize by number of steps
            if steps > 0:
                results['rewards']['energy'].append(episode_rewards['energy'] / steps)
                results['rewards']['latency'].append(episode_rewards['latency'] / steps)
                results['rewards']['combined'].append(episode_rewards['combined'] / steps)
                results['energy'].append(episode_energy / steps)
                results['latency'].append(episode_latency / steps)
            else:
                print(f"   ⚠️ Episode {episode}: 0 steps completed")
        
        # Calculate averages
        avg_results = {
            'reward_energy': np.mean(results['rewards']['energy']) if results['rewards']['energy'] else 0,
            'reward_latency': np.mean(results['rewards']['latency']) if results['rewards']['latency'] else 0,
            'reward_combined': np.mean(results['rewards']['combined']) if results['rewards']['combined'] else 0,
            'energy': np.mean(results['energy']) if results['energy'] else 0,
            'latency': np.mean(results['latency']) if results['latency'] else 0
        }
        
        print(f"   ✅ Random Strategy evaluated: Energy={avg_results['energy']:.2f}J, Latency={avg_results['latency']:.2f}s")
        return avg_results
    
    def _evaluate_zero_deployment_strategy(self, scenario_idx: int, n_episodes: int,
                                          use_eval_set: bool = False) -> Dict:
        """Evaluate zero-deployment strategy"""
        print(f"\n☁️  Evaluating Zero-Deployment Strategy ({'EVAL' if use_eval_set else 'TRAIN'} set)...")
        
        env = EdgeEnvironment(
            self.comparator, 
            self.prediction_model, 
            self.config
        )
        
        env.set_agent(self.dummy_agents['zero_deployment'])
        
        results = {
            'rewards': {'energy': [], 'latency': [], 'combined': []},
            'energy': [],
            'latency': []
        }
        
        for episode in range(n_episodes):
            state_dict = env.reset(scenario_idx, use_eval_set=use_eval_set)
            if state_dict is None:
                print(f"   ⚠️ Episode {episode}: state_dict is None, skipping")
                continue
                
            episode_rewards = {'energy': 0, 'latency': 0, 'combined': 0}
            episode_energy = 0
            episode_latency = 0
            steps = 0
            
            while True:
                # Zero-Deployment action
                actions = {}
                for edge_id in env.edge_nodes.keys():
                    if edge_id == env.cloud_edge_id:
                        continue
                    actions[edge_id] = 0
                
                next_state_dict, reward, done, info = env.step(actions)
                
                if info is None or not isinstance(info, dict):
                    print(f"   ⚠️ Episode {episode}, step {steps}: info is not a dict, breaking")
                    break
                
                # Extract metrics
                energy = info.get('total_energy', 0)
                latency = info.get('latency', 0)
                
                # Calculate rewards
                energy_reward = self.reward_calculator.energy_only_reward(energy, latency)
                latency_reward = self.reward_calculator.latency_only_reward(energy, latency)
                combined_reward = self.reward_calculator.energy_latency_combined_reward(energy, latency)
                
                episode_rewards['energy'] += energy_reward
                episode_rewards['latency'] += latency_reward
                episode_rewards['combined'] += combined_reward
                
                episode_energy += energy
                episode_latency += latency
                
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state_dict = next_state_dict
            
            if steps > 0:
                results['rewards']['energy'].append(episode_rewards['energy'] / steps)
                results['rewards']['latency'].append(episode_rewards['latency'] / steps)
                results['rewards']['combined'].append(episode_rewards['combined'] / steps)
                results['energy'].append(episode_energy / steps)
                results['latency'].append(episode_latency / steps)
        
        avg_results = {
            'reward_energy': np.mean(results['rewards']['energy']) if results['rewards']['energy'] else 0,
            'reward_latency': np.mean(results['rewards']['latency']) if results['rewards']['latency'] else 0,
            'reward_combined': np.mean(results['rewards']['combined']) if results['rewards']['combined'] else 0,
            'energy': np.mean(results['energy']) if results['energy'] else 0,
            'latency': np.mean(results['latency']) if results['latency'] else 0
        }
        
        print(f"   ✅ Zero-Deployment Strategy evaluated: Energy={avg_results['energy']:.2f}J, Latency={avg_results['latency']:.2f}s")
        return avg_results
    
    def _evaluate_prediction_follower_strategy(self, scenario_idx: int, n_episodes: int,
                                              use_eval_set: bool = False) -> Dict:
        """Evaluate prediction follower strategy"""
        print(f"\n📊 Evaluating Prediction Follower ({'EVAL' if use_eval_set else 'TRAIN'} set)...")
        
        env = EdgeEnvironment(
            self.comparator, 
            self.prediction_model, 
            self.config
        )
        
        env.set_agent(self.dummy_agents['prediction_follower'])
        
        results = {
            'rewards': {'energy': [], 'latency': [], 'combined': []},
            'energy': [],
            'latency': []
        }
        
        for episode in range(n_episodes):
            state_dict = env.reset(scenario_idx, use_eval_set=use_eval_set)
            if state_dict is None:
                print(f"   ⚠️ Episode {episode}: state_dict is None, skipping")
                continue
                
            episode_rewards = {'energy': 0, 'latency': 0, 'combined': 0}
            episode_energy = 0
            episode_latency = 0
            steps = 0
            
            while True:
                # Strategy: deploy all predicted services
                predicted_services = state_dict['predicted_services']
                actions = {}
                
                for edge_id in env.edge_nodes.keys():
                    if edge_id == env.cloud_edge_id:
                        continue
                    
                    # Find edge index in predictions
                    edge_indices = [i for i, (eid, _) in enumerate(state_dict['edge_states'].items()) 
                                  if eid == edge_id]
                    if edge_indices:
                        edge_pred_idx = edge_indices[0]
                        if edge_pred_idx < predicted_services.shape[0]:
                            edge_preds = predicted_services[edge_pred_idx]
                            
                            # Create action based on predictions
                            action = 0
                            for service_idx in range(self.config.n_services):
                                if service_idx < len(edge_preds) and edge_preds[service_idx] > 0.5:
                                    action |= (1 << service_idx)
                            
                            actions[edge_id] = action
                        else:
                            actions[edge_id] = 0
                    else:
                        actions[edge_id] = 0
                
                # Execute step
                next_state_dict, reward, done, info = env.step(actions)
                
                # Extract metrics
                energy = info.get('total_energy', 0)
                latency = info.get('latency', 0)
                
                # Calculate rewards
                energy_reward = self.reward_calculator.energy_only_reward(energy, latency)
                latency_reward = self.reward_calculator.latency_only_reward(energy, latency)
                combined_reward = self.reward_calculator.energy_latency_combined_reward(energy, latency)
                
                episode_rewards['energy'] += energy_reward
                episode_rewards['latency'] += latency_reward
                episode_rewards['combined'] += combined_reward
                
                episode_energy += energy
                episode_latency += latency
                
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state_dict = next_state_dict
            
            if steps > 0:
                results['rewards']['energy'].append(episode_rewards['energy'] / steps)
                results['rewards']['latency'].append(episode_rewards['latency'] / steps)
                results['rewards']['combined'].append(episode_rewards['combined'] / steps)
                results['energy'].append(episode_energy / steps)
                results['latency'].append(episode_latency / steps)
        
        avg_results = {
            'reward_energy': np.mean(results['rewards']['energy']) if results['rewards']['energy'] else 0,
            'reward_latency': np.mean(results['rewards']['latency']) if results['rewards']['latency'] else 0,
            'reward_combined': np.mean(results['rewards']['combined']) if results['rewards']['combined'] else 0,
            'energy': np.mean(results['energy']) if results['energy'] else 0,
            'latency': np.mean(results['latency']) if results['latency'] else 0
        }
        
        print(f"   ✅ Prediction Follower evaluated: Energy={avg_results['energy']:.2f}J, Latency={avg_results['latency']:.2f}s")
        return avg_results
    
    def _evaluate_rl_strategy(self, reward_type: str, scenario_idx: int, n_episodes: int,
                             use_eval_set: bool = False) -> Dict:
        """Evaluate specific RL agent"""
        print(f"\n🤖 Evaluating RL-{reward_type.replace('_', ' ').title()} Strategy...")
        
        if reward_type not in self.rl_agents:
            print(f"   ❌ No RL agent found for reward type: {reward_type}")
            return {
                'reward_energy': 0, 'reward_latency': 0, 'reward_combined': 0,
                'energy': 0, 'latency': 0
            }
        
        agent = self.rl_agents[reward_type]
        env = EdgeEnvironment(self.comparator, self.prediction_model, self.config)
        env.set_agent(agent)
        
        results = {
            'rewards': {'energy': [], 'latency': [], 'combined': []},
            'energy': [],
            'latency': []
        }
        
        for episode in range(n_episodes):
            state_dict = env.reset(scenario_idx, use_eval_set=use_eval_set)
            
            if state_dict is None:
                continue
                
            state = state_dict['state_vector']
            predicted_services = state_dict['predicted_services']
            
            episode_reward = 0
            episode_energy = 0
            episode_latency = 0
            steps = 0
            
            while True:
                # RL agent action
                actions = agent.select_action(state, predicted_services, explore=False)
                
                # Execute step
                next_state_dict, reward, done, info = env.step(actions)
                
                # Extract metrics
                energy = info.get('total_energy', 0)
                latency = info.get('latency', 0)
                
                episode_reward += reward
                episode_energy += energy
                episode_latency += latency
                
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state_dict = next_state_dict
                state = state_dict['state_vector']
                predicted_services = state_dict['predicted_services']
            
            if steps > 0:
                # For RL, we only have one reward (training reward)
                if reward_type == 'energy_only':
                    results['rewards']['energy'].append(episode_reward / steps)
                elif reward_type == 'latency_only':
                    results['rewards']['latency'].append(episode_reward / steps)
                else:
                    results['rewards']['combined'].append(episode_reward / steps)
                
                results['energy'].append(episode_energy / steps)
                results['latency'].append(episode_latency / steps)
        
        avg_results = {
            'reward_energy': np.mean(results['rewards']['energy']) if results['rewards']['energy'] else 0,
            'reward_latency': np.mean(results['rewards']['latency']) if results['rewards']['latency'] else 0,
            'reward_combined': np.mean(results['rewards']['combined']) if results['rewards']['combined'] else 0,
            'energy': np.mean(results['energy']) if results['energy'] else 0,
            'latency': np.mean(results['latency']) if results['latency'] else 0
        }
        
        print(f"   ✅ RL-{reward_type.replace('_', ' ').title()} evaluated: "
              f"Energy={avg_results['energy']:.2f}J, Latency={avg_results['latency']:.2f}s")
        
        return avg_results
    
    def _generate_all_evaluation_plots(self, all_results: Dict):
        """Generate the evaluation plots"""
        self._plot_reward_comparison(all_results)
        self._plot_energy_comparison(all_results)
        self._plot_latency_comparison(all_results)
    
    def _plot_reward_comparison(self, all_results: Dict):
        """Figure 1: Comparison of Rewards (3 formulas)"""
        plt.figure(figsize=(16, 8))
        
        # Groups: Random, Zero-Deployment, Prediction-follower, RL
        strategy_groups = ['Random', 'Zero-Deployment', 'Prediction-Follower', 'RL']
        
        # Data organized by group
        group_data = {
            'Random': {
                'Energy': all_results['random']['reward_energy'],
                'Latency': all_results['random']['reward_latency'],
                'Combined': all_results['random']['reward_combined']
            },
            'Zero-Deployment': {
                'Energy': all_results['zero_deployment']['reward_energy'],
                'Latency': all_results['zero_deployment']['reward_latency'],
                'Combined': all_results['zero_deployment']['reward_combined']
            },
            'Prediction-Follower': {
                'Energy': all_results['prediction_follower']['reward_energy'],
                'Latency': all_results['prediction_follower']['reward_latency'],
                'Combined': all_results['prediction_follower']['reward_combined']
            },
            'RL': {
                'Energy': all_results['rl_energy']['reward_energy'],
                'Latency': all_results['rl_latency']['reward_latency'],
                'Combined': all_results['rl_combined']['reward_combined']
            }
        }
        
        # Configuration
        bar_width = 0.25
        x = np.arange(len(strategy_groups))
        
        # Colors for the 3 reward formulas
        colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
        
        # Create grouped bars
        for i, (formula, color) in enumerate(zip(['Energy', 'Latency', 'Combined'], colors)):
            values = [group_data[group][formula] for group in strategy_groups]
            positions = x + (i - 1) * bar_width
            
            bars = plt.bar(positions, values, bar_width, 
                          color=color, alpha=0.8,
                          edgecolor='black', linewidth=1.5,
                          label=f'{formula} Reward')
            
            # Add values
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.2f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        
        # Axis configuration
        plt.xlabel('Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Value', fontsize=14, fontweight='bold')
        plt.title('Objective Comparison Across Strategies and Formulas', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(x, strategy_groups, fontsize=12, fontweight='bold')
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig('experiment_plots/strategy_evaluation_plots/reward_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_energy_comparison(self, all_results: Dict):
        """Figure 2: Energy Consumption Comparison"""
        plt.figure(figsize=(14, 8))
        
        # Data for each strategy
        strategies = ['Random', 'Zero-Deployment', 'Prediction-Follower', 
                     'RL-Energy', 'RL-Latency', 'RL-Combined']
        values = [
            all_results['random']['energy'],
            all_results['zero_deployment']['energy'],
            all_results['prediction_follower']['energy'],
            all_results['rl_energy']['energy'],
            all_results['rl_latency']['energy'],
            all_results['rl_combined']['energy']
        ]
        
        # Colors
        colors = ['#95A5A6', '#95A5A6', '#95A5A6',  # Gray for non-RL
                 '#E74C3C', '#2ECC71', '#3498DB']   # Red, Green, Blue for RL
        
        x = np.arange(len(strategies))
        
        bars = plt.bar(x, values, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        # Add values
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        plt.xlabel('Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Energy Consumption (Joules)', fontsize=14, fontweight='bold')
        plt.title('Energy Consumption Across Strategies', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(x, strategies, fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('experiment_plots/strategy_evaluation_plots/energy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_latency_comparison(self, all_results: Dict):
        """Figure 3: Latency Comparison"""
        plt.figure(figsize=(14, 8))
        
        strategies = ['Random', 'Zero-Deployment', 'Prediction-Follower', 
                     'RL-Energy', 'RL-Latency', 'RL-Combined']
        values = [
            all_results['random']['latency'],
            all_results['zero_deployment']['latency'],
            all_results['prediction_follower']['latency'],
            all_results['rl_energy']['latency'],
            all_results['rl_latency']['latency'],
            all_results['rl_combined']['latency']
        ]
        
        colors = ['#95A5A6', '#95A5A6', '#95A5A6',
                 '#E74C3C', '#2ECC71', '#3498DB']
        
        x = np.arange(len(strategies))
        
        bars = plt.bar(x, values, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        plt.xlabel('Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Average Latency (s)', fontsize=14, fontweight='bold')
        plt.title('Latency Performance Across Strategies', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(x, strategies, fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('experiment_plots/strategy_evaluation_plots/latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()