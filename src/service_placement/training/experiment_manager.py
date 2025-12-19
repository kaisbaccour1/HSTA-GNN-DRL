"""Manages RL experiments with multiple reward types"""
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import stats

from ..config import ServicePlacementConfig
from ..models.comparator import ModelComparator
from ..models.gnn_predictor import TemporalAttentionPredictor
from ..environment.edge_environment import EdgeEnvironment
from ..agents.rl_agent import RewardSpecificAgent
from ..training.rl_trainer import RLTrainer


class ExperimentManager:
    def __init__(self, config: ServicePlacementConfig, comparator: ModelComparator,
                 prediction_model: TemporalAttentionPredictor):
        self.config = config
        self.comparator = comparator
        self.prediction_model = prediction_model
        self.device = comparator.device
        
        # Results storage
        self.all_results = {}
        self.summary_stats = {}
        self.best_agents = {}
        
        # Create directories
        os.makedirs('experiment_plots/comparison_CI95', exist_ok=True)
        os.makedirs('models/best_agents', exist_ok=True)

    
    def run_experiments(self) -> Tuple[Dict, Dict]:
        """Execute all experiments"""
        print("="*80)
        print("🎯 STARTING EXPERIMENTS - 30 REPETITIONS PER REWARD TYPE")
        print("="*80)
        
        for reward_type in self.config.reward_types:
            print(f"\n{'='*60}")
            print(f"🧪 REWARD TYPE: {reward_type.upper()}")
            print(f"{'='*60}")
            
            results, agents = self.run_reward_type_experiment(reward_type)
            self.all_results[reward_type] = results
            
            # Find best agent
            if agents:
                best_agent_info = max(agents, key=lambda x: x['final_reward'])
                self.best_agents[reward_type] = best_agent_info
                
                print(f"   🏆 Best agent {reward_type}: "
                      f"Repetition {best_agent_info['repetition']+1}, "
                      f"Final reward: {best_agent_info['final_reward']:.3f}")
        
        # Save best agents
        self._save_best_agents()
        
        # Analyze and visualize
        self.analyze_all_results()
        self.plot_comparative_results()
        
        return self.all_results, self.best_agents
    
    def run_reward_type_experiment(self, reward_type: str) -> Tuple[List, List]:
        """Execute 30 repetitions for a given reward type"""
        all_repetitions = []
        all_agents = []
        
        for repetition in range(self.config.n_repetitions):
            print(f"\n🔁 Repetition {repetition + 1}/{self.config.n_repetitions} for {reward_type}")
            
            # Create environment and agent
            env = EdgeEnvironment(
                self.comparator, 
                self.prediction_model, 
                self.config
            )
            
            sample_state = env.get_state()
            state_dim = len(sample_state['state_vector']) if sample_state else 100
            action_dim = 2 ** self.config.n_services
            
            agent = RewardSpecificAgent(
                state_dim, action_dim, self.config, self.device, reward_type
            )
            
            env.set_agent(agent)
            trainer = RLTrainer(env, agent, self.config)
            
            # Training
            metrics = trainer.train(num_episodes=self.config.n_episodes)
            
            # Add identification
            metrics['reward_type'] = reward_type
            metrics['repetition'] = repetition
            metrics['agent'] = agent
            
            all_repetitions.append(metrics)
            all_agents.append({
                'agent': agent,
                'repetition': repetition,
                'best_reward': agent.best_reward,
                'final_reward': metrics.get('rewards', [0])[-1] if metrics.get('rewards') else 0
            })
        
        return all_repetitions, all_agents
    
    def _save_best_agents(self):
        """Save best agents to disk"""
        for reward_type, agent_info in self.best_agents.items():
            # Get agent and state_dim
            agent = agent_info['agent']
            
            # Save with pickle
            pickle_filename = f'/models/best_agents/best_{reward_type}_agent.pkl'
            with open(pickle_filename, 'wb') as f:
                pickle.dump(agent_info, f)
            
            # Save PyTorch weights
            torch_filename = f'best_agents/best_{reward_type}_agent_weights.pth'
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_target_state_dict': agent.actor_target.state_dict(),
                'critic_target_state_dict': agent.critic_target.state_dict(),
                'state_dim': agent.actor.network[0].in_features,
                'action_dim': agent.action_dim,
                'config': self.config,
                'reward_type': reward_type,
                'best_reward': agent_info['best_reward'],
                'final_reward': agent_info['final_reward'],
                'repetition': agent_info['repetition']
            }, torch_filename)
            
            print(f"💾 Best agent {reward_type} saved")
    
    def load_best_agents(self) -> Dict:
        """Load best agents from disk"""
        loaded_agents = {}
        
        for reward_type in self.config.reward_types:
            pickle_filename = f'models/best_agents/best_{reward_type}_agent.pkl'
            torch_filename = f'best_agents/best_{reward_type}_agent_weights.pth'
            
            if os.path.exists(pickle_filename) and os.path.exists(torch_filename):
                print(f"\n📂 Loading {reward_type} agent...")
                
                # Load checkpoint
                checkpoint = torch.load(torch_filename, map_location=self.device)
                
                # Use saved dimensions
                state_dim = checkpoint['state_dim']
                action_dim = checkpoint['action_dim']
                
                # Create new agent
                new_agent = RewardSpecificAgent(
                    state_dim, action_dim, self.config, self.device, reward_type
                )
                
                # Load weights
                new_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                new_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                new_agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                new_agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                
                # Create agent_info
                agent_info = {
                    'agent': new_agent,
                    'repetition': checkpoint.get('repetition', 0),
                    'best_reward': checkpoint.get('best_reward', 0),
                    'final_reward': checkpoint.get('final_reward', 0),
                    'state_dim': state_dim
                }
                
                loaded_agents[reward_type] = agent_info
                
                print(f"   ✅ Agent {reward_type} loaded")
            else:
                print(f"   ⚠️ No saved agent found for {reward_type}")
        
        self.best_agents = loaded_agents
        return loaded_agents
    
    def analyze_all_results(self):
        """Statistical analysis of all results"""
        print("\n" + "="*80)
        print("📊 STATISTICAL ANALYSIS OF RESULTS")
        print("="*80)
        
        for reward_type in self.config.reward_types:
            if reward_type not in self.all_results:
                continue
            
            results = self.all_results[reward_type]
            n_episodes = self.config.n_episodes
            
            # Initialize matrices
            reward_matrix = np.zeros((len(results), n_episodes))
            energy_matrix = np.zeros((len(results), n_episodes))
            latency_matrix = np.zeros((len(results), n_episodes))
            
            # Fill matrices
            for rep_idx, rep in enumerate(results):
                if 'rewards' in rep and len(rep['rewards']) == n_episodes:
                    reward_matrix[rep_idx] = rep['rewards']
                
                if 'avg_energies' in rep and len(rep['avg_energies']) == n_episodes:
                    energy_matrix[rep_idx] = rep['avg_energies']
                
                if 'avg_latencies' in rep and len(rep['avg_latencies']) == n_episodes:
                    latency_matrix[rep_idx] = rep['avg_latencies']
            
            # Calculate statistics
            stats_dict = {
                'reward': self._calculate_confidence_interval_stats(reward_matrix, 'reward'),
                'energy': self._calculate_confidence_interval_stats(energy_matrix, 'energy'),
                'latency': self._calculate_confidence_interval_stats(latency_matrix, 'latency')
            }
            
            self.summary_stats[reward_type] = stats_dict
            
            # Display with CI95
            print(f"\n📈 {reward_type.upper()} - CI95 (30 repetitions):")
            print("-"*80)
            print(f"Final reward: {stats_dict['reward']['mean_overall']:.3f} "
                  f"[{stats_dict['reward']['ci_lower']:.3f}, {stats_dict['reward']['ci_upper']:.3f}]")
            print(f"Average energy: {stats_dict['energy']['mean_overall']:.3f} "
                  f"[{stats_dict['energy']['ci_lower']:.3f}, {stats_dict['energy']['ci_upper']:.3f}] J")
            print(f"Average latency: {stats_dict['latency']['mean_overall']:.3f} "
                  f"[{stats_dict['latency']['ci_lower']:.3f}, {stats_dict['latency']['ci_upper']:.3f}] ms")
    
    def _calculate_confidence_interval_stats(self, data_matrix, metric_name, confidence=0.95):
        """Calculate statistics with confidence interval"""
        # Mean and std per episode
        mean_per_episode = np.nanmean(data_matrix, axis=0)
        std_per_episode = np.nanstd(data_matrix, axis=0, ddof=1)
        
        # Overall statistics
        n_experiments = data_matrix.shape[0]
        mean_overall = np.nanmean(data_matrix)
        std_overall = np.nanstd(data_matrix.flatten()[~np.isnan(data_matrix.flatten())])
        
        # Standard error
        se = std_overall / np.sqrt(n_experiments)
        
        # Critical value
        if n_experiments >= 30:
            critical_value = stats.norm.ppf((1 + confidence) / 2)
        else:
            critical_value = stats.t.ppf((1 + confidence) / 2, df=n_experiments-1)
        
        # Calculate confidence interval
        ci_lower = mean_overall - critical_value * se
        ci_upper = mean_overall + critical_value * se
        
        # Calculate CI per episode
        ci_lower_per_episode = mean_per_episode - critical_value * (std_per_episode / np.sqrt(n_experiments))
        ci_upper_per_episode = mean_per_episode + critical_value * (std_per_episode / np.sqrt(n_experiments))
        
        return {
            'mean_per_episode': mean_per_episode,
            'ci_lower_per_episode': ci_lower_per_episode,
            'ci_upper_per_episode': ci_upper_per_episode,
            'mean_overall': mean_overall,
            'std_overall': std_overall,
            'standard_error': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence,
            'critical_value': critical_value,
            'n_experiments': n_experiments
        }
    
    def plot_comparative_results(self):
        """Generate comparative plots with confidence intervals"""
        self._plot_metric_comparison('reward', 'Reward per Episode', 'Reward')
        self._plot_metric_comparison('energy', 'Average Energy per Episode', 'Energy (J)')
        self._plot_metric_comparison('latency', 'Average Latency per Episode', 'Latency (s)')
    
    def _plot_metric_comparison(self, metric_name, title, ylabel):
        """Generate comparative plot for a specific metric"""
        plt.figure(figsize=(14, 8))
        
        # Color mapping
        color_map = {
            'energy_only': '#E74C3C',
            'latency_only': '#2ECC71',
            'energy_latency_combined': '#3498DB'
        }
        
        # Legend name mapping
        legend_names = {
            'energy_only': 'RL-Energy',
            'latency_only': 'RL-Latency',
            'energy_latency_combined': 'RL-Combined'
        }
        
        for reward_type in self.config.reward_types:
            if reward_type in self.summary_stats:
                stats = self.summary_stats[reward_type][metric_name]
                mean_series = stats['mean_per_episode']
                ci_lower = stats['ci_lower_per_episode']
                ci_upper = stats['ci_upper_per_episode']
                episodes = range(1, len(mean_series) + 1)
                
                # Mean line
                plt.plot(episodes, mean_series, 
                        label=legend_names.get(reward_type, reward_type),
                        color=color_map.get(reward_type, '#000000'),
                        linewidth=2.5,
                        zorder=3)
                
                # Confidence interval
                plt.fill_between(episodes, ci_lower, ci_upper,
                                alpha=0.25,
                                color=color_map.get(reward_type, '#000000'),
                                zorder=2)
        
        plt.title(f'{title} (95% Confidence Interval over 30 repetitions)', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add reference line at 0 for reward
        if metric_name == 'reward':
            plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(f'experiment_plots/{metric_name}_comparison_CI95.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()