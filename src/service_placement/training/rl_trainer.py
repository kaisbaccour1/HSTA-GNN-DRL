"""RL Trainer for service placement"""
import numpy as np
from typing import Dict
from ..config import ServicePlacementConfig
from ..environment.edge_environment import EdgeEnvironment
from ..agents.rl_agent import RLAgent


class RLTrainer:
    def __init__(self, env: EdgeEnvironment, agent: RLAgent, config: ServicePlacementConfig):
        self.env = env
        self.agent = agent
        self.config = config
        self.env.agent = agent
        
        # Performance tracking
        self.episode_rewards = []
        self.best_mean_reward = -float('inf')
        self.patience_counter = 0
    
    def train(self, num_episodes: int = 100) -> Dict:
        """Train RL agent"""
        print("🚀 Starting RL Training...")
        
        episode_rewards = []
        episode_avg_energies = []
        episode_avg_latencies = []
        
        for episode in range(num_episodes):
            state_dict = self.env.reset()
            self.env.set_agent(self.agent)
            
            if state_dict is None:
                continue
                
            state = state_dict['state_vector']
            predicted_services = state_dict['predicted_services']
            total_reward = 0
            steps = 0
            
            while True:
                # Select action
                actions = self.agent.select_action(state, predicted_services, explore=True)
                
                # Take step in environment
                next_state_dict, reward, done, info = self.env.step(actions)
                
                # Store transition (except for last episode which is in evaluation mode)
                if next_state_dict is not None:
                    next_state = next_state_dict['state_vector']
                    next_predicted_services = next_state_dict['predicted_services']
                    
                    # Store transition for each edge (using first edge's action for simplicity)
                    if actions:
                        first_action = list(actions.values())[0]
                        self.agent.buffer.push(state, first_action, reward, next_state, done)
                
                # Update agent (except for last episode)
                if len(self.agent.buffer) >= self.config.batch_size:
                    self.agent.update(self.config.batch_size)
                
                total_reward += reward
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state = next_state_dict['state_vector']
                predicted_services = next_state_dict['predicted_services']
            
            # Calculate average reward for this episode
            if steps > 0:
                episode_mean_reward = total_reward / steps
            else:
                episode_mean_reward = 0
            
            # Save best model (except for last episode)
            if episode < num_episodes - 1:
                self.agent.save_best_model(episode_mean_reward, episode)
            
            # Save checkpoint every 50 episodes (except last)
            if episode % 50 == 0 and episode < num_episodes - 1:
                self.agent.save_checkpoint(episode, episode_mean_reward)
            
            # Get episode metrics
            episode_metrics = self.env.get_episode_metrics(steps)
            episode_rewards.append(episode_mean_reward)
            episode_avg_energies.append(episode_metrics.get('avg_energy_per_step', 0))
            episode_avg_latencies.append(episode_metrics.get('avg_latency_per_step', 0))
            
            # Log progress
            if episode % 10 == 0:
                mode = "TRAINING"
                print(f"\n📈 Episode {episode:3d} | Mode: {mode} | "
                      f"Reward: {episode_mean_reward:7.2f} | "
                      f"Avg Energy: {episode_metrics.get('avg_energy_per_step', 0):7.2f} | "
                      f"Avg Latency: {episode_metrics.get('avg_latency_per_step', 0):5.2f} | "
                      f"Steps: {steps:3d}")
                
                # Print exploration rate
                print(f"   Exploration: ε={self.agent.epsilon:.3f} | "
                      f"Buffer: {len(self.agent.buffer)}/{self.config.buffer_size}")
        
        # Final summary
        self._print_final_summary(episode_rewards, episode_avg_energies, episode_avg_latencies)
        
        return {
            'rewards': episode_rewards,
            'avg_energies': episode_avg_energies,
            'avg_latencies': episode_avg_latencies,
            'best_reward': self.agent.best_reward,
            'best_episode': self.agent.best_episode
        }
    
    def _print_final_summary(self, rewards: list, energies: list, latencies: list):
        """Print final training summary"""
        print("\n" + "="*80)
        print("🎯 TRAINING SUMMARY")
        print("="*80)
        
        if rewards:
            print(f"📊 Reward Statistics:")
            print(f"   Average reward: {np.mean(rewards):.4f}")
            print(f"   Best reward: {max(rewards):.4f}")
            print(f"   Final reward: {rewards[-1]:.4f}")
        
        if energies:
            print(f"\n⚡ Energy Statistics:")
            print(f"   Average energy per step: {np.mean(energies):.2f} J")
            print(f"   Minimum energy: {min(energies):.2f} J")
            print(f"   Maximum energy: {max(energies):.2f} J")
        
        if latencies:
            print(f"\n⏱️  Latency Statistics:")
            print(f"   Average latency per step: {np.mean(latencies):.4f} s")
            print(f"   Minimum latency: {min(latencies):.4f} s")
            print(f"   Maximum latency: {max(latencies):.4f} s")
        
        print(f"\n🤖 Agent Statistics:")
        print(f"   Best episode: {self.agent.best_episode}")
        print(f"   Best reward: {self.agent.best_reward:.4f}")
        print(f"   Final exploration rate: {self.agent.epsilon:.4f}")
        print(f"   Buffer size: {len(self.agent.buffer)}/{self.config.buffer_size}")
        
        # Loss statistics
        if self.agent.actor_losses:
            print(f"\n📉 Loss Statistics:")
            print(f"   Average actor loss: {np.mean(self.agent.actor_losses[-100:]):.6f}")
            print(f"   Average critic loss: {np.mean(self.agent.critic_losses[-100:]):.6f}")
        
        print("="*80)
    
    def evaluate(self, num_episodes: int = 10, use_eval_set: bool = True) -> Dict:
        """Evaluate trained agent"""
        print("\n" + "="*80)
        print("🧪 EVALUATION MODE")
        print("="*80)
        
        episode_rewards = []
        episode_energies = []
        episode_latencies = []
        
        for episode in range(num_episodes):
            # Reset environment with evaluation set
            state_dict = self.env.reset(use_eval_set=use_eval_set)
            
            if state_dict is None:
                continue
                
            state = state_dict['state_vector']
            predicted_services = state_dict['predicted_services']
            total_reward = 0
            steps = 0
            
            while True:
                # Select action without exploration
                actions = self.agent.select_action(state, predicted_services, explore=False)
                
                # Take step
                next_state_dict, reward, done, info = self.env.step(actions)
                
                total_reward += reward
                steps += 1
                
                if done or next_state_dict is None:
                    break
                    
                state = next_state_dict['state_vector']
                predicted_services = next_state_dict['predicted_services']
            
            # Store metrics
            if steps > 0:
                episode_rewards.append(total_reward / steps)
                episode_energies.append(self.env.episode_energy / steps)
                episode_latencies.append(self.env.episode_latency / steps)
            
            print(f"📊 Evaluation Episode {episode}: "
                  f"Reward={total_reward/steps if steps > 0 else 0:.4f}, "
                  f"Energy={self.env.episode_energy/steps if steps > 0 else 0:.2f}J, "
                  f"Latency={self.env.episode_latency/steps if steps > 0 else 0:.4f}s")
        
        # Evaluation summary
        print("\n" + "="*80)
        print("📈 EVALUATION SUMMARY")
        print("="*80)
        
        if episode_rewards:
            print(f"Average Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
        
        if episode_energies:
            print(f"Average Energy: {np.mean(episode_energies):.2f}J ± {np.std(episode_energies):.2f}J")
        
        if episode_latencies:
            print(f"Average Latency: {np.mean(episode_latencies):.4f}s ± {np.std(episode_latencies):.4f}s")
        
        print("="*80)
        
        return {
            'rewards': episode_rewards,
            'energies': episode_energies,
            'latencies': episode_latencies
        }