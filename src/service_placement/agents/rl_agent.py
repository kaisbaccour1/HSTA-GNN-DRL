"""RL Agent for service placement"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
import os
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple
from ..config import ServicePlacementConfig
from ..models.actor_critic import Actor, Critic
from ..agents.buffer import ReplayBuffer


Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class RLAgent:
    def __init__(self, state_dim: int, action_dim: int, config: ServicePlacementConfig, device):
        self.config = config
        self.device = device
        self.max_action_dim = action_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Networks
        self.actor = Actor(state_dim, self.max_action_dim, config.hidden_dim).to(device)
        self.critic = Critic(state_dim, config.hidden_dim).to(device)
        
        # Target networks
        self.actor_target = Actor(state_dim, action_dim, config.hidden_dim).to(device)
        self.critic_target = Critic(state_dim, config.hidden_dim).to(device)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size)
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.rewards = []
        
        # Exploration
        self.epsilon = config.epsilon
        self.min_epsilon = config.min_epsilon
        self.epsilon_decay = config.epsilon_decay
        
        # Best model tracking
        self.best_reward = -float('inf')
        self.best_actor_state = None
        self.best_critic_state = None
        self.best_episode = 0
    
    def select_action(self, state: np.ndarray, predicted_services: torch.Tensor, 
                     explore: bool = True) -> Dict[int, int]:
        """Select action for each edge"""
        if explore and random.random() < self.epsilon:
            return self._random_action_based_on_predictions(predicted_services)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_logits = self.actor.network(state_tensor)  # [1, action_dim]
        
        actions = {}
        for edge_id in range(predicted_services.shape[0]):
            edge_predictions = predicted_services[edge_id]
            valid_actions = self._get_valid_actions_for_edge(edge_predictions)
            
            if not valid_actions:
                actions[edge_id] = 0  # No valid action
            else:
                edge_logits = all_logits[0].clone()  # [action_dim]
                
                # Create mask: -inf for invalid actions
                mask = torch.ones_like(edge_logits) * float('-inf')
                for valid_action in valid_actions:
                    mask[valid_action] = 0
                
                # Apply mask to logits
                masked_logits = edge_logits + mask
                
                # Convert to probabilities
                action_probs = F.softmax(masked_logits, dim=-1)
                action_probs = (action_probs + 1e-8) / (1.0 + 1e-8 * action_probs.size(-1))
                
                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                selected_action = action_dist.sample().item()
                actions[edge_id] = selected_action
        
        return actions
    
    def _get_valid_actions_for_edge(self, edge_predictions: torch.Tensor) -> List[int]:
        """
        Return list of valid actions for edge based on its predictions
        edge_predictions: tensor [n_services] with 0/1
        """
        predicted_indices = torch.where(edge_predictions == 1)[0].tolist()
        
        if not predicted_indices:
            return [0]  # Only "deploy nothing" action
        
        # Generate all valid combinations of predicted services
        valid_actions = []
        n_predicted = len(predicted_indices)
        
        # For k predicted services, we have 2^k possible actions
        for action_mask in range(1 << n_predicted):  # 0 to 2^k - 1
            # Convert action mask to global action
            global_action = 0
            for i in range(n_predicted):
                if action_mask & (1 << i):
                    service_idx = predicted_indices[i]
                    global_action |= (1 << service_idx)
            valid_actions.append(global_action)
        
        return valid_actions
    
    def _random_action_based_on_predictions(self, predicted_services: torch.Tensor) -> Dict[int, int]:
        """Random action but only on predicted services"""
        actions = {}
        for edge_id in range(predicted_services.shape[0]):
            edge_predictions = predicted_services[edge_id]
            valid_actions = self._get_valid_actions_for_edge(edge_predictions)
            actions[edge_id] = random.choice(valid_actions)
        
        return actions
    
    def action_to_services(self, action: int, edge_predictions: Optional[torch.Tensor] = None) -> List[int]:
        """Convert action to list of services to deploy"""
        services = []
        for i in range(self.config.n_services):
            if action & (1 << i):
                # If edge_predictions provided, verify service was predicted
                if edge_predictions is None or edge_predictions[i] == 1:
                    services.append(i)
        return services
    
    def update(self, batch_size: int):
        """Update agent with experience replay"""
        if len(self.buffer) < batch_size:
            return
        
        transitions = self.buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare batch tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_action_probs = self.actor_target(next_state_batch)
            next_q_values = self.critic_target(next_state_batch)
            target_q = reward_batch + (1 - done_batch) * self.config.gamma * next_q_values
        
        current_q = self.critic(state_batch)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm_critic)
        self.critic_optimizer.step()
        
        # Actor update
        actor_logits = self.actor.network(state_batch)
        actor_probs = F.softmax(actor_logits, dim=-1)
        actor_probs = (actor_probs + 1e-8) / (1.0 + 1e-8 * actor_probs.size(-1))
        
        actor_log_probs = torch.log(actor_probs.gather(1, action_batch))
        
        with torch.no_grad():
            advantage = target_q - current_q
        
        actor_loss = -(actor_log_probs * advantage.detach()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm_actor)
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.actor_target, self.actor, self.config.tau)
        self.soft_update(self.critic_target, self.critic, self.config.tau)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
    
    def soft_update(self, target, source, tau):
        """Soft update of target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_best_model(self, episode_reward: float, episode: int):
        """Save weights of best model"""
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = episode
            
            # Save network states (deep copy)
            self.best_actor_state = {
                'actor': copy.deepcopy(self.actor.state_dict()),
                'actor_target': copy.deepcopy(self.actor_target.state_dict()),
                'actor_optimizer': copy.deepcopy(self.actor_optimizer.state_dict())
            }
            
            self.best_critic_state = {
                'critic': copy.deepcopy(self.critic.state_dict()),
                'critic_target': copy.deepcopy(self.critic_target.state_dict()),
                'critic_optimizer': copy.deepcopy(self.critic_optimizer.state_dict())
            }
            
            print(f"🎉 NEW BEST MODEL! Reward: {episode_reward:.2f} (Episode {episode})")
    
    def load_best_model(self):
        """Load saved best model"""
        if self.best_actor_state is not None and self.best_critic_state is not None:
            self.actor.load_state_dict(self.best_actor_state['actor'])
            self.actor_target.load_state_dict(self.best_actor_state['actor_target'])
            self.actor_optimizer.load_state_dict(self.best_actor_state['actor_optimizer'])
            
            self.critic.load_state_dict(self.best_critic_state['critic'])
            self.critic_target.load_state_dict(self.best_critic_state['critic_target'])
            self.critic_optimizer.load_state_dict(self.best_critic_state['critic_optimizer'])
            
            print(f"✅ Best model loaded (Reward: {self.best_reward:.2f}, Episode: {self.best_episode})")
        else:
            print("❌ No saved best model found")
    
    def save_checkpoint(self, episode: int, reward: float, path: str = 'checkpoints/'):
        """Save complete checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'reward': reward,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_target_state': self.actor_target.state_dict(),
            'critic_target_state': self.critic_target.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode
        }
        
        torch.save(checkpoint, f'{path}checkpoint_episode_{episode}.pth')
        print(f"💾 Checkpoint saved: {path}checkpoint_episode_{episode}.pth")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            self.actor.load_state_dict(checkpoint['actor_state'])
            self.critic.load_state_dict(checkpoint['critic_state'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
            
            self.epsilon = checkpoint['epsilon']
            self.best_reward = checkpoint['best_reward']
            self.best_episode = checkpoint['best_episode']
            
            print(f"✅ Checkpoint loaded: {checkpoint_path} (Episode {checkpoint['episode']}, Reward: {checkpoint['reward']:.2f})")
            return checkpoint['episode']
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 0
    
    def get_action_stats(self) -> Dict:
        """Get statistics about actions taken"""
        if not self.buffer.buffer:
            return {}
        
        actions = [t.action for t in self.buffer.buffer]
        unique_actions = set(actions)
        
        return {
            'total_actions': len(actions),
            'unique_actions': len(unique_actions),
            'action_distribution': {a: actions.count(a) for a in unique_actions}
        }


class RewardSpecificAgent(RLAgent):
    def __init__(self, state_dim: int, action_dim: int, config: ServicePlacementConfig, 
                 device, reward_type: str):
        super().__init__(state_dim, action_dim, config, device)
        self.reward_type = reward_type
        from ..utils.reward_calculator import RewardCalculator
        self.reward_calculator = RewardCalculator()
    
    def calculate_reward(self, energy: float, latency: float) -> float:
        """Calculates reward according to specific type"""
        if self.reward_type == 'energy_only':
            return self.reward_calculator.energy_only_reward(energy, latency)
        elif self.reward_type == 'latency_only':
            return self.reward_calculator.latency_only_reward(energy, latency)
        elif self.reward_type == 'energy_latency_combined':
            return self.reward_calculator.energy_latency_combined_reward(energy, latency)
        else:
            # Default combined reward
            return self.reward_calculator.energy_latency_combined_reward(energy, latency)