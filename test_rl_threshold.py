#!/usr/bin/env python3
"""
Test script for RL-based thresholding system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class ThresholdAgent:
    """
    Reinforcement Learning agent for learning optimal TTT threshold
    Uses a simple neural network to map state to threshold value
    """
    
    def __init__(self, state_dim=2, hidden_dim=32, learning_rate=0.001):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Simple neural network for threshold prediction
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = []  # Store (state, action, reward) tuples
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Track performance for reward calculation
        self.adaptation_history = []
        self.threshold_history = []
        
    def get_threshold(self, state):
        """
        Get threshold based on current state using RL agent
        
        Args:
            state: torch.Tensor of shape [2] containing [mean_confidence, adaptation_success_rate]
            
        Returns:
            threshold: float between 0 and 1
        """
        with torch.no_grad():
            if random.random() < self.epsilon:
                # Exploration: random threshold
                threshold = random.uniform(0.1, 0.9)
            else:
                # Exploitation: use neural network
                threshold = self.network(state.unsqueeze(0)).item()
                # Ensure threshold is in reasonable range
                threshold = max(0.1, min(0.9, threshold))
        
        self.threshold_history.append(threshold)
        return threshold
    
    def update(self, state, threshold, adaptation_success_rate, accuracy_improvement):
        """
        Update the agent based on adaptation results
        
        Args:
            state: Current state [mean_confidence, adaptation_success_rate]
            threshold: Threshold used
            adaptation_success_rate: Success rate of adaptation
            accuracy_improvement: Improvement in accuracy after TTT
        """
        # Calculate reward based on adaptation success and accuracy improvement
        reward = self._calculate_reward(adaptation_success_rate, accuracy_improvement)
        
        # Store experience
        self.memory.append((state, threshold, reward))
        
        # Update adaptation history
        self.adaptation_history.append(adaptation_success_rate)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Train the network if we have enough experiences
        if len(self.memory) >= 10:
            self._train_network()
    
    def _calculate_reward(self, adaptation_success_rate, accuracy_improvement):
        """
        Calculate reward based on adaptation performance
        
        Args:
            adaptation_success_rate: Rate of successful adaptations
            accuracy_improvement: Improvement in accuracy
            
        Returns:
            reward: float reward value
        """
        # Base reward from adaptation success
        base_reward = adaptation_success_rate * 10.0
        
        # Bonus for accuracy improvement
        accuracy_bonus = accuracy_improvement * 20.0
        
        # Penalty for too many or too few samples selected
        # (This would need to be passed as a parameter in real implementation)
        selection_penalty = 0.0  # Placeholder
        
        total_reward = base_reward + accuracy_bonus - selection_penalty
        return total_reward
    
    def _train_network(self):
        """
        Train the neural network using stored experiences
        """
        if len(self.memory) < 10:
            return
        
        # Sample recent experiences
        recent_memories = self.memory[-10:]
        
        states = torch.stack([mem[0] for mem in recent_memories])
        thresholds = torch.tensor([mem[1] for mem in recent_memories]).unsqueeze(1)
        rewards = torch.tensor([mem[2] for mem in recent_memories]).unsqueeze(1)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Predict thresholds
        predicted_thresholds = self.network(states)
        
        # Calculate loss (MSE between predicted and actual thresholds, weighted by rewards)
        loss = F.mse_loss(predicted_thresholds, thresholds, reduction='none')
        weighted_loss = (loss * (1 + rewards)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Clear old memories to prevent memory overflow
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]
    
    def get_adaptation_success_rate(self):
        """Get current adaptation success rate"""
        if not self.adaptation_history:
            return 0.5  # Default value
        return np.mean(self.adaptation_history[-10:])  # Average of last 10 adaptations
    
    def reset(self):
        """Reset the agent for new training session"""
        self.memory = []
        self.adaptation_history = []
        self.threshold_history = []
        self.epsilon = 0.1

def test_rl_thresholding():
    """Test the RL-based thresholding system"""
    print("🧪 Testing RL-based thresholding system...")
    
    # Create agent
    agent = ThresholdAgent()
    
    # Test 1: Basic threshold generation
    print("\n1. Testing basic threshold generation...")
    state = torch.tensor([0.5, 0.7])  # [mean_confidence, adaptation_success_rate]
    threshold = agent.get_threshold(state)
    print(f"   Generated threshold: {threshold:.4f}")
    assert 0.1 <= threshold <= 0.9, "Threshold should be between 0.1 and 0.9"
    
    # Test 2: Multiple threshold generations
    print("\n2. Testing multiple threshold generations...")
    thresholds = []
    for i in range(10):
        state = torch.tensor([0.3 + i*0.1, 0.5 + i*0.05])
        threshold = agent.get_threshold(state)
        thresholds.append(threshold)
        print(f"   State {i+1}: {state.tolist()} -> Threshold: {threshold:.4f}")
    
    # Test 3: Agent learning
    print("\n3. Testing agent learning...")
    for i in range(20):
        state = torch.tensor([0.4 + i*0.02, 0.6 + i*0.01])
        threshold = agent.get_threshold(state)
        
        # Simulate adaptation results
        adaptation_success = 0.7 + 0.1 * np.sin(i * 0.5)  # Varying success rate
        accuracy_improvement = 0.1 + 0.05 * np.cos(i * 0.3)  # Varying improvement
        
        # Update agent
        agent.update(state, threshold, adaptation_success, accuracy_improvement)
        
        if i % 5 == 0:
            print(f"   Step {i+1}: Threshold={threshold:.4f}, Success={adaptation_success:.3f}, Improvement={accuracy_improvement:.3f}")
    
    # Test 4: Check learning progress
    print("\n4. Checking learning progress...")
    print(f"   Memory size: {len(agent.memory)}")
    print(f"   Adaptation history: {len(agent.adaptation_history)}")
    print(f"   Current epsilon: {agent.epsilon:.4f}")
    print(f"   Average adaptation success: {agent.get_adaptation_success_rate():.3f}")
    
    print("\n✅ RL-based thresholding system test completed successfully!")
    return True

if __name__ == "__main__":
    test_rl_thresholding()
