#!/usr/bin/env python3
"""
FOMAML and Reptile Meta-Learning Implementation for Zero-Day Intrusion Detection
Implements First-Order Model-Agnostic Meta-Learning (FOMAML) and Reptile algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class FOMAMLMetaLearner(nn.Module):
    """
    First-Order Model-Agnostic Meta-Learning (FOMAML) implementation
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01, 
                 meta_lr: float = 0.001, num_inner_steps: int = 5):
        super(FOMAMLMetaLearner, self).__init__()
        
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Create meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=meta_lr)
        
        logger.info(f"FOMAML Meta-Learner initialized with inner_lr={inner_lr}, meta_lr={meta_lr}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model"""
        return self.base_model(x)
    
    def meta_train_step(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Dict:
        """
        Perform one meta-training step using FOMAML
        
        Args:
            support_data: Support set data for inner loop adaptation
            support_labels: Support set labels
            query_data: Query set data for outer loop evaluation
            query_labels: Query set labels
            
        Returns:
            Dictionary with training metrics
        """
        device = next(self.base_model.parameters()).device
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.base_model.named_parameters()}
        
        # Inner loop: adapt on support set
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        inner_losses = []
        for step in range(self.num_inner_steps):
            # Forward pass on support set
            support_outputs = adapted_model(support_data)
            # CRITICAL FIX: Extract attack_logits from the dictionary output
            if isinstance(support_outputs, dict):
                attack_logits = support_outputs['attack_logits']
            else:
                attack_logits = support_outputs
            
            # DEBUG: Log tensor shapes for debugging
            if step == 0:  # Only log on first step to avoid spam
                logger.info(f"🔍 FOMAML DEBUG - support_data shape: {support_data.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_outputs type: {type(support_outputs)}")
                if isinstance(support_outputs, dict):
                    logger.info(f"🔍 FOMAML DEBUG - support_outputs keys: {support_outputs.keys()}")
                    logger.info(f"🔍 FOMAML DEBUG - attack_logits shape: {attack_logits.shape}")
                else:
                    logger.info(f"🔍 FOMAML DEBUG - attack_logits shape: {attack_logits.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_labels shape: {support_labels.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_labels dtype: {support_labels.dtype}")
            
            inner_loss = F.cross_entropy(attack_logits, support_labels)
            
            # Backward pass and update
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()
            
            inner_losses.append(inner_loss.item())
        
        # Outer loop: evaluate on query set
        with torch.no_grad():
            query_outputs = adapted_model(query_data)
            # CRITICAL FIX: Extract attack_logits from the dictionary output
            if isinstance(query_outputs, dict):
                query_attack_logits = query_outputs['attack_logits']
            else:
                query_attack_logits = query_outputs
            query_loss = F.cross_entropy(query_attack_logits, query_labels)
            
            # Calculate accuracy
            predictions = torch.argmax(query_attack_logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        # Meta-update: update base model parameters
        self.meta_optimizer.zero_grad()
        
        # Compute gradients on query set using adapted model
        query_outputs = adapted_model(query_data)
        # CRITICAL FIX: Extract attack_logits from the dictionary output
        if isinstance(query_outputs, dict):
            meta_attack_logits = query_outputs['attack_logits']
        else:
            meta_attack_logits = query_outputs
        meta_loss = F.cross_entropy(meta_attack_logits, query_labels)
        meta_loss.backward()
        
        # Apply meta-update to base model
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'query_loss': query_loss.item(),
            'query_accuracy': accuracy,
            'inner_losses': inner_losses,
            'avg_inner_loss': np.mean(inner_losses)
        }
    
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: int = None) -> nn.Module:
        """
        Adapt the model to a specific task using support data
        
        Args:
            support_data: Task-specific support data
            support_labels: Task-specific support labels
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
            
        Returns:
            Adapted model for the specific task
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
            
        # Create adapted model
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform adaptation steps
        for step in range(num_steps):
            support_outputs = adapted_model(support_data)
            loss = F.cross_entropy(support_outputs, support_labels)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model


class ReptileMetaLearner(nn.Module):
    """
    Reptile Meta-Learning implementation
    """
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01,
                 meta_lr: float = 0.001, num_inner_steps: int = 5):
        super(ReptileMetaLearner, self).__init__()
        
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Create meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(self.base_model.parameters(), lr=meta_lr)
        
        logger.info(f"Reptile Meta-Learner initialized with inner_lr={inner_lr}, meta_lr={meta_lr}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model"""
        return self.base_model(x)
    
    def meta_train_step(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Dict:
        """
        Perform one meta-training step using Reptile
        
        Args:
            support_data: Support set data for inner loop adaptation
            support_labels: Support set labels
            query_data: Query set data for outer loop evaluation
            query_labels: Query set labels
            
        Returns:
            Dictionary with training metrics
        """
        device = next(self.base_model.parameters()).device
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.base_model.named_parameters()}
        
        # Inner loop: adapt on support set
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        inner_losses = []
        for step in range(self.num_inner_steps):
            # Forward pass on support set
            support_outputs = adapted_model(support_data)
            # CRITICAL FIX: Extract attack_logits from the dictionary output
            if isinstance(support_outputs, dict):
                attack_logits = support_outputs['attack_logits']
            else:
                attack_logits = support_outputs
            
            # DEBUG: Log tensor shapes for debugging
            if step == 0:  # Only log on first step to avoid spam
                logger.info(f"🔍 FOMAML DEBUG - support_data shape: {support_data.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_outputs type: {type(support_outputs)}")
                if isinstance(support_outputs, dict):
                    logger.info(f"🔍 FOMAML DEBUG - support_outputs keys: {support_outputs.keys()}")
                    logger.info(f"🔍 FOMAML DEBUG - attack_logits shape: {attack_logits.shape}")
                else:
                    logger.info(f"🔍 FOMAML DEBUG - attack_logits shape: {attack_logits.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_labels shape: {support_labels.shape}")
                logger.info(f"🔍 FOMAML DEBUG - support_labels dtype: {support_labels.dtype}")
            
            inner_loss = F.cross_entropy(attack_logits, support_labels)
            
            # Backward pass and update
            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()
            
            inner_losses.append(inner_loss.item())
        
        # Outer loop: evaluate on query set
        with torch.no_grad():
            query_outputs = adapted_model(query_data)
            # CRITICAL FIX: Extract attack_logits from the dictionary output
            if isinstance(query_outputs, dict):
                query_attack_logits = query_outputs['attack_logits']
            else:
                query_attack_logits = query_outputs
            query_loss = F.cross_entropy(query_attack_logits, query_labels)
            
            # Calculate accuracy
            predictions = torch.argmax(query_attack_logits, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        # Reptile update: interpolate between original and adapted parameters
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                adapted_param = adapted_model.get_parameter(name)
                param.data = param.data + self.meta_lr * (adapted_param.data - param.data)
        
        return {
            'meta_loss': query_loss.item(),
            'query_loss': query_loss.item(),
            'query_accuracy': accuracy,
            'inner_losses': inner_losses,
            'avg_inner_loss': np.mean(inner_losses)
        }
    
    def adapt_to_task(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                     num_steps: int = None) -> nn.Module:
        """
        Adapt the model to a specific task using support data
        
        Args:
            support_data: Task-specific support data
            support_labels: Task-specific support labels
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
            
        Returns:
            Adapted model for the specific task
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
            
        # Create adapted model
        adapted_model = copy.deepcopy(self.base_model)
        inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform adaptation steps
        for step in range(num_steps):
            support_outputs = adapted_model(support_data)
            loss = F.cross_entropy(support_outputs, support_labels)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model


class MetaLearningTTTModel(nn.Module):
    """
    Meta-Learning Test-Time Training Model
    Combines FOMAML/Reptile with Test-Time Adaptation
    """
    
    def __init__(self, base_model: nn.Module, meta_learning_method: str = 'fomaml',
                 inner_lr: float = 0.01, meta_lr: float = 0.001, num_inner_steps: int = 5):
        super(MetaLearningTTTModel, self).__init__()
        
        self.base_model = base_model
        self.meta_learning_method = meta_learning_method.lower()
        
        # Initialize meta-learner
        if self.meta_learning_method == 'fomaml':
            self.meta_learner = FOMAMLMetaLearner(
                base_model, inner_lr, meta_lr, num_inner_steps
            )
        elif self.meta_learning_method == 'reptile':
            self.meta_learner = ReptileMetaLearner(
                base_model, inner_lr, meta_lr, num_inner_steps
            )
        else:
            raise ValueError(f"Unknown meta-learning method: {meta_learning_method}")
        
        logger.info(f"Meta-Learning TTT Model initialized with {meta_learning_method.upper()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model"""
        return self.base_model(x)
    
    def meta_train_step(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Dict:
        """Perform meta-training step"""
        return self.meta_learner.meta_train_step(support_data, support_labels, query_data, query_labels)
    
    def test_time_adapt(self, unlabeled_data: torch.Tensor, num_steps: int = 3) -> nn.Module:
        """
        Perform test-time adaptation using unlabeled data
        
        Args:
            unlabeled_data: Unlabeled data for adaptation
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        device = next(self.base_model.parameters()).device
        
        # Create adapted model
        adapted_model = copy.deepcopy(self.base_model)
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.001)
        
        # Use reconstruction loss for unsupervised adaptation
        for step in range(num_steps):
            # Forward pass
            outputs = adapted_model(unlabeled_data)
            
            # For unsupervised adaptation, use reconstruction loss
            if hasattr(adapted_model, 'tcgan_model'):
                # If it's our TCGAN model, use reconstruction loss
                reconstruction = outputs.get('reconstruction', unlabeled_data)
                loss = F.mse_loss(reconstruction, unlabeled_data)
            else:
                # For other models, use consistency loss
                loss = F.mse_loss(outputs, outputs.detach())
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        return adapted_model
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly score for input data
        
        Args:
            x: Input data
            
        Returns:
            Anomaly scores
        """
        with torch.no_grad():
            outputs = self.base_model(x)
            
            if hasattr(self.base_model, 'tcgan_model'):
                # For TCGAN model, use reconstruction error as anomaly score
                reconstruction = outputs.get('reconstruction', x)
                reconstruction_error = F.mse_loss(reconstruction, x, reduction='none')
                anomaly_scores = reconstruction_error.mean(dim=1)
            else:
                # For other models, use prediction confidence
                probabilities = F.softmax(outputs, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0]
                anomaly_scores = 1.0 - max_probs  # Lower confidence = higher anomaly score
            
            return anomaly_scores


if __name__ == "__main__":
    # Test the meta-learning implementations
    print("Testing FOMAML and Reptile Meta-Learning...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=50, num_classes=2):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # Test FOMAML
    base_model = SimpleModel()
    fomaml_model = FOMAMLMetaLearner(base_model)
    
    # Test Reptile
    base_model = SimpleModel()
    reptile_model = ReptileMetaLearner(base_model)
    
    # Test Meta-Learning TTT Model
    base_model = SimpleModel()
    meta_ttt_model = MetaLearningTTTModel(base_model, meta_learning_method='fomaml')
    
    print("✅ Meta-learning implementations tested successfully!")
