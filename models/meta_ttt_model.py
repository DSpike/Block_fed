#!/usr/bin/env python3
"""
Meta TTT Model for Zero-Day Intrusion Detection
Combines Meta-learning (FOMAML) with Test-Time Training (TTT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MetaTTTModel(nn.Module):
    """
    Meta TTT Model: Combines Meta-learning with Test-Time Training
    
    This model:
    1. Uses FOMAML meta-learning to learn how to adapt quickly
    2. Performs Test-Time Training to adapt to zero-day attacks during inference
    3. Processes static features (no temporal modeling for UNSW-NB15)
    """
    
    def __init__(self, base_model: nn.Module, meta_learning_method: str = 'fomaml',
                 enable_meta_learning: bool = True, enable_ttt_adaptation: bool = True,
                 inner_lr: float = 0.01, inner_steps: int = 5, ttt_steps: int = 10):
        super().__init__()
        
        self.base_model = base_model
        self.meta_learning_method = meta_learning_method
        self.enable_meta_learning = enable_meta_learning
        self.enable_ttt_adaptation = enable_ttt_adaptation
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.ttt_steps = ttt_steps
        
        # Initialize meta-learner if enabled
        if self.enable_meta_learning:
            if meta_learning_method == 'fomaml':
                from models.fomaml_reptile_meta_learner import FOMAMLMetaLearner
                self.meta_learner = FOMAMLMetaLearner(
                    model=self.base_model,
                    inner_lr=inner_lr,
                    num_inner_steps=inner_steps
                )
            else:
                raise ValueError(f"Unsupported meta-learning method: {meta_learning_method}")
        else:
            self.meta_learner = None
        
        # TTT adaptation layers (shallow adaptation)
        if self.enable_ttt_adaptation:
            self.ttt_adapter = nn.Sequential(
                nn.Linear(self.base_model.latent_dim, self.base_model.latent_dim // 2),
                nn.ReLU(),
                nn.Linear(self.base_model.latent_dim // 2, self.base_model.latent_dim)
            )
        else:
            self.ttt_adapter = None
        
        logger.info(f"Meta TTT Model initialized:")
        logger.info(f"  - Meta-learning: {self.enable_meta_learning} ({meta_learning_method})")
        logger.info(f"  - TTT adaptation: {self.enable_ttt_adaptation}")
        logger.info(f"  - Base model: {type(self.base_model).__name__}")
    
    def forward(self, x, use_ttt_adapter=False):
        """
        Forward pass with optional TTT adaptation
        
        Args:
            x: Input features [batch_size, input_dim]
            use_ttt_adapter: Whether to use TTT adaptation layers
        """
        # Ensure input is 2D (static features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Add sequence dimension for base model (sequence_length=1 for static features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Forward pass through base model
        outputs = self.base_model(x)
        
        # Apply TTT adaptation if enabled and requested
        if use_ttt_adapter and self.ttt_adapter is not None:
            adapted_latent = self.ttt_adapter(outputs['latent'])
            # Re-classify with adapted embeddings
            adapted_logits = self.base_model.attack_classifier(adapted_latent)
            outputs['attack_logits'] = adapted_logits
        
        return outputs
    
    def meta_train_step(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Dict:
        """
        Meta-training step using FOMAML
        
        Args:
            support_data: Support set features
            support_labels: Support set labels
            query_data: Query set features  
            query_labels: Query set labels
            
        Returns:
            Dictionary with meta-training results
        """
        if not self.enable_meta_learning or self.meta_learner is None:
            raise RuntimeError("Meta-learning not enabled or meta-learner not initialized")
        
        # Use the meta-learner to perform meta-training step
        return self.meta_learner.meta_train_step(
            support_data, support_labels, query_data, query_labels
        )
    
    def test_time_adapt(self, test_data: torch.Tensor, adaptation_steps: int = None) -> nn.Module:
        """
        Test-Time Training adaptation for zero-day attacks
        
        Args:
            test_data: Unlabeled test data for adaptation
            adaptation_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        if not self.enable_ttt_adaptation:
            logger.warning("TTT adaptation not enabled, returning original model")
            return self
        
        if adaptation_steps is None:
            adaptation_steps = self.ttt_steps
        
        # Create adapted model
        adapted_model = copy.deepcopy(self)
        adapted_model.train()
        
        # Setup TTT optimizer (only adapt TTT adapter layers)
        if adapted_model.ttt_adapter is not None:
            ttt_optimizer = torch.optim.Adam(adapted_model.ttt_adapter.parameters(), lr=self.inner_lr)
        else:
            # If no TTT adapter, adapt base model parameters
            ttt_optimizer = torch.optim.Adam(adapted_model.base_model.parameters(), lr=self.inner_lr)
        
        logger.info(f"Starting TTT adaptation for {adaptation_steps} steps")
        
        for step in range(adaptation_steps):
            ttt_optimizer.zero_grad()
            
            # Forward pass with TTT adaptation
            outputs = adapted_model(test_data, use_ttt_adapter=True)
            
            # Self-supervised losses for TTT (no labels needed)
            losses = self._compute_ttt_losses(outputs, test_data)
            
            # Backward pass
            total_loss = sum(losses.values())
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            
            ttt_optimizer.step()
            
            if step % 5 == 0:
                logger.info(f"TTT Step {step}: Loss={total_loss.item():.4f}")
        
        logger.info(f"TTT adaptation completed")
        return adapted_model
    
    def _compute_ttt_losses(self, outputs: Dict, test_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute self-supervised losses for TTT (no labels needed)
        """
        latent = outputs['latent']
        attack_logits = outputs['attack_logits']
        
        losses = {}
        
        # 1. Consistency Loss: Encourage consistent predictions
        probs = F.softmax(attack_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        losses['consistency'] = torch.mean(entropy)
        
        # 2. Feature Regularization: Prevent feature collapse
        latent_norm = torch.norm(latent, p=2, dim=1)
        losses['feature_reg'] = torch.mean((latent_norm - 1.0) ** 2)
        
        # 3. Entropy Regularization: Encourage confident predictions
        losses['entropy'] = -torch.mean(entropy)
        
        # 4. Reconstruction Loss: Encourage meaningful latent representations
        # Use the base model's reconstruction if available
        if 'reconstruction' in outputs:
            losses['reconstruction'] = F.mse_loss(outputs['reconstruction'], test_data)
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=test_data.device)
        
        return losses
    
    def train_standard(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                      val_data: torch.Tensor, val_labels: torch.Tensor, 
                      epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Standard training method (fallback when meta-learning is not available)
        """
        logger.info("Using standard training (meta-learning not available)")
        
        # Use base model's training method
        if hasattr(self.base_model, 'train_standard'):
            return self.base_model.train_standard(
                train_data, train_labels, val_data, val_labels, epochs, batch_size, learning_rate
            )
        else:
            raise NotImplementedError("Base model does not support standard training")
    
    def has_meta_learner(self) -> bool:
        """Check if model has meta-learner"""
        return self.enable_meta_learning and self.meta_learner is not None
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent embeddings for analysis"""
        outputs = self.forward(x)
        return outputs['latent']
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores for zero-day detection"""
        outputs = self.forward(x)
        probs = F.softmax(outputs['attack_logits'], dim=1)
        # Return probability of attack class (class 1)
        return probs[:, 1]
    
    def state_dict(self):
        """Return state dict including all components"""
        state_dict = {
            'base_model': self.base_model.state_dict(),
            'meta_learning_method': self.meta_learning_method,
            'enable_meta_learning': self.enable_meta_learning,
            'enable_ttt_adaptation': self.enable_ttt_adaptation,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
            'ttt_steps': self.ttt_steps
        }
        
        if self.ttt_adapter is not None:
            state_dict['ttt_adapter'] = self.ttt_adapter.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load state dict including all components"""
        self.base_model.load_state_dict(state_dict['base_model'])
        self.meta_learning_method = state_dict['meta_learning_method']
        self.enable_meta_learning = state_dict['enable_meta_learning']
        self.enable_ttt_adaptation = state_dict['enable_ttt_adaptation']
        self.inner_lr = state_dict['inner_lr']
        self.inner_steps = state_dict['inner_steps']
        self.ttt_steps = state_dict['ttt_steps']
        
        if 'ttt_adapter' in state_dict and self.ttt_adapter is not None:
            self.ttt_adapter.load_state_dict(state_dict['ttt_adapter'])


