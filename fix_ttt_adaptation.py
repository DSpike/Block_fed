#!/usr/bin/env python3
"""
Fix TTT Adaptation Issues
Corrects the loss functions and adaptation strategy for better performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class FixedTTTAdapter:
    """
    Fixed TTT adapter with corrected loss functions and adaptation strategy
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        
        # Fixed TTT parameters
        self.ttt_lr = 0.0001  # Much lower learning rate to avoid destabilization
        self.ttt_steps = 10   # Fewer steps to prevent overfitting
        self.warmup_steps = 2  # Warmup phase
        
        # Fixed loss weights
        self.recon_weight = 0.1
        self.consistency_weight = 0.3
        self.entropy_weight = 0.1
        self.diversity_weight = 0.1
        
    def adapt_model(self, test_x: torch.Tensor, steps: Optional[int] = None) -> nn.Module:
        """
        Perform fixed TTT adaptation with corrected loss functions
        
        Args:
            test_x: Unlabeled test data for adaptation
            steps: Number of adaptation steps (default: self.ttt_steps)
            
        Returns:
            Adapted model
        """
        if steps is None:
            steps = self.ttt_steps
            
        # Create a copy of the model for adaptation
        adapted_model = self._create_model_copy()
        adapted_model.train()
        
        # Initialize optimizer with much lower learning rate
        optimizer = optim.AdamW(
            adapted_model.parameters(), 
            lr=self.ttt_lr, 
            weight_decay=1e-5
        )
        
        # Learning rate scheduler for gradual adaptation
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=steps,
            eta_min=self.ttt_lr * 0.1
        )
        
        adaptation_losses = []
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = adapted_model(test_x)
            
            # FIXED LOSS FUNCTIONS
            
            # 1. FIXED Reconstruction Loss - minimize variance for stability
            try:
                embeddings = adapted_model.get_embeddings(test_x)
                # Use L2 regularization instead of variance maximization
                embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
                reconstruction_loss = 0.01 * embedding_norm  # Small regularization
            except:
                reconstruction_loss = torch.tensor(0.0, device=self.device)
            
            # 2. FIXED Consistency Loss - minimize prediction variance for consistency
            probs = torch.softmax(outputs, dim=1)
            # Minimize variance of predictions (opposite of current implementation)
            pred_variance = torch.mean(torch.var(probs, dim=1))
            consistency_loss = pred_variance  # Minimize this
            
            # 3. FIXED Entropy Loss - encourage moderate confidence (not extreme)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            # Target entropy around 0.5 (moderate confidence)
            target_entropy = 0.5
            entropy_loss = torch.abs(entropy - target_entropy)
            
            # 4. NEW: Diversity Loss - encourage feature diversity without instability
            try:
                feature_diversity = torch.mean(torch.std(embeddings, dim=0))
                # Encourage moderate diversity
                target_diversity = 0.1
                diversity_loss = torch.abs(feature_diversity - target_diversity)
            except:
                diversity_loss = torch.tensor(0.0, device=self.device)
            
            # FIXED Total Loss - balanced and stable
            total_loss = (
                self.recon_weight * reconstruction_loss +
                self.consistency_weight * consistency_loss +
                self.entropy_weight * entropy_loss +
                self.diversity_weight * diversity_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=0.5)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Log progress
            if step % 2 == 0:
                logger.info(f"Fixed TTT Step {step}: "
                          f"Recon={reconstruction_loss.item():.4f}, "
                          f"Consistency={consistency_loss.item():.4f}, "
                          f"Entropy={entropy_loss.item():.4f}, "
                          f"Diversity={diversity_loss.item():.4f}, "
                          f"Total={total_loss.item():.4f}, "
                          f"LR={optimizer.param_groups[0]['lr']:.6f}")
            
            adaptation_losses.append(total_loss.item())
            
            # Early stopping if loss stabilizes
            if step > 3 and len(adaptation_losses) >= 4:
                recent_losses = adaptation_losses[-3:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    logger.info(f"Early stopping at step {step} - loss stabilized")
                    break
        
        logger.info(f"✅ Fixed TTT adaptation completed in {len(adaptation_losses)} steps")
        logger.info(f"Final loss: {adaptation_losses[-1]:.6f}")
        
        # Set model to evaluation mode
        adapted_model.eval()
        
        return adapted_model
    
    def _create_model_copy(self) -> nn.Module:
        """Create a deep copy of the model for adaptation"""
        import copy
        model_copy = copy.deepcopy(self.model)
        return model_copy.to(self.device)

def apply_ttt_fix(main_system):
    """
    Apply the TTT fix to the main system by patching the _perform_test_time_training method
    """
    
    def fixed_perform_test_time_training(self, test_x: torch.Tensor, use_federated_ttt: bool = True):
        """
        Fixed test-time training with corrected loss functions
        """
        logger.info("🔧 Using FIXED TTT adaptation with corrected loss functions...")
        
        try:
            # Create fixed TTT adapter
            ttt_adapter = FixedTTTAdapter(self.model, self.device)
            
            # Perform fixed adaptation
            adapted_model = ttt_adapter.adapt_model(test_x)
            
            logger.info("✅ Fixed TTT adaptation completed successfully")
            return adapted_model
            
        except Exception as e:
            logger.error(f"Fixed TTT adaptation failed: {str(e)}")
            return self.model  # Return original model if TTT fails
    
    # Patch the method
    main_system._perform_test_time_training = fixed_perform_test_time_training.__get__(
        main_system, type(main_system)
    )
    
    logger.info("✅ TTT fix applied to main system")

if __name__ == "__main__":
    print("TTT Fix Module Loaded")
    print("Use apply_ttt_fix(system) to apply the fix to your main system")


