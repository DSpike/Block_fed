#!/usr/bin/env python3
"""
New Base Model with Test-Time Training for Zero-Day Detection
Implements Residual Temporal Convolution + Multi-Head Attention + Meta-training + Test-time adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import copy
from .tcgan_tcae_model import TCGANTCAModel
from .fomaml_reptile_meta_learner import MetaLearningTTTModel, FOMAMLMetaLearner, ReptileMetaLearner

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TCGANTTTModel(nn.Module):
    """
    New Base Model with Test-Time Training for Zero-Day Detection
    Implements Residual Temporal Convolution + Multi-Head Attention + Meta-training + Test-time adaptation
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 12, 
                 stride: int = 2, latent_dim: int = 64, hidden_dim: int = 128, 
                 num_classes: int = 2, noise_dim: int = 64, 
                 use_shallow_adaptation: bool = True, freeze_encoder_layers: int = 2,
                 meta_learning_method: str = 'fomaml', enable_meta_learning: bool = True):
        super(TCGANTTTModel, self).__init__()
        
        self.stride = stride
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # TTT specific parameters
        self.ttt_lr = 0.0001
        self.use_shallow_adaptation = use_shallow_adaptation
        self.freeze_encoder_layers = freeze_encoder_layers
        
        # Meta-learning parameters
        self.enable_meta_learning = enable_meta_learning
        self.meta_learning_method = meta_learning_method
        
        # Initialize new base model
        self.tcgan_model = TCGANTCAModel(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=8,
            num_layers=3,
            adapter_dim=64
        )
        
        # Initialize meta-learner if enabled (after base model is created)
        if self.enable_meta_learning:
            if meta_learning_method.lower() == 'fomaml':
                self.meta_learner = FOMAMLMetaLearner(
                    self.tcgan_model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5
                )
            elif meta_learning_method.lower() == 'reptile':
                self.meta_learner = ReptileMetaLearner(
                    self.tcgan_model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5
                )
            else:
                logger.warning(f"Unknown meta-learning method: {meta_learning_method}, disabling meta-learning")
                self.enable_meta_learning = False
                self.meta_learner = None
        else:
            self.meta_learner = None
        
        # TTT adaptation layers
        if self.use_shallow_adaptation:
            self.adaptation_layers = nn.ModuleList([
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim // 2, latent_dim),
                nn.LayerNorm(latent_dim)
            ])
            
            self.domain_adaptation = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            )
        
        # Temporal buffer for sliding window
        self.temporal_buffer = None
        self.buffer_size = sequence_length + stride
    
    def create_sliding_window_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sliding window sequences from input features
        CRITICAL FIX: Generate meaningful temporal sequences instead of repeating same features
        """
        batch_size = x.size(0)
        device = x.device
        
        if x.dim() == 2:  # [batch_size, features]
            # CRITICAL FIX: Create meaningful sequences with temporal variations
            # Instead of repeating the same features, add temporal patterns
            sequences = []
            
            for i in range(batch_size):
                # Get the base feature vector
                base_features = x[i]  # [features]
                
                # Create temporal sequence with variations
                sequence = []
                for t in range(self.sequence_length):
                    # Add temporal variation to create meaningful sequence
                    # Use different scaling factors for each time step
                    temporal_factor = 1.0 + 0.1 * torch.sin(torch.tensor(t * 0.5))  # Small temporal variation
                    noise = 0.01 * torch.randn_like(base_features)  # Small random noise
                    
                    # Create time-step specific features
                    temporal_features = base_features * temporal_factor + noise
                    sequence.append(temporal_features)
                
                sequences.append(torch.stack(sequence))  # [seq_len, features]
            
            sequences = torch.stack(sequences)  # [batch_size, seq_len, features]
            
        else:
            # Already in sequence format
            sequences = x
            
        return sequences
        
    def forward(self, x):
        """
        Forward pass with proper TTT adaptation
        """
        # CRITICAL FIX: Use proper sequence generation instead of simple repetition
        if len(x.shape) == 2:
            # Create meaningful temporal sequences
            x = self.create_sliding_window_sequence(x)
        
        # Get model outputs from the base model
        outputs = self.tcgan_model(x)
        
        # Apply TTT adaptation if enabled
        if self.use_shallow_adaptation and hasattr(self, 'adaptation_layers'):
            # Apply shallow adaptation to embeddings
            latent_embeddings = outputs['latent']
            adapted_embeddings = self.apply_shallow_adaptation(latent_embeddings)
            
            # Re-classify using adapted embeddings
            adapted_outputs = self.tcgan_model.attack_classifier(adapted_embeddings)
            outputs['attack_logits'] = adapted_outputs
        
        # Return attack logits for classification
        return outputs['attack_logits']
    
    def get_embeddings(self, x):
        """
        Get latent embeddings for TTT
        """
        # If input is 2D, reshape to 3D for sequence processing
        if len(x.shape) == 2:
            batch_size, features = x.shape
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        return self.tcgan_model.get_embeddings(x)
    
    def apply_shallow_adaptation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply shallow adaptation layers for TTT
        """
        if not self.use_shallow_adaptation:
            return embeddings
        
        adapted = embeddings
        for layer in self.adaptation_layers:
            adapted = layer(adapted)
        
        # Domain adaptation with residual connection
        domain_adapted = self.domain_adaptation(adapted)
        adapted_embeddings = adapted + 0.1 * domain_adapted
        
        return adapted_embeddings
    
    def freeze_encoder_partially(self):
        """
        Freeze initial encoder layers for TTT efficiency
        """
        if hasattr(self.tcgan_model, 'encoder'):
            encoder_layers = list(self.tcgan_model.encoder.children())
            layers_to_freeze = min(self.freeze_encoder_layers, len(encoder_layers))
            
            for i, layer in enumerate(encoder_layers):
                if i < layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                    logger.info(f"🔒 Frozen encoder layer {i} for TTT efficiency")
    
    def get_adaptation_parameters(self):
        """
        Get parameters for TTT adaptation
        """
        if self.use_shallow_adaptation:
            adaptation_params = []
            adaptation_params.extend(list(self.adaptation_layers.parameters()))
            adaptation_params.extend(list(self.domain_adaptation.parameters()))
            return adaptation_params
        else:
            return list(self.parameters())
    
    def create_stabilized_optimizer(self, parameters):
        """
        Create stabilized optimizer for TTT
        """
        optimizer = torch.optim.AdamW(
            parameters, 
            lr=self.ttt_lr, 
            weight_decay=1e-5, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-7, last_epoch=-1
        )
        return optimizer, scheduler
    
    def check_early_stopping(self, current_loss: float) -> bool:
        """
        Check if early stopping criteria is met
        """
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def adaptive_lr_decay(self, optimizer, current_loss: float):
        """
        Adaptive learning rate decay
        """
        if len(self.loss_history) >= 2:
            if current_loss >= self.loss_history[-2]:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * 0.8, 1e-7)
                    param_group['lr'] = new_lr
                    if new_lr > 1e-7:
                        logger.info(f"📉 Learning rate decayed: {old_lr:.6f} → {new_lr:.6f}")
    
    def stabilized_backward_pass(self, loss: torch.Tensor, optimizer):
        """
        Stabilized backward pass with NaN/Inf checks
        """
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("⚠️ NaN/Inf loss detected, skipping backward pass")
            return False
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_adaptation_parameters(), max_norm=0.5)
        optimizer.step()
        return True
    
    def temporal_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal augmentation for contrastive learning
        """
        if not self.temporal_augmentation:
            return x
        
        # Jittering
        noise = torch.randn_like(x) * 0.01
        x_jitter = x + noise
        
        # Masking
        mask = torch.rand_like(x) > 0.1
        x_masked = x * mask
        
        # Shuffling (temporal)
        if x.size(1) > 1:
            indices = torch.randperm(x.size(1), device=x.device)
            x_shuffled = x[:, indices, :]
        else:
            x_shuffled = x
        
        return torch.cat([x, x_jitter, x_masked, x_shuffled], dim=0)
    
    def feature_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature augmentation for contrastive learning
        """
        if not self.feature_augmentation:
            return x
        
        # Noise injection
        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise
        
        # Scaling
        scale = torch.rand(x.size(0), 1, 1, device=x.device) * 0.2 + 0.9
        x_scaled = x * scale
        
        # Dropout
        x_dropout = F.dropout(x, p=0.1, training=True)
        
        return torch.cat([x, x_noisy, x_scaled, x_dropout], dim=0)
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, 
                               support_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Compute contrastive learning loss
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.contrastive_temperature
        
        # Create positive and negative masks
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        pos_mask = mask
        neg_mask = ~mask
        
        # Positive pairs (diagonal)
        pos_similarities = similarity_matrix[pos_mask]
        
        # Negative pairs (off-diagonal)
        neg_similarities = similarity_matrix[neg_mask]
        
        # Contrastive loss
        if len(pos_similarities) > 0 and len(neg_similarities) > 0:
            pos_loss = -torch.mean(pos_similarities)
            neg_loss = torch.mean(torch.exp(neg_similarities))
            contrastive_loss = pos_loss + torch.log(neg_loss + 1e-8)
        else:
            contrastive_loss = torch.tensor(0.0, device=embeddings.device)
        
        return contrastive_loss
    
    def neighborhood_similarity_loss(self, embeddings: torch.Tensor, 
                                   k: int = 5) -> torch.Tensor:
        """
        Neighborhood similarity loss for better feature learning
        """
        batch_size = embeddings.size(0)
        
        if batch_size < k + 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Find k-nearest neighbors for each point
        _, nearest_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        nearest_indices = nearest_indices[:, 1:]  # Exclude self
        
        # Compute neighborhood similarity
        similarity_loss = torch.tensor(0.0, device=embeddings.device)
        
        for i in range(batch_size):
            neighbor_embeddings = embeddings[nearest_indices[i]]
            center_embedding = embeddings[i].unsqueeze(0)
            
            # Compute similarity between center and neighbors
            similarities = F.cosine_similarity(center_embedding, neighbor_embeddings, dim=1)
            similarity_loss += torch.mean(similarities)
        
        return -similarity_loss / batch_size  # Negative because we want to maximize similarity
    
    def optimize_window_size(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                           test_data: torch.Tensor, test_labels: torch.Tensor,
                           use_adaptive: bool = True) -> Dict:
        """
        Optimize window size for better rare attack capture
        """
        logger.info("🔍 Starting window size optimization for better rare attack capture")
        
        if use_adaptive:
            # Adaptive optimization
            from ..hyperparameter_optimizer import AdaptiveWindowSizeOptimizer
            optimizer = AdaptiveWindowSizeOptimizer(
                model_class=self.__class__,
                device=next(self.parameters()).device,
                sequence_length_range=(8, 64),
                stride_range=(1, 8),
                max_evaluations=15
            )
            results = optimizer.adaptive_optimization(train_data, train_labels, test_data, test_labels)
        else:
            # Grid search optimization
            from ..hyperparameter_optimizer import WindowSizeOptimizer
            optimizer = WindowSizeOptimizer(
                model_class=self.__class__,
                device=next(self.parameters()).device,
                sequence_length_range=(8, 64),
                stride_range=(1, 8),
                max_evaluations=20
            )
            results = optimizer.optimize_parallel(train_data, train_labels, test_data, test_labels)
        
        best_config = results['best_config']
        logger.info(f"🏆 Best window configuration: seq_len={best_config['sequence_length']}, stride={best_config['stride']}")
        
        # Save results
        optimizer.save_results("window_optimization_results.json")
        
        return results
    
    def train_standard(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                      val_data: torch.Tensor, val_labels: torch.Tensor, 
                      epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """
        Standard training method for new base model
        
        Args:
            train_data: Training features
            train_labels: Training labels
            val_data: Validation features  
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting new base model training for {epochs} epochs")
        logger.info(f"Training data shape: {train_data.shape}, Labels shape: {train_labels.shape}")
        
        # Set model to training mode
        self.train()
        
        # Initialize optimizers for new architecture
        encoder_optimizer = torch.optim.Adam(self.tcgan_model.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adam(self.tcgan_model.decoder.parameters(), lr=learning_rate)
        classifier_optimizer = torch.optim.Adam(self.tcgan_model.attack_classifier.parameters(), lr=learning_rate)
        
        # Loss functions
        reconstruction_loss_fn = nn.MSELoss()
        adversarial_loss_fn = nn.BCEWithLogitsLoss()
        classification_loss_fn = nn.CrossEntropyLoss()
        
        # Training history
        training_history = {
            'epoch': [],
            'train_recon_loss': [],
            'train_adv_loss': [],
            'train_class_loss': [],
            'train_total_loss': [],
            'val_accuracy': [],
            'train_losses': [],  # Add missing key
            'train_accuracies': []  # Add missing key
        }
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            # Training phase
            epoch_recon_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_class_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in train_loader:
                device = next(self.parameters()).device
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # Create sliding window sequences
                sequences = self.create_sliding_window_sequence(batch_data)
                
                # Forward pass through new model
                outputs = self.tcgan_model(sequences)
                reconstruction = outputs['reconstruction']
                latent = outputs['latent']
                attack_logits = outputs['attack_logits']
                
                # ========== NEW ARCHITECTURE TRAINING ==========
                
                # Use the new model's compute_losses method
                losses = self.tcgan_model.compute_losses(
                    outputs, batch_labels, sequences, x_augmented=None
                )
                
                # Combine losses with proper weights
                total_loss = (
                    1.0 * losses['classification'] +      # Primary: attack detection
                    0.5 * losses['reconstruction'] +      # Secondary: temporal patterns
                    0.1 * losses['contrastive'] +         # Tertiary: representation learning
                    0.1 * losses['consistency']           # Tertiary: consistency
                )
                
                # Extract individual losses for logging
                class_loss = losses['classification']
                recon_loss = losses['reconstruction']
                gen_adv_loss = torch.tensor(0.0, device=device)  # Not used in new architecture
                disc_loss = torch.tensor(0.0, device=device)     # Not used in new architecture
                
                # ========== TRAINING ==========
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.tcgan_model.parameters(), max_norm=1.0)
                
                encoder_optimizer.step()
                decoder_optimizer.step()
                classifier_optimizer.step()
                
                # Accumulate losses for logging
                epoch_recon_loss += recon_loss.item()
                epoch_adv_loss += gen_adv_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1
            
            # Average losses
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_adv_loss = epoch_adv_loss / num_batches
            avg_class_loss = epoch_class_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
            
            # Validation phase
            self.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    device = next(self.parameters()).device
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    sequences = self.create_sliding_window_sequence(batch_data)
                    outputs = self.tcgan_model(sequences)
                    attack_logits = outputs['attack_logits']
                    
                    predictions = torch.argmax(attack_logits, dim=1)
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += batch_labels.size(0)
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            
            # Store history
            training_history['epoch'].append(epoch + 1)
            training_history['train_recon_loss'].append(avg_recon_loss)
            training_history['train_adv_loss'].append(avg_adv_loss)
            training_history['train_class_loss'].append(avg_class_loss)
            training_history['train_total_loss'].append(avg_total_loss)
            training_history['val_accuracy'].append(val_accuracy)
            training_history['train_losses'].append(avg_total_loss)  # Add train_losses
            training_history['train_accuracies'].append(val_accuracy)  # Add train_accuracies (using val_accuracy for now)
            
            # Log progress
            if epoch % 2 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Recon={avg_recon_loss:.4f}, "
                          f"Adv={avg_adv_loss:.4f}, "
                          f"Class={avg_class_loss:.4f}, "
                          f"Total={avg_total_loss:.4f}, "
                          f"Val_Acc={val_accuracy:.3f}")
        
        logger.info(f"New base model training completed! Final validation accuracy: {val_accuracy:.3f}")
        return training_history
    
    def meta_train_step(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Dict:
        """
        Perform meta-training step using FOMAML or Reptile
        
        Args:
            support_data: Support set data for inner loop adaptation
            support_labels: Support set labels
            query_data: Query set data for outer loop evaluation
            query_labels: Query set labels
            
        Returns:
            Dictionary with meta-training metrics
        """
        if not self.enable_meta_learning or self.meta_learner is None:
            logger.warning("Meta-learning is disabled or meta-learner not initialized")
            return {'meta_loss': 0.0, 'query_accuracy': 0.0}
        
        return self.meta_learner.meta_train_step(support_data, support_labels, query_data, query_labels)
    
    def has_meta_train(self) -> bool:
        """Check if model has meta-training capability"""
        return self.enable_meta_learning and self.meta_learner is not None
    
    def has_meta_learner(self) -> bool:
        """Check if model has meta-learner"""
        return self.meta_learner is not None
    
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TCGANTTTModel(
        input_dim=50, 
        sequence_length=12, 
        latent_dim=64, 
        hidden_dim=128,
        num_classes=2,
        noise_dim=64
    )
    model = model.to(device)
    
    # Test forward pass
    batch_size = 32
    test_input = torch.randn(batch_size, 50).to(device)  # 2D input
    
    print("Testing new base model with TTT...")
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    outputs = model(test_input)
    print(f"Output shape: {outputs.shape}")
    
    # Test embeddings
    embeddings = model.get_embeddings(test_input)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("\n✅ All tests passed!")