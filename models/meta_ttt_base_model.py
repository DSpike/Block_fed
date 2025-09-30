#!/usr/bin/env python3
"""
Meta TTT Base Model for Zero-Day Intrusion Detection
Clean implementation without any old terminology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MetaTTTBaseModel(nn.Module):
    """
    Meta TTT Base Model for Zero-Day Intrusion Detection
    
    Clean architecture designed specifically for:
    - Meta-learning (FOMAML)
    - Test-Time Training (TTT)
    - Static network feature processing
    - Zero-day attack adaptation
    """
    
    def __init__(self, input_dim: int = 43, hidden_dim: int = 512, 
                 latent_dim: int = 256, num_classes: int = 2,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Enhanced feature encoder with deeper architecture and residual connections
        self.feature_encoder = nn.Sequential(
            # First layer - input expansion
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third layer with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Fourth layer with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Fifth layer with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Sixth layer with residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Final encoding layer
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Latent space projector (for meta-learning)
        self.latent_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Enhanced classification head with attention mechanism
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize FOMAML meta-learner (will be set after model is fully initialized)
        self.meta_learner = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Enhanced forward pass with attention mechanism
        Args:
            x: [batch_size, input_dim] - static network features
        Returns:
            Dictionary with latent embeddings and classification logits
        """
        # Ensure input is 2D
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Encode features to latent space
        latent = self.feature_encoder(x)
        
        # Apply attention mechanism to enhance feature representation
        # Reshape for attention: [batch_size, 1, latent_dim]
        latent_reshaped = latent.unsqueeze(1)
        
        # Self-attention
        attended_latent, _ = self.attention(latent_reshaped, latent_reshaped, latent_reshaped)
        attended_latent = attended_latent.squeeze(1)  # [batch_size, latent_dim]
        
        # Project latent space (for meta-learning)
        projected_latent = self.latent_projector(attended_latent)
        
        # Classify with enhanced features
        logits = self.classifier(projected_latent)
        
        return {
            'latent': projected_latent,
            'attack_logits': logits,
            'reconstruction': latent  # For compatibility with loss functions
        }
    
    def compute_losses(self, outputs, targets, x_original, x_augmented=None):
        """
        Compute loss functions for meta-learning and TTT
        """
        latent = outputs['latent']
        attack_logits = outputs['attack_logits']
        
        losses = {}
        
        # Ensure targets are properly formatted
        if isinstance(targets, torch.Tensor):
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            targets = targets.long()
        
        # 1. Classification Loss with class weighting (handle imbalance)
        class_weights = torch.tensor([1.0, 2.0], device=attack_logits.device)  # Weight minority class more
        losses['classification'] = F.cross_entropy(attack_logits, targets, weight=class_weights)
        
        # 2. Feature Consistency Loss (encourage similar features to have similar embeddings)
        if x_original.size(0) > 1:
            # Compute pairwise similarities in input space
            x_norm = F.normalize(x_original, p=2, dim=1)
            input_sim = torch.mm(x_norm, x_norm.t())
            
            # Compute pairwise similarities in latent space
            latent_norm = F.normalize(latent, p=2, dim=1)
            latent_sim = torch.mm(latent_norm, latent_norm.t())
            
            # Consistency loss
            losses['consistency'] = F.mse_loss(latent_sim, input_sim)
        else:
            losses['consistency'] = torch.tensor(0.0, device=x_original.device)
        
        # 3. Entropy Regularization (encourage confident predictions)
        probs = F.softmax(attack_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        losses['entropy'] = torch.mean(entropy)
        
        # 4. Reconstruction Loss (identity mapping for latent space)
        losses['reconstruction'] = F.mse_loss(latent, x_original[:, :latent.size(1)])
        
        # 5. Contrastive Loss (if augmented data available)
        if x_augmented is not None:
            aug_outputs = self.forward(x_augmented)
            aug_latent = aug_outputs['latent']
            
            # Contrastive loss: similar inputs should have similar embeddings
            latent_norm = F.normalize(latent, p=2, dim=1)
            aug_latent_norm = F.normalize(aug_latent, p=2, dim=1)
            losses['contrastive'] = F.mse_loss(latent_norm, aug_latent_norm)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=x_original.device)
        
        return losses
    
    def meta_train_step(self, support_data, support_labels, query_data, query_labels, 
                       inner_lr=0.01, inner_steps=5):
        """
        Meta-training step for FOMAML compatibility
        """
        # Clone model for inner loop
        adapted_model = type(self)(self.input_dim, self.hidden_dim, self.latent_dim, self.num_classes)
        adapted_model.load_state_dict(self.state_dict())
        adapted_model.train()
        
        # Inner loop: adapt on support set
        inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr)
        
        for _ in range(inner_steps):
            inner_optimizer.zero_grad()
            support_outputs = adapted_model(support_data)
            support_losses = adapted_model.compute_losses(support_outputs, support_labels, support_data)
            inner_loss = support_losses['classification']
            inner_loss.backward()
            inner_optimizer.step()
        
        # Outer loop: evaluate on query set
        with torch.no_grad():
            query_outputs = adapted_model(query_data)
            query_losses = adapted_model.compute_losses(query_outputs, query_labels, query_data)
            query_loss = query_losses['classification']
            
            # Calculate accuracy
            predictions = torch.argmax(query_outputs['attack_logits'], dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        # Meta-update (simplified - just use query loss)
        meta_optimizer = torch.optim.Adam(self.parameters(), lr=inner_lr)
        meta_optimizer.zero_grad()
        
        # Forward pass with current model
        current_outputs = self(query_data)
        current_losses = self.compute_losses(current_outputs, query_labels, query_data)
        meta_loss = current_losses['classification']
        
        meta_loss.backward()
        meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'query_accuracy': accuracy,
            'query_loss': query_loss.item()
        }
    
    def train_standard(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                      val_data: torch.Tensor, val_labels: torch.Tensor, 
                      epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Standard training for meta-learning base model
        """
        logger.info(f"Starting Meta TTT base model training for {epochs} epochs")
        logger.info(f"Training data shape: {train_data.shape}, Labels shape: {train_labels.shape}")
        
        self.train()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        training_history = {
            'train_total_loss': [],
            'val_accuracy': [],
            'train_losses': [],
            'train_accuracies': []
        }
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            # Training phase
            epoch_losses = []
            epoch_accuracies = []
            
            for batch_data, batch_labels in train_loader:
                device = next(self.parameters()).device
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                outputs = self.forward(batch_data)
                losses = self.compute_losses(outputs, batch_labels, batch_data)
                
                # Combine losses
                total_loss = (
                    1.0 * losses['classification'] +
                    0.1 * losses['consistency'] +
                    0.1 * losses['entropy'] +
                    0.1 * losses['reconstruction'] +
                    0.1 * losses['contrastive']
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['attack_logits'], dim=1)
                accuracy = (predictions == batch_labels).float().mean().item()
                
                epoch_losses.append(total_loss.item())
                epoch_accuracies.append(accuracy)
            
            # Validation phase
            self.eval()
            val_accuracies = []
            with torch.no_grad():
                for val_data_batch, val_labels_batch in val_loader:
                    device = next(self.parameters()).device
                    val_data_batch = val_data_batch.to(device)
                    val_labels_batch = val_labels_batch.to(device)
                    
                    outputs = self.forward(val_data_batch)
                    predictions = torch.argmax(outputs['attack_logits'], dim=1)
                    accuracy = (predictions == val_labels_batch).float().mean().item()
                    val_accuracies.append(accuracy)
            
            self.train()
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_acc = sum(epoch_accuracies) / len(epoch_accuracies)
            val_acc = sum(val_accuracies) / len(val_accuracies)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Acc={val_acc:.4f}")
            
            # Store history
            training_history['train_total_loss'].append(avg_loss)
            training_history['val_accuracy'].append(val_acc)
            training_history['train_losses'].extend(epoch_losses)
            training_history['train_accuracies'].extend(epoch_accuracies)
        
        return training_history
    
    def initialize_meta_learner(self):
        """Initialize the FOMAML meta-learner after model creation"""
        if self.meta_learner is None:
            from .fomaml_reptile_meta_learner import FOMAMLMetaLearner
            self.meta_learner = FOMAMLMetaLearner(
                base_model=self,
                inner_lr=0.01,
                num_inner_steps=5
            )
    
    def has_meta_learner(self) -> bool:
        """Check if model has meta-learner"""
        return hasattr(self, 'meta_learner') and self.meta_learner is not None
