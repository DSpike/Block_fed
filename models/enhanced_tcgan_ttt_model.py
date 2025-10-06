#!/usr/bin/env python3
"""
Enhanced TCGAN/TTT Model with Improved Performance
Fixes critical issues and improves zero-day detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .enhanced_tcgan_model import EnhancedTCGANModel

logger = logging.getLogger(__name__)

class EnhancedTCGANTTTModel(nn.Module):
    """
    Enhanced TCGAN/TTT Model with improved performance
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 512, 
                 num_classes: int = 2, noise_dim: int = 128):
        super(EnhancedTCGANTTTModel, self).__init__()
        
        # Core enhanced TCGAN model
        self.tcgan_model = EnhancedTCGANModel(
            input_dim, latent_dim, hidden_dim, num_classes, noise_dim
        )
        
        # Enhanced TTT-specific components
        self.ttt_lr = 0.005  # Increased learning rate
        self.ttt_steps = 15  # More adaptation steps
        self.ttt_threshold = 0.1
        self.adaptation_threshold = 0.3
        
        # Enhanced anomaly detection
        self.anomaly_threshold = 0.5
        self.confidence_threshold = 0.7
        
        # TTT optimizer
        self.ttt_optimizer = None
        
        # Enhanced loss functions for TTT
        self.ttt_recon_loss = nn.MSELoss()
        self.ttt_consistency_loss = nn.MSELoss()
        self.ttt_entropy_loss = self._entropy_loss
        
    def _entropy_loss(self, logits):
        """Entropy regularization loss"""
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return torch.mean(entropy)
    
    def train_transductive(self, train_x, train_y, val_x, val_y, epochs: int = 10):
        """Enhanced transductive training"""
        device = next(self.parameters()).device
        self.train()
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            train_accs = []
            
            # Create batches
            batch_size = min(64, len(train_x))
            num_batches = len(train_x) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_x = train_x[start_idx:end_idx]
                batch_y = train_y[start_idx:end_idx]
                
                # Semi-supervised training with labeled data
                losses = self.tcgan_model.train_semi_supervised(
                    batch_x, batch_y, batch_x  # Use same data as unlabeled for now
                )
                
                train_losses.append(sum(losses.values()))
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = self.forward(batch_x)
                    pred_classes = torch.argmax(predictions, dim=1)
                    accuracy = (pred_classes == batch_y).float().mean().item()
                    train_accs.append(accuracy)
            
            # Validation
            with torch.no_grad():
                val_predictions = self.forward(val_x)
                val_loss = nn.CrossEntropyLoss()(val_predictions, val_y)
                val_pred_classes = torch.argmax(val_predictions, dim=1)
                val_accuracy = (val_pred_classes == val_y).float().mean().item()
            
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {avg_train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'final_train_loss': avg_train_loss,
            'final_train_acc': avg_train_acc,
            'final_val_loss': val_loss.item(),
            'final_val_acc': val_accuracy
        }
    
    def adapt_to_zero_day(self, zero_day_sequences, normal_sequences, steps: int = 15):
        """Enhanced zero-day adaptation with better loss functions"""
        device = next(self.parameters()).device
        self.train()
        
        # Initialize TTT optimizer
        if self.ttt_optimizer is None:
            self.ttt_optimizer = optim.AdamW(self.parameters(), lr=self.ttt_lr, weight_decay=1e-4)
        
        adaptation_losses = []
        
        for step in range(steps):
            self.ttt_optimizer.zero_grad()
            
            # Combine zero-day and normal sequences
            all_sequences = torch.cat([zero_day_sequences, normal_sequences], dim=0)
            
            # Forward pass
            outputs = self.forward(all_sequences)
            
            # Enhanced TTT losses
            # 1. Reconstruction loss
            try:
                embeddings = self.get_embeddings(all_sequences)
                recon_loss = torch.mean(torch.var(embeddings, dim=1))
            except:
                recon_loss = torch.tensor(0.0, device=device)
            
            # 2. Consistency loss
            probs = torch.softmax(outputs, dim=1)
            consistency_loss = torch.mean(torch.var(probs, dim=1))
            
            # 3. Entropy regularization
            entropy_loss = self.ttt_entropy_loss(outputs)
            
            # 4. Feature diversity loss
            feature_diversity = torch.mean(torch.std(embeddings, dim=0))
            diversity_loss = -feature_diversity  # Maximize diversity
            
            # Total TTT loss
            total_loss = (recon_loss + 
                         0.5 * consistency_loss + 
                         0.3 * entropy_loss + 
                         0.1 * diversity_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update
            self.ttt_optimizer.step()
            
            adaptation_losses.append(total_loss.item())
            
            if step % 3 == 0:
                logger.info(f"TTT Step {step}: Loss={total_loss.item():.4f}")
        
        return adaptation_losses
    
    def forward(self, x):
        """Forward pass"""
        return self.tcgan_model.forward(x)
    
    def get_embeddings(self, x):
        """Get embeddings for TTT adaptation"""
        return self.tcgan_model.get_embeddings(x)
    
    def predict_with_confidence(self, x, threshold=0.5):
        """Predict with confidence scores"""
        return self.tcgan_model.predict_with_confidence(x, threshold)
    
    def detect_zero_day_anomaly(self, x, threshold=None):
        """Enhanced zero-day anomaly detection"""
        if threshold is None:
            threshold = self.anomaly_threshold
        
        with torch.no_grad():
            # Get reconstruction error
            embeddings = self.get_embeddings(x)
            reconstructed = self.tcgan_model.decode(embeddings)
            recon_error = torch.mean((x - reconstructed) ** 2, dim=1)
            
            # Get prediction confidence
            _, confidence, _ = self.predict_with_confidence(x)
            
            # Combine reconstruction error and confidence
            anomaly_score = recon_error + (1 - confidence)
            
            # Binary prediction
            predictions = (anomaly_score > threshold).long()
            
        return predictions, anomaly_score, recon_error, confidence





