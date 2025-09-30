#!/usr/bin/env python3
"""
Simplified TCGAN/TTT Model for Zero-Day Attack Detection
Combines Simple TCGAN with Test-Time Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .simple_tcgan_model import SimpleTCGANModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTCGANTTTModel(nn.Module):
    """
    Simplified TCGAN/TTT Model for Zero-Day Attack Detection
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128, 
                 num_classes: int = 2, noise_dim: int = 64):
        super(SimpleTCGANTTTModel, self).__init__()
        
        # Core TCGAN model
        self.tcgan_model = SimpleTCGANModel(
            input_dim, latent_dim, hidden_dim, num_classes, noise_dim
        )
        
        # TTT-specific components
        self.ttt_lr = 0.001
        self.ttt_steps = 10  # Reduced for faster adaptation
        self.ttt_threshold = 0.1
        self.adaptation_threshold = 0.3
        
        # Anomaly detection threshold
        self.anomaly_threshold = 0.5  # Threshold for zero-day detection
        self.confidence_threshold = 0.7
        
        # TTT optimizer (for test-time adaptation)
        self.ttt_optimizer = None
        
    def train_transductive(self, train_x, train_y, val_x, val_y, epochs: int = 10):
        """
        Train the model using transductive learning approach
        """
        device = next(self.parameters()).device
        self.train()
        
        # Training history
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        # Create data loaders with improved batch size for better training
        batch_size = 256  # Increased batch size for better gradient estimates
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Generate unlabeled data for semi-supervised training
                batch_size_actual = batch_x.size(0)
                fake_x = self.tcgan_model.generate(batch_size_actual, device)
                
                # Use semi-supervised training method
                metrics = self.tcgan_model.train_semi_supervised(
                    batch_x, batch_y, fake_x
                )
                
                # Extract losses
                total_loss = (metrics['ae_loss'] + metrics['d_loss'] + 
                             metrics['g_loss'] + metrics['class_loss'])
                
                train_loss += total_loss
                
                # Calculate accuracy
                with torch.no_grad():
                    logits = self.tcgan_model(batch_x)
                    predictions = torch.argmax(logits, dim=1)
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    logits = self.tcgan_model(batch_x)
                    loss = nn.CrossEntropyLoss()(logits, batch_y)
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # Store history
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_acc)
            
            if epoch % 1 == 0:  # Log every epoch for faster training
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return history
    
    def _perform_test_time_training(self, test_data, test_labels):
        """
        Perform test-time training (TTT) adaptation
        """
        device = next(self.parameters()).device
        self.train()  # Enable training mode for TTT
        
        # Initialize TTT optimizer if not exists
        if self.ttt_optimizer is None:
            # Only optimize classifier and discriminator for faster TTT
            ttt_params = list(self.tcgan_model.classifier.parameters()) + \
                        list(self.tcgan_model.discriminator.parameters())
            self.ttt_optimizer = optim.Adam(ttt_params, lr=self.ttt_lr)
        
        # TTT adaptation data storage
        ttt_losses = []
        ttt_accuracies = []
        
        # Create mini-batches for TTT
        batch_size = min(64, len(test_data))
        num_batches = len(test_data) // batch_size
        
        for step in range(self.ttt_steps):
            step_loss = 0.0
            step_correct = 0
            step_total = 0
            
            # Shuffle data for each TTT step
            indices = torch.randperm(len(test_data))
            test_data_shuffled = test_data[indices]
            test_labels_shuffled = test_labels[indices]
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(test_data))
                
                batch_x = test_data_shuffled[start_idx:end_idx].to(device)
                batch_y = test_labels_shuffled[start_idx:end_idx].to(device)
                
                # TTT forward pass
                self.ttt_optimizer.zero_grad()
                
                # Get predictions
                logits = self.tcgan_model(batch_x)
                loss = nn.CrossEntropyLoss()(logits, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.tcgan_model.classifier.parameters()) + 
                    list(self.tcgan_model.discriminator.parameters()), 
                    max_norm=1.0
                )
                
                # Update parameters
                self.ttt_optimizer.step()
                
                # Calculate metrics
                step_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                step_correct += (predictions == batch_y).sum().item()
                step_total += batch_y.size(0)
            
            # Average metrics for this step
            avg_loss = step_loss / num_batches if num_batches > 0 else 0
            avg_acc = step_correct / step_total if step_total > 0 else 0
            
            ttt_losses.append(avg_loss)
            ttt_accuracies.append(avg_acc)
            
            # Early stopping if converged
            if len(ttt_losses) > 3:
                recent_losses = ttt_losses[-3:]
                if max(recent_losses) - min(recent_losses) < self.ttt_threshold:
                    logger.info(f"TTT converged early at step {step+1}")
                    break
        
        # Set back to eval mode
        self.eval()
        
        return {
            'ttt_losses': ttt_losses,
            'ttt_accuracies': ttt_accuracies,
            'final_ttt_loss': ttt_losses[-1] if ttt_losses else 0,
            'final_ttt_accuracy': ttt_accuracies[-1] if ttt_accuracies else 0,
            'ttt_steps_completed': len(ttt_losses)
        }
    
    def detect_zero_day(self, query_x):
        """
        Zero-day attack detection using TCGAN+TCAE reconstruction-based anomaly detection
        """
        device = next(self.parameters()).device
        self.eval()
        
        with torch.no_grad():
            # Get reconstruction error as anomaly score
            query_embeddings = self.tcgan_model.encode(query_x)
            reconstructed = self.tcgan_model.decode(query_embeddings)
            
            # Calculate reconstruction error (higher = more anomalous)
            reconstruction_error = torch.nn.functional.mse_loss(
                reconstructed, query_x, reduction='none'
            ).mean(dim=1)
            
            # Get discriminator output for additional anomaly signal
            disc_logits, class_logits = self.tcgan_model.discriminate(query_x)
            disc_probs = torch.sigmoid(disc_logits).squeeze()
            
            # Combined anomaly score (reconstruction error + discriminator uncertainty)
            # Higher reconstruction error + lower discriminator confidence = more anomalous
            anomaly_scores = reconstruction_error + (1.0 - disc_probs)
            
            # Convert to probabilities (sigmoid to map to [0,1])
            anomaly_probs = torch.sigmoid(anomaly_scores)
            
            # Binary classification: 1 = anomaly (zero-day attack), 0 = normal
            predictions = (anomaly_probs > self.anomaly_threshold).long()
            
            # Confidence scores (anomaly probabilities)
            confidence_scores = anomaly_probs.cpu().numpy()
            
            return predictions.cpu().numpy(), confidence_scores, anomaly_probs.cpu().numpy()
    
    def forward(self, x):
        """Forward pass for inference"""
        return self.tcgan_model(x)
    
    def get_embeddings(self, x):
        """Get embeddings from TCGAN encoder"""
        return self.tcgan_model.encode(x)
