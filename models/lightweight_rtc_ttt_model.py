#!/usr/bin/env python3
"""
Lightweight RTC TTT Model for Federated Learning
Wrapper around LightweightRTCModel with TTT capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional
from models.lightweight_rtc_model import LightweightRTCModel

logger = logging.getLogger(__name__)

class LightweightRTCTTTModel(nn.Module):
    """
    Lightweight RTC model with Test-Time Training (TTT) capabilities
    Optimized for federated learning with minimal memory footprint
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 10, 
                 latent_dim: int = 32, hidden_dim: int = 64, 
                 num_classes: int = 2, noise_dim: int = 32):
        super(LightweightRTCTTTModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # Core lightweight RTC model
        self.rtc_model = LightweightRTCModel(
            input_dim=input_dim,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        # TTT parameters
        self.ttt_lr = 0.001
        self.ttt_steps = 15  # Increased for better adaptation
        self.ttt_threshold = 0.1
        self.adaptation_threshold = 0.3
        self.anomaly_threshold = 0.5
        self.confidence_threshold = 0.7
        
        # TTT optimizer
        self.ttt_optimizer = None
        
    def forward(self, x):
        """Forward pass for classification"""
        return self.rtc_model.classify(x)
    
    def get_embeddings(self, x):
        """Extract embeddings from the RTC model"""
        return self.rtc_model.get_embeddings(x)
    
    def train_standard(self, train_x, train_y, val_x, val_y, epochs: int = 10):
        """
        Standard training using the lightweight RTC model
        """
        return self.rtc_model.train_standard(train_x, train_y, val_x, val_y, epochs)
    
    def adapt_to_zero_day(self, zero_day_sequences, normal_sequences, steps: int = 5):
        """
        Lightweight TTT adaptation for zero-day attacks
        """
        logger.info(f"Starting lightweight TTT adaptation for {steps} steps")
        
        device = next(self.parameters()).device
        self.train()
        
        # Setup TTT optimizer
        if self.ttt_optimizer is None:
            self.ttt_optimizer = torch.optim.AdamW(self.parameters(), lr=self.ttt_lr, weight_decay=1e-4)
        
        ttt_losses = []
        
        for step in range(steps):
            self.ttt_optimizer.zero_grad()
            
            # Combine zero-day and normal sequences
            if zero_day_sequences is not None and normal_sequences is not None:
                combined_sequences = torch.cat([zero_day_sequences, normal_sequences], dim=0)
            elif zero_day_sequences is not None:
                combined_sequences = zero_day_sequences
            else:
                combined_sequences = normal_sequences
            
            # Forward pass
            outputs = self.forward(combined_sequences)
            
            # Lightweight self-supervision losses
            # 1. Reconstruction loss (lightweight)
            try:
                reconstructed, latent = self.rtc_model(combined_sequences)
                reconstruction_loss = F.mse_loss(reconstructed, combined_sequences)
            except:
                reconstruction_loss = torch.tensor(0.0, device=device)
            
            # 2. Consistency loss (lightweight)
            probs = F.softmax(outputs, dim=1)
            consistency_loss = torch.mean(torch.var(probs, dim=1))
            
            # 3. Entropy regularization (lightweight)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            # Total lightweight loss
            total_loss = reconstruction_loss + consistency_loss + entropy_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.ttt_optimizer.step()
            
            ttt_losses.append(total_loss.item())
            
            if step % 2 == 0:
                logger.info(f"Lightweight TTT Step {step}: Loss = {total_loss.item():.4f}")
        
        logger.info(f"Lightweight TTT adaptation completed - Final loss: {ttt_losses[-1]:.4f}")
        return self
    
    def detect_zero_day_anomaly(self, x, threshold: float = 0.5):
        """
        Detect zero-day anomalies using lightweight model
        """
        return self.rtc_model.detect_anomalies(x, threshold)
    
    def predict_with_confidence(self, x):
        """
        Predict with confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            predictions = torch.argmax(logits, dim=1)
            
            return {
                'predictions': predictions,
                'probabilities': probs,
                'confidence': confidence,
                'logits': logits
            }
    
    def evaluate_zero_day_detection(self, test_x, test_y, train_x, train_y):
        """
        Evaluate zero-day detection performance
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Get predictions
            results = self.predict_with_confidence(test_x)
            predictions = results['predictions']
            confidence = results['confidence']
            
            # Calculate metrics
            accuracy = (predictions == test_y).float().mean()
            
            # Anomaly detection
            anomaly_results = self.detect_zero_day_anomaly(test_x)
            anomaly_accuracy = (anomaly_results['is_anomaly'] == (test_y == 1)).float().mean()
            
            return {
                'accuracy': accuracy.item(),
                'anomaly_accuracy': anomaly_accuracy.item(),
                'confidence': confidence.mean().item(),
                'predictions': predictions,
                'anomaly_scores': anomaly_results['anomaly_score']
            }

def main():
    """Test the lightweight RTC TTT model"""
    logger.info("Testing Lightweight RTC TTT Model")
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 25
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Initialize lightweight TTT model
    model = LightweightRTCTTTModel(
        input_dim=n_features,
        sequence_length=10,
        latent_dim=32,
        hidden_dim=64,
        num_classes=2,
        noise_dim=32
    )
    
    # Test forward pass
    outputs = model(X[:10])
    logger.info(f"Input shape: {X[:10].shape}")
    logger.info(f"Output shape: {outputs.shape}")
    
    # Test TTT adaptation
    zero_day_data = X[:5]
    normal_data = X[5:10]
    adapted_model = model.adapt_to_zero_day(zero_day_data, normal_data, steps=3)
    logger.info("TTT adaptation completed")
    
    # Test zero-day detection
    detection_results = model.evaluate_zero_day_detection(X[:20], y[:20], X[20:40], y[20:40])
    logger.info(f"Zero-day detection accuracy: {detection_results['accuracy']:.4f}")
    
    logger.info("✅ Lightweight RTC TTT model test completed!")

if __name__ == "__main__":
    main()
