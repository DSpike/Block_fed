#!/usr/bin/env python3
"""
Lightweight Residual Temporal Convolution (RTC) Model
Optimized for federated learning with minimal memory footprint and fast training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class LightweightRTCBlock(nn.Module):
    """
    Lightweight Residual Temporal Convolution block
    Optimized for speed and memory efficiency
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        super(LightweightRTCBlock, self).__init__()
        
        # Single lightweight convolution
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             dilation=dilation, padding=(kernel_size-1)*dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        
        # Lightweight residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """Fast forward pass with causal convolution"""
        residual = x
        
        # Single convolution
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Causal padding: remove future information
        if out.size(2) > residual.size(2):
            out = out[:, :, :residual.size(2)]
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
            if residual.size(2) > out.size(2):
                residual = residual[:, :, :out.size(2)]
        
        out = out + residual
        return out

class LightweightTemporalEncoder(nn.Module):
    """
    Lightweight temporal encoder with RTC blocks
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 10, 
                 hidden_dim: int = 64, latent_dim: int = 32):
        super(LightweightTemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Lightweight RTC blocks (only 2 blocks for speed)
        self.temporal_blocks = nn.ModuleList([
            LightweightRTCBlock(hidden_dim, hidden_dim, dilation=2**i, dropout=0.1)
            for i in range(2)  # Only 2 blocks with dilation 1, 2
        ])
        
        # Global pooling and latent projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """Encode input to latent representation"""
        batch_size = x.size(0)
        
        # Create sequence from features
        x_expanded = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Add simple position encoding
        position_encoding = torch.linspace(0, 1, self.sequence_length).unsqueeze(0).unsqueeze(-1).to(x.device)
        position_encoding = position_encoding.expand(batch_size, self.sequence_length, self.input_dim)
        x_with_pos = x_expanded + 0.1 * position_encoding
        
        # Project to hidden dimension
        x_projected = self.input_projection(x_with_pos)
        x_conv = x_projected.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        # Apply lightweight RTC blocks
        for block in self.temporal_blocks:
            x_conv = block(x_conv)
        
        # Global pooling
        x_pooled = self.global_pool(x_conv).squeeze(-1)  # (batch, hidden_dim)
        
        # Project to latent space
        latent = self.latent_projection(x_pooled)
        
        return latent

class LightweightTemporalDecoder(nn.Module):
    """
    Lightweight temporal decoder with RTC blocks
    """
    
    def __init__(self, latent_dim: int, sequence_length: int = 10, 
                 output_dim: int = 25, hidden_dim: int = 64):
        super(LightweightTemporalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Latent expansion
        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * sequence_length),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Lightweight RTC blocks (only 2 blocks for speed)
        self.temporal_blocks = nn.ModuleList([
            LightweightRTCBlock(hidden_dim, hidden_dim, dilation=2**i, dropout=0.1)
            for i in range(2)  # Only 2 blocks with dilation 1, 2
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, latent):
        """Decode latent to reconstruction"""
        # Expand latent to sequence
        x_expanded = self.latent_expansion(latent)
        x_reshaped = x_expanded.view(-1, self.sequence_length, self.hidden_dim).transpose(1, 2)
        
        # Apply lightweight RTC blocks
        for block in self.temporal_blocks:
            x_reshaped = block(x_reshaped)
        
        # Output projection
        reconstructed = self.output_projection(x_reshaped)
        
        # Global average pooling
        reconstructed = torch.mean(reconstructed, dim=2)
        
        return reconstructed

class LightweightTemporalClassifier(nn.Module):
    """
    Lightweight temporal classifier with RTC blocks
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 10, 
                 hidden_dim: int = 64, num_classes: int = 2):
        super(LightweightTemporalClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Lightweight RTC blocks (only 2 blocks for speed)
        self.temporal_blocks = nn.ModuleList([
            LightweightRTCBlock(hidden_dim, hidden_dim, dilation=2**i, dropout=0.2)
            for i in range(2)  # Only 2 blocks with dilation 1, 2
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """Forward pass for classification"""
        batch_size = x.size(0)
        
        # Create sequence from features
        x_expanded = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Add simple position encoding
        position_encoding = torch.linspace(0, 1, self.sequence_length).unsqueeze(0).unsqueeze(-1).to(x.device)
        position_encoding = position_encoding.expand(batch_size, self.sequence_length, self.input_dim)
        x_with_pos = x_expanded + 0.1 * position_encoding
        
        # Project to hidden dimension
        x_projected = self.input_projection(x_with_pos)
        x_conv = x_projected.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        # Apply lightweight RTC blocks
        for block in self.temporal_blocks:
            x_conv = block(x_conv)
        
        # Global pooling
        x_pooled = self.global_pool(x_conv).squeeze(-1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits

class LightweightRTCModel(nn.Module):
    """
    Lightweight Residual Temporal Convolution Model
    Optimized for federated learning with minimal memory footprint
    """
    
    def __init__(self, input_dim: int, sequence_length: int = 10, 
                 latent_dim: int = 32, hidden_dim: int = 64, 
                 num_classes: int = 2):
        super(LightweightRTCModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Core components (lightweight)
        self.encoder = LightweightTemporalEncoder(input_dim, sequence_length, hidden_dim, latent_dim)
        self.decoder = LightweightTemporalDecoder(latent_dim, sequence_length, input_dim, hidden_dim)
        self.classifier = LightweightTemporalClassifier(input_dim, sequence_length, hidden_dim, num_classes)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        """Forward pass through encoder-decoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def classify(self, x):
        """Forward pass for classification"""
        return self.classifier(x)
    
    def get_embeddings(self, x):
        """Extract embeddings from encoder"""
        return self.encoder(x)
    
    def train_standard(self, train_x, train_y, val_x, val_y, epochs: int = 10):
        """
        Fast standard training for lightweight model
        """
        logger.info(f"Starting Lightweight RTC training for {epochs} epochs")
        
        device = next(self.parameters()).device
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        
        training_history = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        # Create data loaders with larger batch size (lightweight model can handle it)
        batch_size = 128
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            epoch_accuracies = []
            
            # Training phase - fast supervised learning
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Classification loss
                logits = self.classify(batch_x)
                class_loss = self.classification_loss(logits, batch_y)
                
                # Reconstruction loss (lightweight)
                reconstructed, latent = self.forward(batch_x)
                recon_loss = self.reconstruction_loss(reconstructed, batch_x)
                
                # Combined loss
                total_loss = class_loss + 0.1 * recon_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == batch_y).float().mean()
                
                epoch_losses.append(total_loss.item())
                epoch_accuracies.append(accuracy.item())
            
            # Validation phase
            self.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = self.classify(batch_x)
                    val_loss = self.classification_loss(logits, batch_y)
                    val_accuracy = (torch.argmax(logits, dim=1) == batch_y).float().mean()
                    
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy.item())
            
            # Store metrics
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_train_acc = sum(epoch_accuracies) / len(epoch_accuracies)
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = sum(val_accuracies) / len(val_accuracies)
            
            training_history['epoch_losses'].append(avg_train_loss)
            training_history['epoch_accuracies'].append(avg_train_acc)
            training_history['val_losses'].append(avg_val_loss)
            training_history['val_accuracies'].append(avg_val_acc)
            
            if epoch % 1 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")
        
        logger.info("Lightweight RTC training completed")
        return training_history
    
    def detect_anomalies(self, x, threshold: float = 0.5):
        """Detect anomalies using reconstruction error"""
        self.eval()
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            recon_error = torch.mean((x - reconstructed) ** 2, dim=1)
            
            # Get classification confidence
            logits = self.classify(x)
            probs = F.softmax(logits, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            # Anomaly score combines reconstruction error and low confidence
            anomaly_score = recon_error + (1.0 - confidence)
            is_anomaly = anomaly_score > threshold
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'reconstruction_error': recon_error,
                'confidence': confidence,
                'predictions': torch.argmax(logits, dim=1),
                'probabilities': probs
            }

def main():
    """Test the lightweight RTC model"""
    logger.info("Testing Lightweight RTC Model")
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 25
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Initialize lightweight model
    model = LightweightRTCModel(input_dim=n_features, sequence_length=10, latent_dim=32, hidden_dim=64)
    
    # Test forward pass
    reconstructed, latent = model(X[:10])
    logger.info(f"Input shape: {X[:10].shape}")
    logger.info(f"Reconstructed shape: {reconstructed.shape}")
    logger.info(f"Latent shape: {latent.shape}")
    
    # Test classification
    logits = model.classify(X[:10])
    logger.info(f"Classification logits shape: {logits.shape}")
    
    # Test anomaly detection
    anomaly_results = model.detect_anomalies(X[:10])
    logger.info(f"Anomaly detection results: {anomaly_results['is_anomaly'].sum().item()} anomalies detected")
    
    logger.info("✅ Lightweight RTC model test completed!")

if __name__ == "__main__":
    main()




