#!/usr/bin/env python3
"""
New Base Model for Zero-Day Intrusion Detection
Implements Residual Temporal Convolution + Multi-Head Attention + Meta-training + Test-time adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ResidualTemporalBlock(nn.Module):
    """
    Residual Temporal Convolution Block with causal dilated 1D convolutions
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        # Crop to maintain sequence length (causal convolution)
        if out.size(2) > residual.size(2):
            out = out[:, :, :residual.size(2)].contiguous()
        
        out = out + residual
        return self.activation(out)


class TemporalEncoder(nn.Module):
    """
    Encoder with Residual Temporal Convolution + Multi-Head Attention
    """
    def __init__(self, input_dim: int, sequence_length: int = 12, 
                 hidden_dim: int = 128, latent_dim: int = 64, 
                 num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Residual Temporal Convolution layers
        self.temporal_layers = nn.ModuleList([
            ResidualTemporalBlock(hidden_dim, hidden_dim, dilation=2**i)
            for i in range(num_layers)
        ])
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Latent space projection
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim * sequence_length, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            latent: [batch_size, latent_dim]
            attention_weights: attention weights for interpretability
        """
        batch_size = x.size(0)
        
        # Transpose for conv1d: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply temporal convolutions
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Transpose back for attention: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply Multi-Head Self-Attention
        attended_x, attention_weights = self.attention(x, x, x)
        
        # Residual connection + Layer normalization
        x = self.layer_norm(x + attended_x)
        
        # Flatten for projection
        x = x.reshape(batch_size, -1)  # [batch, seq_len * hidden_dim]
        
        # Project to latent space
        latent = self.latent_projection(x)  # [batch, latent_dim]
        
        return latent, attention_weights


class TemporalDecoder(nn.Module):
    """
    Decoder for reconstruction
    """
    def __init__(self, latent_dim: int, sequence_length: int = 12, 
                 hidden_dim: int = 128, output_dim: int = 43):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Latent to sequence projection
        self.latent_to_seq = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * sequence_length // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * sequence_length // 4, hidden_dim * sequence_length)
        )
        
        # Transposed convolutions for upsampling
        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim // 4, output_dim, 3, padding=1)
        ])
        
    def forward(self, latent):
        """
        Args:
            latent: [batch_size, latent_dim]
        Returns:
            reconstruction: [batch_size, sequence_length, output_dim]
        """
        batch_size = latent.size(0)
        
        # Project latent to sequence representation
        seq_repr = self.latent_to_seq(latent)  # [batch, hidden_dim * seq_len]
        seq_repr = seq_repr.reshape(batch_size, self.hidden_dim, -1)  # [batch, hidden_dim, seq_len//4]
        
        # Apply transposed convolutions
        x = seq_repr
        for layer in self.deconv_layers:
            x = layer(x)
        
        # Ensure correct sequence length
        if x.size(2) != self.sequence_length:
            x = F.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)
        
        # Transpose to [batch, seq_len, output_dim]
        reconstruction = x.transpose(1, 2)
        
        return reconstruction


class AdapterLayers(nn.Module):
    """
    Adapter layers for selective test-time updates
    """
    def __init__(self, hidden_dim: int, adapter_dim: int = 64):
        super().__init__()
        
        self.adapter_dim = adapter_dim
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, hidden_dim]
        Returns:
            adapted_x: [batch_size, hidden_dim]
        """
        residual = x
        
        # Adapter forward pass
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        
        # Residual connection
        return x + out


class TCGANTCAModel(nn.Module):
    """
    New Base Model for Zero-Day Intrusion Detection
    Implements the complete architecture with meta-training and test-time adaptation
    """
    
    def __init__(self, input_dim: int = 43, sequence_length: int = 12,
                 hidden_dim: int = 128, latent_dim: int = 64,
                 num_heads: int = 8, num_layers: int = 3,
                 adapter_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Core components
        self.encoder = TemporalEncoder(input_dim, sequence_length, hidden_dim, latent_dim, num_heads, num_layers)
        self.decoder = TemporalDecoder(latent_dim, sequence_length, hidden_dim, input_dim)
        
        # Adapter layers for test-time adaptation
        self.adapter = AdapterLayers(hidden_dim, adapter_dim)
        
        # Classification head
        self.attack_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary classification: normal vs attack
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights PROPERLY"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias instead of zero
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias instead of zero
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for the attack classifier to break symmetry
        if hasattr(self, 'attack_classifier'):
            # Initialize the final layer with small random weights
            for layer in self.attack_classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)  # Small positive bias
    
    def forward(self, x, use_adapter=False):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, input_dim] or [batch_size, input_dim] for static features
            use_adapter: whether to use adapter layers
        Returns:
            outputs: dictionary with reconstruction, latent, classification, attention_weights
        """
        # Handle static features (no temporal modeling)
        if x.dim() == 2:  # [batch_size, input_dim]
            # For static features, we'll use a simplified approach
            # Create a simple feed-forward encoder
            if not hasattr(self, 'static_encoder'):
                self.static_encoder = nn.Sequential(
                    nn.Linear(x.size(1), self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.latent_dim)
                ).to(x.device)
            
            # Encode static features directly
            latent = self.static_encoder(x)
            attention_weights = None  # No attention for static features
            
            # For reconstruction, we'll use a simple decoder
            if not hasattr(self, 'static_decoder'):
                self.static_decoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, x.size(1))
                ).to(x.device)
            
            reconstruction = self.static_decoder(latent)
        else:
            # Original temporal sequence processing
            # Encode to latent space
            latent, attention_weights = self.encoder(x)
            
            # Apply adapter if requested
            if use_adapter:
                # For adapter, we need to apply it to the hidden representation
                # This is a simplified version - in practice, adapters would be applied at multiple layers
                pass  # TODO: Implement proper adapter application
            
            # Decode/reconstruct
            reconstruction = self.decoder(latent)
        
        # Classify
        attack_logits = self.attack_classifier(latent)
        
        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'attack_logits': attack_logits,
            'attention_weights': attention_weights
        }
    
    def compute_losses(self, outputs, targets, x_original, x_augmented=None):
        """
        Compute all loss functions
        Args:
            outputs: model outputs
            targets: classification targets
            x_original: original input sequences
            x_augmented: augmented input sequences (optional)
        Returns:
            losses: dictionary of computed losses
        """
        reconstruction = outputs['reconstruction']
        latent = outputs['latent']
        attack_logits = outputs['attack_logits']
        
        losses = {}
        
        # CRITICAL FIX: Ensure targets are properly formatted
        if isinstance(targets, torch.Tensor):
            if targets.dim() > 1:
                # Convert one-hot to class indices
                targets = torch.argmax(targets, dim=1)
            targets = targets.long()
        
        # 1. Reconstruction Loss (L1)
        losses['reconstruction'] = F.l1_loss(reconstruction, x_original)
        
        # 2. Classification Loss
        losses['classification'] = F.cross_entropy(attack_logits, targets)
        
        # 3. Contrastive Loss (if augmented data available)
        if x_augmented is not None:
            # Get latent representations for augmented data
            latent_aug, _ = self.encoder(x_augmented)
            
            # NT-Xent style contrastive loss
            losses['contrastive'] = self._compute_contrastive_loss(latent, latent_aug)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=latent.device)
        
        # 4. Consistency Loss (between original and augmented)
        if x_augmented is not None:
            outputs_aug = self.forward(x_augmented)
            losses['consistency'] = F.mse_loss(reconstruction, outputs_aug['reconstruction'])
        else:
            losses['consistency'] = torch.tensor(0.0, device=latent.device)
        
        return losses
    
    def _compute_contrastive_loss(self, z1, z2, temperature=0.1):
        """
        Compute NT-Xent contrastive loss
        Args:
            z1, z2: latent representations [batch_size, latent_dim]
            temperature: temperature parameter
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z1.device)
        
        # Symmetric loss
        loss_1_2 = F.cross_entropy(sim_matrix, labels)
        loss_2_1 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_1_2 + loss_2_1) / 2
    
    def meta_train_step(self, support_data, support_labels, query_data, query_labels, 
                       inner_lr=0.01, inner_steps=5):
        """
        Meta-training step (MAML-style)
        Args:
            support_data: support set data
            support_labels: support set labels
            query_data: query set data  
            query_labels: query set labels
            inner_lr: inner loop learning rate
            inner_steps: number of inner loop steps
        Returns:
            meta_loss: meta-training loss
        """
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Inner loop: adapt on support set
        adapted_params = self._inner_loop_update(support_data, support_labels, inner_lr, inner_steps)
        
        # Temporarily update model parameters
        for name, param in self.named_parameters():
            param.data = adapted_params[name]
        
        # Outer loop: evaluate on query set
        with torch.no_grad():
            query_outputs = self.forward(query_data)
            meta_loss = F.cross_entropy(query_outputs['attack_logits'], query_labels)
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name]
        
        return meta_loss
    
    def _inner_loop_update(self, support_data, support_labels, lr, steps):
        """
        Inner loop update for meta-training
        """
        adapted_params = {name: param.clone() for name, param in self.named_parameters()}
        
        for step in range(steps):
            # Forward pass
            outputs = self.forward(support_data)
            loss = F.cross_entropy(outputs['attack_logits'], support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            # Update parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - lr * grad
        
        return adapted_params
    
    def test_time_adapt(self, test_data, adaptation_steps=3, lr=0.001):
        """
        Test-time adaptation
        Args:
            test_data: test data for adaptation
            adaptation_steps: number of adaptation steps
            lr: learning rate for adaptation
        Returns:
            adapted_model: model after adaptation
        """
        # Create a copy of the model for adaptation
        adapted_model = TCGANTCAModel(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        adapted_model.load_state_dict(self.state_dict())
        adapted_model.train()
        
        # Optimizer for adapter parameters only
        adapter_params = list(adapted_model.adapter.parameters())
        optimizer = torch.optim.Adam(adapter_params, lr=lr)
        
        # Adaptation loop
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = adapted_model.forward(test_data, use_adapter=True)
            
            # Self-supervised loss (reconstruction)
            loss = F.l1_loss(outputs['reconstruction'], test_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            
            optimizer.step()
        
        return adapted_model
    
    def get_anomaly_score(self, x):
        """
        Compute anomaly score for zero-day detection
        Args:
            x: input data
        Returns:
            anomaly_score: anomaly score
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Reconstruction error
            recon_error = F.l1_loss(outputs['reconstruction'], x, reduction='none')
            recon_error = recon_error.mean(dim=[1, 2])  # Average over sequence and features
            
            # Latent similarity (distance from normal cluster center)
            # This is a simplified version - in practice, you'd use a learned normal cluster center
            latent_norm = torch.norm(outputs['latent'], dim=1)
            
            # Combined anomaly score
            anomaly_score = recon_error + 0.1 * latent_norm
            
            return anomaly_score