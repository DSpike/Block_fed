#!/usr/bin/env python3
"""
Enhanced TCGAN Model with Improved Architecture
Fixes critical performance issues identified in analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedTCGANModel(nn.Module):
    """
    Enhanced TCGAN model with better architecture for improved performance
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 512, 
                 num_classes: int = 2, noise_dim: int = 128):
        super(EnhancedTCGANModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # Enhanced Encoder with residual connections and batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, latent_dim)
        )
        
        # Enhanced Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Enhanced Generator with better noise processing
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Enhanced Discriminator with better feature extraction
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Enhanced Classifier with focal loss support
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        # Enhanced optimizers with better learning rates and weight decay
        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=0.005, weight_decay=1e-3)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=0.005, weight_decay=1e-3)
        self.generator_optimizer = optim.AdamW(self.generator.parameters(), lr=0.003, weight_decay=1e-3)
        self.discriminator_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.002, weight_decay=1e-3)
        self.classifier_optimizer = optim.AdamW(self.classifier.parameters(), lr=0.007, weight_decay=1e-3)
        
        # Enhanced loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Focal loss for class imbalance
        self.focal_loss = self._focal_loss
        
        # Loss weights - better balanced
        self.lambda_recon = 2.0  # Increased for better reconstruction
        self.lambda_class = 3.0  # Increased for better classification
        self.lambda_adv = 1.0    # Standard adversarial loss
        self.lambda_focal = 1.0  # Focal loss weight
        
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        """Focal loss for handling class imbalance"""
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent to reconstruction"""
        return self.decoder(latent)
    
    def generate(self, batch_size, device):
        """Generate fake samples from noise"""
        noise = torch.randn(batch_size, self.noise_dim).to(device)
        return self.generator(noise)
    
    def discriminate(self, x):
        """Discriminate real vs fake"""
        disc_logits = self.discriminator(x)
        class_logits = self.classifier(x)
        return disc_logits, class_logits
    
    def train_autoencoder(self, x):
        """Enhanced autoencoder training with better regularization"""
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Forward pass
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Enhanced latent regularization
        latent_reg = torch.mean(torch.sum(latent**2, dim=1))
        latent_sparsity = torch.mean(torch.sum(torch.abs(latent), dim=1))
        
        # Total loss with better weighting
        total_loss = (self.lambda_recon * recon_loss + 
                     0.01 * latent_reg + 
                     0.001 * latent_sparsity)
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=2.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=2.0)
        
        # Update
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return recon_loss.item(), latent_reg.item()
    
    def train_discriminator(self, real_x, fake_x, real_labels, fake_labels, attack_labels):
        """Enhanced discriminator training with focal loss"""
        self.discriminator_optimizer.zero_grad()
        
        # Get discriminator outputs
        real_disc_logits, real_class_logits = self.discriminate(real_x)
        fake_disc_logits, _ = self.discriminate(fake_x)
        
        # Enhanced adversarial losses
        real_loss = self.adversarial_loss(real_disc_logits, real_labels)
        fake_loss = self.adversarial_loss(fake_disc_logits, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        
        # Enhanced classification loss with focal loss
        class_loss = self.focal_loss(real_class_logits, attack_labels)
        
        # Total loss with better weighting
        total_loss = d_loss + self.lambda_class * class_loss
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=2.0)
        
        # Update
        self.discriminator_optimizer.step()
        
        return d_loss.item(), class_loss.item()
    
    def train_generator(self, fake_x, real_labels, attack_labels):
        """Enhanced generator training"""
        self.generator_optimizer.zero_grad()
        
        # Get discriminator outputs for fake data
        fake_disc_logits, fake_class_logits = self.discriminate(fake_x)
        
        # Enhanced adversarial loss
        adv_loss = self.adversarial_loss(fake_disc_logits, real_labels)
        
        # Enhanced classification loss with focal loss
        class_loss = self.focal_loss(fake_class_logits, attack_labels)
        
        # Total loss with better weighting
        total_loss = adv_loss + self.lambda_class * class_loss
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=2.0)
        
        # Update
        self.generator_optimizer.step()
        
        return adv_loss.item(), class_loss.item()
    
    def train_semi_supervised(self, labeled_x, labeled_y, unlabeled_x):
        """Enhanced semi-supervised training with better loss balancing"""
        device = next(self.parameters()).device
        batch_size = labeled_x.size(0)
        
        # Generate separate fake samples to avoid gradient conflicts
        fake_x_disc = self.generate(batch_size, device)
        fake_x_gen = self.generate(batch_size, device)
        
        # Create labels for GAN training
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train autoencoder on all data
        ae_loss, latent_reg = self.train_autoencoder(torch.cat([labeled_x, unlabeled_x], dim=0))
        
        # Train discriminator
        d_loss, class_loss = self.train_discriminator(
            labeled_x, fake_x_disc, real_labels, fake_labels, labeled_y
        )
        
        # Train generator
        g_loss, g_class_loss = self.train_generator(fake_x_gen, real_labels, labeled_y)
        
        # Enhanced consistency loss for unlabeled data
        with torch.no_grad():
            unlabeled_latent = self.encode(unlabeled_x.detach())
            unlabeled_reconstructed = self.decode(unlabeled_latent.detach())
            consistency_loss = self.reconstruction_loss(unlabeled_reconstructed, unlabeled_x.detach())
        
        return {
            'ae_loss': ae_loss,
            'latent_reg': latent_reg,
            'd_loss': d_loss,
            'class_loss': class_loss,
            'g_loss': g_loss,
            'g_class_loss': g_class_loss,
            'consistency_loss': consistency_loss.item()
        }
    
    def forward(self, x):
        """Forward pass for inference"""
        _, class_logits = self.discriminate(x)
        return class_logits
    
    def get_embeddings(self, x):
        """Get embeddings for TTT adaptation"""
        return self.encode(x)
    
    def predict_with_confidence(self, x, threshold=0.5):
        """Predict with confidence scores"""
        with torch.no_grad():
            class_logits = self.forward(x)
            probabilities = F.softmax(class_logits, dim=1)
            predictions = (probabilities[:, 1] > threshold).long()
            confidence = torch.max(probabilities, dim=1)[0]
            
        return predictions, confidence, probabilities




