#!/usr/bin/env python3
"""
Improved TCGAN Model with Enhanced Training for Better Discrimination
Addresses the base model training issues by implementing proper TCGAN training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTCGANModel(nn.Module):
    """
    Improved TCGAN model with enhanced training for better discrimination
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128, 
                 num_classes: int = 2, noise_dim: int = 64):
        super(ImprovedTCGANModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # Enhanced Encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Enhanced Decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Enhanced Generator with batch normalization
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
        
        # Enhanced Discriminator with batch normalization
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Enhanced Classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Enhanced optimizers with different learning rates
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        # Enhanced loss weights for better balance
        self.lambda_recon = 2.0  # Increased reconstruction weight
        self.lambda_class = 3.0  # Increased classification weight
        self.lambda_adv = 1.0    # Standard adversarial weight
        self.lambda_feature = 1.0  # Feature matching weight
        
        # Training state
        self.training_step = 0
        self.d_losses = []
        self.g_losses = []
        self.class_losses = []
        self.recon_losses = []
        
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
    
    def forward(self, x):
        """Forward pass for classification (main interface)"""
        return self.classifier(x)
    
    def train_autoencoder_enhanced(self, x):
        """Enhanced autoencoder training with better regularization"""
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Forward pass
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Enhanced latent regularization (KL divergence approximation)
        latent_reg = torch.mean(torch.sum(latent**2, dim=1))
        
        # Feature consistency loss (encourage latent features to be discriminative)
        latent_features = F.normalize(latent, p=2, dim=1)
        feature_consistency = torch.mean(torch.var(latent_features, dim=0))
        
        # Total loss with enhanced weighting
        total_loss = (self.lambda_recon * recon_loss + 
                     0.01 * latent_reg + 
                     0.1 * (1.0 / (feature_consistency + 1e-8)))
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=0.5)
        
        # Update
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        self.recon_losses.append(recon_loss.item())
        
        return recon_loss.item(), latent_reg.item()
    
    def train_discriminator_enhanced(self, real_x, fake_x, real_labels, fake_labels, attack_labels):
        """Enhanced discriminator training with feature matching"""
        self.discriminator_optimizer.zero_grad()
        
        # Get discriminator outputs
        real_disc_logits, real_class_logits = self.discriminate(real_x)
        fake_disc_logits, _ = self.discriminate(fake_x.detach())
        
        # Adversarial losses
        real_loss = self.adversarial_loss(real_disc_logits, real_labels)
        fake_loss = self.adversarial_loss(fake_disc_logits, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        
        # Enhanced classification loss with focal loss approximation
        class_probs = F.softmax(real_class_logits, dim=1)
        class_loss = self.classification_loss(real_class_logits, attack_labels)
        
        # Feature matching loss (encourage discriminator to focus on meaningful features)
        real_features = self.discriminator[0:-1](real_x)  # Features before final layer
        fake_features = self.discriminator[0:-1](fake_x.detach())
        feature_matching_loss = F.mse_loss(real_features.mean(0), fake_features.mean(0))
        
        # Total loss
        total_loss = d_loss + self.lambda_class * class_loss + self.lambda_feature * feature_matching_loss
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        
        # Update
        self.discriminator_optimizer.step()
        
        self.d_losses.append(d_loss.item())
        self.class_losses.append(class_loss.item())
        
        return d_loss.item(), class_loss.item()
    
    def train_generator_enhanced(self, fake_x, real_labels, attack_labels):
        """Enhanced generator training with better objectives"""
        self.generator_optimizer.zero_grad()
        
        # Get discriminator outputs for fake data
        fake_disc_logits, fake_class_logits = self.discriminate(fake_x)
        
        # Adversarial loss (fool discriminator)
        adv_loss = self.adversarial_loss(fake_disc_logits, real_labels)
        
        # Enhanced classification loss (encourage generator to produce classifiable samples)
        class_loss = self.classification_loss(fake_class_logits, attack_labels)
        
        # Feature matching loss (encourage generator to produce realistic features)
        fake_features = self.discriminator[0:-1](fake_x)
        real_features = self.discriminator[0:-1](real_x if hasattr(self, '_last_real_x') else fake_x)
        feature_matching_loss = F.mse_loss(fake_features.mean(0), real_features.mean(0))
        
        # Total loss
        total_loss = adv_loss + self.lambda_class * class_loss + self.lambda_feature * feature_matching_loss
        
        # Backward pass
        total_loss.backward()
        
        # Enhanced gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
        
        # Update
        self.generator_optimizer.step()
        
        self.g_losses.append(adv_loss.item())
        
        return adv_loss.item(), class_loss.item()
    
    def train_enhanced_semi_supervised(self, labeled_x, labeled_y, unlabeled_x):
        """Enhanced semi-supervised training with better balance"""
        device = next(self.parameters()).device
        batch_size = labeled_x.size(0)
        
        # Store real data for feature matching
        self._last_real_x = labeled_x
        
        # Generate separate fake samples for discriminator and generator
        fake_x_disc = self.generate(batch_size, device)
        fake_x_gen = self.generate(batch_size, device)
        
        # Create labels for GAN training
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train autoencoder on all data with enhanced training
        ae_loss, latent_reg = self.train_autoencoder_enhanced(torch.cat([labeled_x, unlabeled_x], dim=0))
        
        # Train discriminator with enhanced training
        d_loss, class_loss = self.train_discriminator_enhanced(
            labeled_x, fake_x_disc, real_labels, fake_labels, labeled_y
        )
        
        # Train generator with enhanced training
        g_loss, g_class_loss = self.train_generator_enhanced(fake_x_gen, real_labels, labeled_y)
        
        # Enhanced consistency loss for unlabeled data
        with torch.no_grad():
            unlabeled_latent = self.encode(unlabeled_x.detach())
            unlabeled_reconstructed = self.decode(unlabeled_latent.detach())
            consistency_loss = self.reconstruction_loss(unlabeled_reconstructed, unlabeled_x.detach())
        
        self.training_step += 1
        
        return {
            'ae_loss': ae_loss,
            'latent_reg': latent_reg,
            'd_loss': d_loss,
            'class_loss': class_loss,
            'g_loss': g_loss,
            'g_class_loss': g_class_loss,
            'consistency_loss': consistency_loss.item(),
            'training_step': self.training_step
        }
    
    def train_classification_only(self, x, y, epochs=10):
        """Train only the classifier for better discrimination"""
        self.classifier_optimizer.zero_grad()
        
        # Forward pass through classifier
        outputs = self.classifier(x)
        loss = self.classification_loss(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
        
        # Update
        self.classifier_optimizer.step()
        
        return loss.item()
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'avg_d_loss': np.mean(self.d_losses[-10:]) if self.d_losses else 0,
            'avg_g_loss': np.mean(self.g_losses[-10:]) if self.g_losses else 0,
            'avg_class_loss': np.mean(self.class_losses[-10:]) if self.class_losses else 0,
            'avg_recon_loss': np.mean(self.recon_losses[-10:]) if self.recon_losses else 0
        }
    
    def reset_training_stats(self):
        """Reset training statistics"""
        self.training_step = 0
        self.d_losses = []
        self.g_losses = []
        self.class_losses = []
        self.recon_losses = []





