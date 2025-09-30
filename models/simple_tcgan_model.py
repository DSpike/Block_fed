#!/usr/bin/env python3
"""
Simplified TCGAN/TTT Model for Single Sample Processing
Handles 2D data (batch_size, features) instead of 3D temporal sequences
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

class SimpleTCGANModel(nn.Module):
    """
    Simplified TCGAN model for single sample processing (not temporal sequences)
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128, 
                 num_classes: int = 2, noise_dim: int = 64):
        super(SimpleTCGANModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # Encoder (for real data)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Generator (from noise)
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Discriminator (real vs fake)
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Binary: real vs fake
        )
        
        # Classifier (attack type classification)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Optimizers with improved learning rates for better training
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.002, weight_decay=1e-4)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.002, weight_decay=1e-4)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.002, weight_decay=1e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001, weight_decay=1e-4)  # Lower for discriminator
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=0.003, weight_decay=1e-4)  # Higher for classifier
        
        # Loss weights
        self.lambda_recon = 1.0
        self.lambda_class = 1.0
        self.lambda_adv = 1.0
    
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
        """Train autoencoder (encoder + decoder)"""
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Forward pass
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, x)
        
        # Latent regularization (encourage latent to be Gaussian)
        latent_reg = torch.mean(torch.sum(latent**2, dim=1))
        
        # Total loss
        total_loss = recon_loss + 0.01 * latent_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        
        # Update
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return recon_loss.item(), latent_reg.item()
    
    def train_discriminator(self, real_x, fake_x, real_labels, fake_labels, attack_labels):
        """Train discriminator"""
        self.discriminator_optimizer.zero_grad()
        
        # Get discriminator outputs
        real_disc_logits, real_class_logits = self.discriminate(real_x)
        fake_disc_logits, _ = self.discriminate(fake_x)
        
        # Adversarial losses
        real_loss = self.adversarial_loss(real_disc_logits, real_labels)
        fake_loss = self.adversarial_loss(fake_disc_logits, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        
        # Classification loss on real data
        class_loss = self.classification_loss(real_class_logits, attack_labels)
        
        # Total loss
        total_loss = d_loss + self.lambda_class * class_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        
        # Update
        self.discriminator_optimizer.step()
        
        return d_loss.item(), class_loss.item()
    
    def train_generator(self, fake_x, real_labels, attack_labels):
        """Train generator"""
        self.generator_optimizer.zero_grad()
        
        # Get discriminator outputs for fake data
        fake_disc_logits, fake_class_logits = self.discriminate(fake_x)
        
        # Adversarial loss (fool discriminator)
        adv_loss = self.adversarial_loss(fake_disc_logits, real_labels)
        
        # Classification loss on generated data
        class_loss = self.classification_loss(fake_class_logits, attack_labels)
        
        # Total loss
        total_loss = adv_loss + self.lambda_class * class_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        # Update
        self.generator_optimizer.step()
        
        return adv_loss.item(), class_loss.item()
    
    def train_semi_supervised(self, labeled_x, labeled_y, unlabeled_x):
        """Semi-supervised training combining all components"""
        device = next(self.parameters()).device
        batch_size = labeled_x.size(0)
        
        # Generate separate fake samples for discriminator and generator to avoid gradient conflicts
        fake_x_disc = self.generate(batch_size, device)
        fake_x_gen = self.generate(batch_size, device)
        
        # Create labels for GAN training
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train autoencoder on all data
        ae_loss, latent_reg = self.train_autoencoder(torch.cat([labeled_x, unlabeled_x], dim=0))
        
        # Train discriminator with its own fake samples
        d_loss, class_loss = self.train_discriminator(
            labeled_x, fake_x_disc, real_labels, fake_labels, labeled_y
        )
        
        # Train generator with its own fake samples
        g_loss, g_class_loss = self.train_generator(fake_x_gen, real_labels, labeled_y)
        
        # Consistency loss for unlabeled data (detach to avoid gradient conflicts)
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
        # Get classification logits
        _, class_logits = self.discriminate(x)
        return class_logits
