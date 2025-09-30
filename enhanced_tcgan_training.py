#!/usr/bin/env python3
"""
Enhanced TCGAN Training Coordinator
Implements proper TCGAN training for better discrimination
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from models.improved_tcgan_model import ImprovedTCGANModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTCGANTrainer:
    """
    Enhanced TCGAN trainer that implements proper training for better discrimination
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize enhanced TCGAN trainer
        
        Args:
            model: TCGAN model to train
            device: Device to run on
        """
        self.device = device
        self.model = model.to(device)
        
        # Convert to improved model if needed
        if not isinstance(model, ImprovedTCGANModel):
            logger.warning("Model is not ImprovedTCGANModel, training may not be optimal")
        
        # Training configuration
        self.training_config = {
            'semi_supervised_ratio': 0.7,  # 70% semi-supervised, 30% classification-only
            'classification_epochs': 5,    # Additional classification epochs
            'semi_supervised_epochs': 10,  # Semi-supervised training epochs
            'batch_size': 64,
            'learning_rate': 0.001
        }
        
        logger.info(f"Enhanced TCGAN trainer initialized on {device}")
    
    def train_enhanced_tcgan(self, train_data: torch.Tensor, train_labels: torch.Tensor, 
                           epochs: int = 15) -> Dict[str, List[float]]:
        """
        Enhanced TCGAN training with proper discrimination learning
        
        Args:
            train_data: Training features
            train_labels: Training labels
            epochs: Number of training epochs
            
        Returns:
            training_history: Dictionary with training metrics
        """
        logger.info(f"Starting enhanced TCGAN training for {epochs} epochs")
        
        # Move data to device
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        
        # Training history
        training_history = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'd_losses': [],
            'g_losses': [],
            'class_losses': [],
            'recon_losses': []
        }
        
        # Calculate class distribution
        class_counts = torch.bincount(train_labels)
        total_samples = len(train_labels)
        logger.info(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts.tolist()))}")
        
        # Split data for semi-supervised training
        labeled_size = int(len(train_data) * 0.8)  # 80% labeled, 20% unlabeled
        labeled_indices = torch.randperm(len(train_data))[:labeled_size]
        unlabeled_indices = torch.randperm(len(train_data))[labeled_size:]
        
        labeled_data = train_data[labeled_indices]
        labeled_labels = train_labels[labeled_indices]
        unlabeled_data = train_data[unlabeled_indices]
        
        logger.info(f"Training with {len(labeled_data)} labeled and {len(unlabeled_data)} unlabeled samples")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            batch_count = 0
            
            # Shuffle data
            perm = torch.randperm(len(labeled_data))
            labeled_data_shuffled = labeled_data[perm]
            labeled_labels_shuffled = labeled_labels[perm]
            
            # Mini-batch training
            batch_size = min(self.training_config['batch_size'], len(labeled_data))
            
            for i in range(0, len(labeled_data), batch_size):
                # Get labeled batch
                labeled_batch = labeled_data_shuffled[i:i+batch_size]
                labeled_batch_labels = labeled_labels_shuffled[i:i+batch_size]
                
                # Get corresponding unlabeled batch
                unlabeled_start = i % len(unlabeled_data)
                unlabeled_end = min(unlabeled_start + batch_size, len(unlabeled_data))
                unlabeled_batch = unlabeled_data[unlabeled_start:unlabeled_end]
                
                # Pad unlabeled batch if needed
                if len(unlabeled_batch) < batch_size:
                    remaining = batch_size - len(unlabeled_batch)
                    pad_indices = torch.randint(0, len(unlabeled_data), (remaining,))
                    unlabeled_batch = torch.cat([unlabeled_batch, unlabeled_data[pad_indices]])
                
                # Training strategy based on epoch
                if epoch < self.training_config['semi_supervised_epochs']:
                    # Semi-supervised training
                    loss_dict = self.model.train_enhanced_semi_supervised(
                        labeled_batch, labeled_batch_labels, unlabeled_batch
                    )
                    
                    # Calculate combined loss
                    combined_loss = (loss_dict['ae_loss'] + 
                                   loss_dict['d_loss'] + 
                                   loss_dict['g_loss'] + 
                                   loss_dict['class_loss'])
                    
                    epoch_loss += combined_loss
                    
                    # Store individual losses
                    training_history['d_losses'].append(loss_dict['d_loss'])
                    training_history['g_losses'].append(loss_dict['g_loss'])
                    training_history['class_losses'].append(loss_dict['class_loss'])
                    training_history['recon_losses'].append(loss_dict['ae_loss'])
                    
                else:
                    # Classification-only training for fine-tuning
                    loss = self.model.train_classification_only(labeled_batch, labeled_batch_labels)
                    epoch_loss += loss
                
                batch_count += 1
                
                # Calculate accuracy
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(labeled_batch)
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == labeled_batch_labels).float().mean()
                    epoch_accuracy += accuracy.item()
                    self.model.train()
                
                # Memory management
                if batch_count % 100 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate epoch averages
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_accuracy = epoch_accuracy / batch_count
            
            training_history['epoch_losses'].append(avg_epoch_loss)
            training_history['epoch_accuracies'].append(avg_epoch_accuracy)
            
            # Log progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")
                
                # Log training stats if available
                if hasattr(self.model, 'get_training_stats'):
                    stats = self.model.get_training_stats()
                    logger.info(f"  D Loss: {stats['avg_d_loss']:.4f}, G Loss: {stats['avg_g_loss']:.4f}, "
                              f"Class Loss: {stats['avg_class_loss']:.4f}, Recon Loss: {stats['avg_recon_loss']:.4f}")
        
        logger.info(f"Enhanced TCGAN training completed. Final accuracy: {avg_epoch_accuracy:.4f}")
        return training_history
    
    def evaluate_model(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            test_data: Test features
            test_labels: Test labels
            
        Returns:
            evaluation_metrics: Dictionary with evaluation metrics
        """
        self.model.eval()
        
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            outputs = self.model(test_data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Calculate metrics
            accuracy = (predictions == test_labels).float().mean().item()
            
            # Calculate per-class metrics
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            
            test_labels_np = test_labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels_np, predictions_np, average='binary'
            )
            
            # Confusion matrix
            cm = confusion_matrix(test_labels_np, predictions_np)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Matthews Correlation Coefficient
            from sklearn.metrics import matthews_corrcoef
            try:
                mcc = matthews_corrcoef(test_labels_np, predictions_np)
            except:
                mcc = 0.0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
            }
            
            logger.info(f"Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
            logger.info(f"Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
            
            return metrics
    
    def get_model_performance_analysis(self) -> Dict[str, any]:
        """
        Get detailed performance analysis of the model
        
        Returns:
            analysis: Dictionary with performance analysis
        """
        if hasattr(self.model, 'get_training_stats'):
            stats = self.model.get_training_stats()
            
            analysis = {
                'training_completed': True,
                'training_step': stats['training_step'],
                'discriminator_performance': {
                    'avg_loss': stats['avg_d_loss'],
                    'trend': 'improving' if len(self.model.d_losses) > 10 and 
                            stats['avg_d_loss'] < np.mean(self.model.d_losses[:10]) else 'stable'
                },
                'generator_performance': {
                    'avg_loss': stats['avg_g_loss'],
                    'trend': 'improving' if len(self.model.g_losses) > 10 and 
                            stats['avg_g_loss'] < np.mean(self.model.g_losses[:10]) else 'stable'
                },
                'classification_performance': {
                    'avg_loss': stats['avg_class_loss'],
                    'trend': 'improving' if len(self.model.class_losses) > 10 and 
                            stats['avg_class_loss'] < np.mean(self.model.class_losses[:10]) else 'stable'
                },
                'reconstruction_performance': {
                    'avg_loss': stats['avg_recon_loss'],
                    'trend': 'improving' if len(self.model.recon_losses) > 10 and 
                            stats['avg_recon_loss'] < np.mean(self.model.recon_losses[:10]) else 'stable'
                }
            }
            
            return analysis
        
        return {'training_completed': False, 'message': 'No training statistics available'}



