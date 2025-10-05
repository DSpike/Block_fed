#!/usr/bin/env python3
"""
Patch to enhance the blockchain coordinator with improved TCGAN training
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
from enhanced_tcgan_training import EnhancedTCGANTrainer
from models.improved_tcgan_model import ImprovedTCGANModel

logger = logging.getLogger(__name__)

def patch_enhanced_tcgan_training(coordinator):
    """
    Patch the coordinator to use enhanced TCGAN training
    
    Args:
        coordinator: The blockchain coordinator instance
    """
    
    def enhanced_fallback_training(self, epochs: int, batch_size: int, learning_rate: float):
        """Enhanced fallback training using proper TCGAN training"""
        logger.info(f"Client {self.client_id}: Using enhanced TCGAN training")
        
        # Convert model to ImprovedTCGANModel if needed
        if not isinstance(self.model, ImprovedTCGANModel):
            logger.info(f"Converting model to ImprovedTCGANModel for better training")
            
            # Get current model parameters
            current_state = self.model.state_dict()
            
            # Create new improved model
            input_dim = self.train_data.shape[1]
            improved_model = ImprovedTCGANModel(
                input_dim=input_dim,
                latent_dim=64,
                hidden_dim=128,
                num_classes=2,
                noise_dim=64
            )
            
            # Copy compatible parameters
            improved_model = improved_model.to(self.device)
            
            # Try to copy classifier parameters if they exist
            if 'classifier.0.weight' in current_state:
                improved_model.classifier.load_state_dict({
                    name: param for name, param in current_state.items() 
                    if name.startswith('classifier.')
                }, strict=False)
            
            # Replace the model
            self.model = improved_model
        
        # Create enhanced trainer
        trainer = EnhancedTCGANTrainer(self.model, self.device)
        
        # Enhanced training
        training_history = trainer.train_enhanced_tcgan(
            self.train_data, 
            self.train_labels, 
            epochs=epochs
        )
        
        # Evaluate model
        evaluation_metrics = trainer.evaluate_model(self.train_data, self.train_labels)
        
        # Get model parameters
        model_parameters = {}
        for name, param in self.model.named_parameters():
            model_parameters[name] = param.detach().cpu()
        
        # Compute model hash
        model_hash = self.compute_model_hash(model_parameters)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Get performance analysis
        performance_analysis = trainer.get_model_performance_analysis()
        
        logger.info(f"Enhanced training completed - Accuracy: {evaluation_metrics['accuracy']:.4f}, "
                   f"F1: {evaluation_metrics['f1_score']:.4f}, MCC: {evaluation_metrics['mcc']:.4f}")
        
        # Return enhanced client update
        from coordinators.blockchain_fedavg_coordinator import ClientUpdate
        
        return ClientUpdate(
            client_id=self.client_id,
            model_parameters=model_parameters,
            sample_count=len(self.train_data),
            training_loss=training_history['epoch_losses'][-1] if training_history['epoch_losses'] else 0.0,
            validation_accuracy=evaluation_metrics['accuracy'],
            validation_precision=evaluation_metrics['precision'],
            validation_recall=evaluation_metrics['recall'],
            validation_f1_score=evaluation_metrics['f1_score'],
            model_hash=model_hash,
            timestamp=time.time()
        )
    
    # Patch the method
    coordinator._fallback_basic_training = enhanced_fallback_training.__get__(coordinator, coordinator.__class__)
    
    logger.info("Enhanced TCGAN training patch applied successfully")
    
    return coordinator

def apply_enhanced_training_patch():
    """
    Apply the enhanced training patch to the system
    """
    try:
        # Import the coordinator
        from coordinators.blockchain_fedavg_coordinator import BlockchainFedAVGCoordinator
        
        # Patch the class method
        BlockchainFedAVGCoordinator._fallback_basic_training = enhanced_fallback_training
        
        logger.info("Enhanced training patch applied to BlockchainFedAVGCoordinator")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply enhanced training patch: {str(e)}")
        return False

# Enhanced fallback training function
def enhanced_fallback_training(self, epochs: int, batch_size: int, learning_rate: float):
    """Enhanced fallback training using proper TCGAN training"""
    import time
    
    logger.info(f"Client {self.client_id}: Using enhanced TCGAN training")
    
    # Convert model to ImprovedTCGANModel if needed
    if not isinstance(self.model, ImprovedTCGANModel):
        logger.info(f"Converting model to ImprovedTCGANModel for better training")
        
        # Get current model parameters
        current_state = self.model.state_dict()
        
        # Create new improved model
        input_dim = self.train_data.shape[1]
        improved_model = ImprovedTCGANModel(
            input_dim=input_dim,
            latent_dim=64,
            hidden_dim=128,
            num_classes=2,
            noise_dim=64
        )
        
        # Copy compatible parameters
        improved_model = improved_model.to(self.device)
        
        # Try to copy classifier parameters if they exist
        if 'classifier.0.weight' in current_state:
            try:
                improved_model.classifier.load_state_dict({
                    name: param for name, param in current_state.items() 
                    if name.startswith('classifier.')
                }, strict=False)
                logger.info("Successfully copied classifier parameters")
            except Exception as e:
                logger.warning(f"Could not copy classifier parameters: {str(e)}")
        
        # Replace the model
        self.model = improved_model
    
    # Create enhanced trainer
    trainer = EnhancedTCGANTrainer(self.model, self.device)
    
    # Enhanced training
    training_history = trainer.train_enhanced_tcgan(
        self.train_data, 
        self.train_labels, 
        epochs=epochs
    )
    
    # Evaluate model
    evaluation_metrics = trainer.evaluate_model(self.train_data, self.train_labels)
    
    # Get model parameters
    model_parameters = {}
    for name, param in self.model.named_parameters():
        model_parameters[name] = param.detach().cpu()
    
    # Compute model hash
    model_hash = self.compute_model_hash(model_parameters)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    logger.info(f"Enhanced training completed - Accuracy: {evaluation_metrics['accuracy']:.4f}, "
               f"F1: {evaluation_metrics['f1_score']:.4f}, MCC: {evaluation_metrics['mcc']:.4f}")
    
    # Return enhanced client update
    from coordinators.blockchain_fedavg_coordinator import ClientUpdate
    
    return ClientUpdate(
        client_id=self.client_id,
        model_parameters=model_parameters,
        sample_count=len(self.train_data),
        training_loss=training_history['epoch_losses'][-1] if training_history['epoch_losses'] else 0.0,
        validation_accuracy=evaluation_metrics['accuracy'],
        validation_precision=evaluation_metrics['precision'],
        validation_recall=evaluation_metrics['recall'],
        validation_f1_score=evaluation_metrics['f1_score'],
        model_hash=model_hash,
        timestamp=time.time()
    )




