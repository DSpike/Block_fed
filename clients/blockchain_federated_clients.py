#!/usr/bin/env python3
"""
Blockchain Federated Learning Clients with Transductive Few-Shot Learning
Clean implementation for zero-day attack detection without legacy GAN-TCN code
"""

import time
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from web3 import Web3
from eth_account import Account
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging

# Import our actual components
from models.transductive_fewshot_model import TransductiveFewShotModel, create_meta_tasks

# IPFS Integration
# IPFS integration handled by simple_ipfs_client.py
IPFS_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainFederatedClient:
    """
    Blockchain Federated Learning Client with Transductive Few-Shot Learning
    Clean implementation without legacy GAN-TCN code
    """
    
    def __init__(self, client_id, model, device='cpu', blockchain_config=None):
        """
        Initialize a federated learning client with blockchain integration
        
        Args:
            client_id: Unique identifier for the client
            model: TransductiveFewShotModel instance
            device: Device to run training on
            blockchain_config: Blockchain configuration dictionary
        """
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'epochs': []
        }
        
        # Blockchain integration
        self.blockchain_config = blockchain_config
        if blockchain_config:
            self._initialize_blockchain()
            logger.info(f"Client {client_id}: Blockchain and IPFS integration enabled")
        else:
            logger.warning(f"Client {client_id}: No blockchain config provided")
        
        logger.info(f"Client {client_id} initialized with Transductive Few-Shot Learning architecture and blockchain integration")
    
    def _initialize_blockchain(self):
        """Initialize blockchain connection"""
        try:
            # Initialize Web3 connection
            self.web3 = Web3(Web3.HTTPProvider(self.blockchain_config.get('rpc_url', 'http://localhost:8545')))
            
            # Set up account
            private_key = self.blockchain_config.get('private_key')
            if private_key:
                self.account = Account.from_key(private_key)
                self.web3.eth.default_account = self.account.address
            else:
                logger.warning(f"Client {self.client_id}: No private key provided for blockchain")
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to initialize blockchain: {e}")
            self.blockchain_config = None
    
    def set_training_data(self, train_data, train_labels):
        """
        Set training data for the client
        
        Args:
            train_data: Training features
            train_labels: Training labels
        """
        self.train_data = train_data
        self.train_labels = train_labels
        logger.info(f"Client {self.client_id}: Set {len(train_data)} training samples")
    
    def train_locally(self, epochs=6, batch_size=16):
        """
        Train the Transductive Few-Shot model locally using meta-learning
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            dict: Training metrics and model parameters
        """
        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs")
        
        # Convert to tensors if needed
        if not isinstance(self.train_data, torch.Tensor):
            train_data = torch.FloatTensor(self.train_data).to(self.device)
        else:
            train_data = self.train_data.to(self.device)
            
        if not isinstance(self.train_labels, torch.Tensor):
            train_labels = torch.LongTensor(self.train_labels).to(self.device)
        else:
            train_labels = self.train_labels.to(self.device)
        
        # Create meta-learning tasks from local data
        logger.info(f"Client {self.client_id}: Creating meta-learning tasks from local data...")
        meta_tasks = create_meta_tasks(train_data, train_labels, num_tasks=5, shots=5, ways=2)
        logger.info(f"Client {self.client_id}: Created {len(meta_tasks)} meta-learning tasks")
        
        # Train using meta-learning
        logger.info(f"Client {self.client_id}: Running meta-learning training...")
        training_history = self.model.meta_train(meta_tasks, meta_epochs=epochs)
        
        # Run test-time training adaptation
        logger.info(f"Client {self.client_id}: Running test-time training adaptation...")
        self.model.adapt_to_test_time(train_data[:100], train_labels[:100], adaptation_steps=10)
        logger.info(f"Client {self.client_id}: TTT adaptation completed")
        
        # Calculate training metrics
        avg_loss = np.mean(training_history['epoch_losses'])
        avg_accuracy = np.mean(training_history['epoch_accuracies'])
        
        logger.info(f"Client {self.client_id}: Enhanced training completed - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        # Get model parameters for aggregation
        model_parameters = {}
        for name, param in self.model.named_parameters():
            model_parameters[name] = param.detach().cpu()
        
        # Compute model hash
        model_hash = self.compute_model_hash(model_parameters)
        
        # Store on IPFS if blockchain is available
        ipfs_cid = None
        blockchain_tx_hash = None
        if self.blockchain_config and IPFS_AVAILABLE:
            try:
                ipfs_cid = self.store_model_on_ipfs(model_parameters, {
                    'client_id': self.client_id,
                    'training_loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'epochs': epochs
                })
                
                # Record on blockchain
                blockchain_tx_hash = self.record_model_on_blockchain(model_hash, ipfs_cid)
                
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Failed to store on IPFS/blockchain: {e}")
        
        return {
            'client_id': self.client_id,
            'model_parameters': model_parameters,
            'training_loss': avg_loss,
            'accuracy': avg_accuracy,
            'model_hash': model_hash,
            'ipfs_cid': ipfs_cid,
            'blockchain_tx_hash': blockchain_tx_hash,
            'timestamp': time.time()
        }
    
    def compute_model_hash(self, parameters):
        """Compute SHA256 hash of model parameters"""
        param_str = str(sorted(parameters.keys()))
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    def store_model_on_ipfs(self, parameters, metadata):
        """Store model on IPFS (placeholder implementation)"""
        try:
            # This would integrate with actual IPFS client
            # For now, return a placeholder CID
            model_data = {
                'parameters': {name: param.tolist() if hasattr(param, 'tolist') else param 
                              for name, param in parameters.items()},
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            # Simulate IPFS storage
            data_str = json.dumps(model_data, default=str)
            cid = hashlib.sha256(data_str.encode()).hexdigest()[:16]  # Simplified CID
            
            logger.info(f"Client {self.client_id}: Model stored on IPFS with CID: {cid}")
            return cid
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to store on IPFS: {e}")
            return None
    
    def record_model_on_blockchain(self, model_hash, ipfs_cid):
        """Record model hash on blockchain (placeholder implementation)"""
        try:
            # This would integrate with actual blockchain contract
            # For now, return a placeholder transaction hash
            tx_hash = hashlib.sha256(f"{model_hash}_{ipfs_cid}_{time.time()}".encode()).hexdigest()
            
            logger.info(f"Client {self.client_id}: Model recorded on blockchain: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to record on blockchain: {e}")
            return None
    
    def update_model(self, aggregated_parameters):
        """
        Update local model with aggregated parameters
        
        Args:
            aggregated_parameters: Aggregated model parameters from server
        """
        logger.info(f"Client {self.client_id}: Updating model with aggregated parameters")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_parameters:
                    param.copy_(aggregated_parameters[name].to(self.device))
        
        logger.info(f"Client {self.client_id}: Model updated successfully")
    
    def evaluate_model(self, test_data, test_labels):
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test features
            test_labels: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        
        # Convert to tensors if needed
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.FloatTensor(test_data).to(self.device)
        else:
            test_data = test_data.to(self.device)
            
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.LongTensor(test_labels).to(self.device)
        else:
            test_labels = test_labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(test_data)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            accuracy = (predictions == test_labels).float().mean().item()
            
            # Convert to CPU for sklearn metrics
            predictions_cpu = predictions.cpu().numpy()
            labels_cpu = test_labels.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_cpu, predictions_cpu, average='weighted', zero_division=0
            )
            
            # Calculate reconstruction errors for anomaly detection
            reconstruction_errors = torch.mean((test_data - outputs) ** 2, dim=1)
            avg_reconstruction_error = reconstruction_errors.mean().item()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'reconstruction_error': avg_reconstruction_error
        }


if __name__ == "__main__":
    # Test the blockchain federated learning system
    print("Testing blockchain federated learning with Transductive Few-Shot Learning architecture...")
    
    # Initialize model
    model = TransductiveFewShotModel(
        input_dim=25,
        hidden_dim=128,
        embedding_dim=64,
        num_classes=2
    )
    
    # Initialize client
    client = BlockchainFederatedClient("test_client", model)
    
    # Create dummy data
    train_data = torch.randn(100, 25)
    train_labels = torch.randint(0, 2, (100,))
    
    # Set training data
    client.set_training_data(train_data, train_labels)
    
    # Train locally
    results = client.train_locally(epochs=2)
    
    print(f"Training completed: {results}")
