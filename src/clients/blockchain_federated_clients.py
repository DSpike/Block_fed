#!/usr/bin/env python3
"""
Blockchain Federated Learning Clients with Real GAN-TCN Architecture
Using the complete architecture and preprocessing methodology for zero-day attack detection
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

# Import our extracted architecture and preprocessing
from blockchain_gan_tcn_architecture import create_gan_tcn_model
from blockchain_preprocessing_methodology import BlockchainPreprocessor

# IPFS Integration
# IPFS integration handled by simple_ipfs_client.py
IPFS_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainFederatedClient:
    """
    Blockchain Federated Learning Client with Real GAN-TCN Architecture
    """
    
    def __init__(self, client_id, data_dir="data/CIC-DDoS2019", device='cpu', blockchain_config=None):
        """
        Initialize a federated learning client with blockchain integration
        
        Args:
            client_id: Unique identifier for the client
            data_dir: Path to CIC-DDoS2019 dataset
            device: Device to run training on
            blockchain_config: Blockchain configuration dictionary
        """
        self.client_id = client_id
        self.device = device
        self.data_dir = data_dir
        
        # Initialize preprocessor
        self.preprocessor = BlockchainPreprocessor(data_dir=data_dir)
        
        # Initialize GAN-TCN model
        self.model = create_gan_tcn_model().to(device)
        
        # Initialize optimizers with better learning rates for convergence
        self.optimizer_g = optim.AdamW(self.model.generator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
        self.optimizer_d = optim.AdamW(self.model.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)
        
        # Training history
        self.training_history = []
        
        # Blockchain integration
        self.blockchain_config = blockchain_config
        self.web3 = None
        self.contract = None
        self.account = None
        
        # IPFS integration with simple client
        try:
            from simple_ipfs_client import SimpleIPFSClient
            self.ipfs_client = SimpleIPFSClient()
            if self.ipfs_client.test_connection():
                self.ipfs_available = True
                logger.info(f"Client {client_id}: IPFS enabled and connected")
            else:
                self.ipfs_available = False
                logger.warning(f"Client {client_id}: IPFS connection failed, running without IPFS")
        except Exception as e:
            self.ipfs_client = None
            self.ipfs_available = False
            logger.warning(f"Client {client_id}: IPFS not available: {str(e)}, running without IPFS")
        
        if blockchain_config:
            logger.info(f"Client {client_id}: Initializing blockchain with config: {blockchain_config.get('address', 'No address')}")
            success = self._initialize_blockchain()
            if not success:
                logger.error(f"Client {client_id}: Blockchain initialization failed")
        else:
            logger.warning(f"Client {client_id}: No blockchain config provided")
        
        logger.info(f"Client {client_id} initialized with GAN-TCN architecture and blockchain integration")
    
    def _initialize_blockchain(self):
        """
        Initialize blockchain connection and smart contract
        """
        try:
            # Connect to Ethereum network
            self.web3 = Web3(Web3.HTTPProvider(self.blockchain_config['rpc_url']))
            
            # Add middleware for PoA networks (like Hardhat)
            try:
                from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
                self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                logger.warning(f"Client {self.client_id}: ExtraDataToPOAMiddleware not available, skipping")
            
            # Check connection
            if not self.web3.is_connected():
                logger.error(f"Client {self.client_id}: Failed to connect to blockchain")
                return False
            
            # Load account
            self.account = Account.from_key(self.blockchain_config['private_key'])
            self.web3.eth.default_account = self.account.address
            
            # Load smart contract
            contract_address = self.blockchain_config['contract_address']
            contract_abi = self.blockchain_config['contract_abi']
            self.contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
            
            logger.info(f"Client {self.client_id}: Blockchain initialized successfully")
            logger.info(f"  Account: {self.account.address}")
            logger.info(f"  Contract: {contract_address}")
            return True
        except Exception as e:
            logger.error(f"Client {self.client_id}: Blockchain initialization failed: {str(e)}")
            logger.error(f"Client {self.client_id}: Config keys: {list(self.blockchain_config.keys()) if self.blockchain_config else 'None'}")
            logger.error(f"Client {self.client_id}: Contract address: {self.blockchain_config.get('contract_address', 'None') if self.blockchain_config else 'None'}")
            return False
    
    def _initialize_ipfs(self):
        """
        Initialize IPFS client connection using simple client
        """
        try:
            from simple_ipfs_client import SimpleIPFSClient
            self.ipfs_client = SimpleIPFSClient()
            if self.ipfs_client.test_connection():
                logger.info(f"Client {self.client_id}: IPFS initialized successfully")
                return True
            else:
                logger.warning(f"Client {self.client_id}: Could not connect to IPFS")
                self.ipfs_client = None
                self.ipfs_available = False
                return False
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Could not connect to IPFS: {str(e)}")
            self.ipfs_client = None
            self.ipfs_available = False
            return False
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Blockchain initialization failed: {str(e)}")
            return False
    
    def generate_model_hash(self, parameters):
        """
        Generate SHA256 hash of model parameters
        
        Args:
            parameters: Model parameters dictionary
            
        Returns:
            model_hash: SHA256 hash as bytes32
        """
        # Convert parameters to JSON string
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(param_str.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert to bytes32
        model_hash = self.web3.to_bytes(hexstr=hash_hex)
        
        return model_hash
    
    def upload_model_to_ipfs(self, parameters, metadata=None):
        """
        Upload model parameters to IPFS
        
        Args:
            parameters: Model parameters dictionary
            metadata: Additional metadata (client_id, round, etc.)
            
        Returns:
            ipfs_hash: IPFS hash of uploaded model
        """
        if not self.ipfs_client:
            logger.warning(f"Client {self.client_id}: IPFS not available")
            return None
        
        try:
            import pickle
            
            # Move model parameters to CPU before serialization to avoid GPU memory issues
            if isinstance(parameters, torch.Tensor):
                parameters_cpu = parameters.cpu()
            elif isinstance(parameters, dict):
                parameters_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in parameters.items()}
            else:
                parameters_cpu = parameters
            
            # Prepare model data (on CPU)
            model_data = {
                'parameters': parameters_cpu,
                'metadata': metadata or {}
            }
            
            # Clear GPU cache before serialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Serialize model data (now on CPU)
            serialized_data = pickle.dumps(model_data)
            
            # Upload to IPFS using simple client
            ipfs_hash = self.ipfs_client.add_data(model_data)
            
            logger.info(f"Client {self.client_id}: Model uploaded to IPFS: {ipfs_hash}")
            logger.info(f"  Size: {len(serialized_data)} bytes")
            
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to upload model to IPFS: {str(e)}")
            return None
    
    def download_model_from_ipfs(self, ipfs_hash):
        """
        Download model parameters from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the model
            
        Returns:
            model_data: Dictionary with parameters and metadata
        """
        if not self.ipfs_client:
            logger.warning(f"Client {self.client_id}: IPFS not available")
            return None
        
        try:
            # Download from IPFS using simple client
            model_data = self.ipfs_client.get_data(ipfs_hash)
            
            # Check if data was retrieved successfully
            if model_data is None:
                logger.error(f"Client {self.client_id}: Failed to retrieve data from IPFS")
                return None
            
            logger.info(f"Client {self.client_id}: Model downloaded from IPFS: {ipfs_hash}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to download model from IPFS: {str(e)}")
            return None
    
    def submit_model_to_blockchain(self, round_num):
        """
        Submit model parameters to blockchain with optional IPFS storage
        
        Args:
            round_num: Current federated learning round
            
        Returns:
            result: Dictionary with submission results
        """
        result = {
            'success': False,
            'ipfs_hash': None,
            'blockchain_hash': None,
            'blockchain_tx': None
        }
        
        if not self.contract:
            logger.warning(f"Client {self.client_id}: No blockchain contract available")
            return result
        
        try:
            # Get model parameters
            parameters = self.get_model_parameters()
            
            # Step 1: Upload to IPFS if available
            if self.ipfs_client:
                metadata = {
                    'client_id': self.client_id,
                    'round': round_num,
                    'timestamp': int(time.time()),
                    'model_type': 'gan_tcn',
                    'parameters_count': len(parameters)
                }
                
                ipfs_hash = self.upload_model_to_ipfs(parameters, metadata)
                if ipfs_hash:
                    result['ipfs_hash'] = ipfs_hash
                    logger.info(f"Client {self.client_id}: Model uploaded to IPFS: {ipfs_hash}")
                else:
                    logger.warning(f"Client {self.client_id}: IPFS upload failed, continuing with blockchain-only")
            else:
                logger.info(f"Client {self.client_id}: IPFS not available, using blockchain-only approach")
            
            # Step 2: Submit to blockchain
            if result['ipfs_hash']:
                # Convert IPFS hash (base58) to proper hex format for blockchain
                try:
                    import base58
                    # Decode base58 IPFS hash to bytes
                    ipfs_hash_bytes = base58.b58decode(result['ipfs_hash'])
                    # Convert to hex and pad to 32 bytes (64 hex characters)
                    ipfs_hash_hex = ipfs_hash_bytes.hex().ljust(64, '0')[:64]
                    model_hash = self.web3.to_bytes(hexstr=ipfs_hash_hex)
                    logger.info(f"Client {self.client_id}: Using IPFS hash for blockchain: {result['ipfs_hash']} -> {ipfs_hash_hex}")
                except ImportError:
                    # Fallback: use SHA256 of IPFS hash string
                    import hashlib
                    ipfs_hash_bytes = result['ipfs_hash'].encode('utf-8')
                    ipfs_hash_hex = hashlib.sha256(ipfs_hash_bytes).hexdigest()
                    model_hash = self.web3.to_bytes(hexstr=ipfs_hash_hex)
                    logger.info(f"Client {self.client_id}: Using SHA256 of IPFS hash for blockchain: {result['ipfs_hash']} -> {ipfs_hash_hex}")
            else:
                # Generate traditional model hash
                model_hash = self.generate_model_hash(parameters)
                logger.info(f"Client {self.client_id}: Using model hash for blockchain")
            
            # Submit to blockchain
            # Smart contract expects: submitModelUpdate(modelHash, round)
            tx = self.contract.functions.submitModelUpdate(model_hash, round_num).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.blockchain_config['private_key'])
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                result['success'] = True
                result['blockchain_hash'] = model_hash.hex()
                result['blockchain_tx'] = tx_hash.hex()
                
                logger.info(f"Client {self.client_id}: Model submitted successfully")
                logger.info(f"  IPFS Hash: {result['ipfs_hash'] or 'N/A'}")
                logger.info(f"  Blockchain Hash: {result['blockchain_hash']}")
                logger.info(f"  Transaction: {result['blockchain_tx']}")
                logger.info(f"  Round: {round_num}")
            else:
                logger.error(f"Client {self.client_id}: Blockchain transaction failed")
                
        except Exception as e:
            logger.error(f"Client {self.client_id}: Submission failed: {str(e)}")
            logger.error(f"Client {self.client_id}: Error type: {type(e).__name__}")
            if hasattr(e, 'args') and e.args:
                logger.error(f"Client {self.client_id}: Error args: {e.args}")
        
        return result
    
    def verify_model_integrity(self, model_hash):
        """
        Verify model integrity on blockchain
        
        Args:
            model_hash: Hash of the model to verify
            
        Returns:
            is_valid: Whether the model is valid
        """
        if not self.contract:
            return False
        
        try:
            is_valid = self.contract.functions.verifyModelIntegrity(model_hash).call()
            return is_valid
        except Exception as e:
            logger.error(f"Client {self.client_id}: Model verification failed: {str(e)}")
            return False
    
    def load_local_data(self, max_samples=10000):
        """
        Load local data for this client
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            success: Whether data loading was successful
        """
        try:
            logger.info(f"Client {self.client_id}: Loading local data...")
            
            # Load training data
            train_sequences, train_labels = self.preprocessor.load_training_data(max_samples_per_file=max_samples)
            
            if train_sequences is None:
                logger.error(f"Client {self.client_id}: Failed to load training data")
                return False
            
            # Scale sequences
            self.local_sequences = self.preprocessor.scale_sequences(train_sequences)
            self.local_labels = train_labels
            
            logger.info(f"Client {self.client_id}: Loaded {len(self.local_sequences)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading data: {str(e)}")
            return False

    def create_paper_split_with_non_iid(self, num_clients=3, alpha=10.0):
        """
        Create paper's train/validation/test split with realistic non-IID distribution
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet distribution parameter (α=10.0 for moderate heterogeneity)
            
        Returns:
            split_data: Dictionary with train/validation/test splits
        """
        logger.info(f"Creating paper split with non-IID distribution (α={alpha})")
        
        # Load all available data
        train_sequences, train_labels = self.preprocessor.load_training_data(max_samples_per_file=200000)
        test_sequences, test_labels = self.preprocessor.load_testing_data(max_samples_per_file=200000)
        
        if train_sequences is None or test_sequences is None:
            logger.error("Failed to load data for paper split")
            return None
        
        logger.info(f"Training data: {len(train_sequences):,} samples")
        logger.info(f"Testing data: {len(test_sequences):,} samples")
        
        # Paper's exact split implementation
        # Step 1: Training Set (Normal traffic only) - Paper target: 45,408
        train_set = train_sequences[:min(len(train_sequences), 45408)]
        logger.info(f"Training set: {len(train_set):,} samples")
        
        # Step 2: Validation Set (Mixed normal + known attacks) - Paper target: 79,394
        remaining_normal = train_sequences[len(train_set):]
        val_normal = remaining_normal[:11342] if len(remaining_normal) >= 11342 else remaining_normal
        val_attack_available = test_sequences[:11342 * 6] if len(test_sequences) >= 11342 * 6 else test_sequences
        
        # Handle case where val_normal is empty
        if len(val_normal) == 0:
            val_set = val_attack_available
            val_labels = np.ones(len(val_attack_available))  # All attack
        else:
            val_set = np.concatenate([val_normal, val_attack_available])
            val_labels = np.concatenate([
                np.zeros(len(val_normal)),  # Normal
                np.ones(len(val_attack_available))  # Attack
            ])
        logger.info(f"Validation set: {len(val_set):,} samples")
        
        # Step 3: Testing Set (Mixed normal + all attacks including zero-day) - Paper target: 97,889
        remaining_normal_test = train_sequences[len(train_set) + len(val_normal):]
        test_normal = remaining_normal_test[:50000] if len(remaining_normal_test) >= 50000 else remaining_normal_test
        remaining_attacks = test_sequences[len(val_attack_available):]
        
        # Handle case where test_normal is empty
        if len(test_normal) == 0:
            test_set = remaining_attacks
            test_labels = np.ones(len(remaining_attacks))  # All attack
        else:
            test_set = np.concatenate([test_normal, remaining_attacks])
            test_labels = np.concatenate([
                np.zeros(len(test_normal)),  # Normal
                np.ones(len(remaining_attacks))  # Attack
            ])
        logger.info(f"Testing set: {len(test_set):,} samples")
        
        # Create non-IID distribution for training data
        client_data = self._create_non_iid_distribution(train_set, num_clients, alpha)
        
        return {
            'train': train_set,
            'val': val_set,
            'val_labels': val_labels,
            'test': test_set,
            'test_labels': test_labels,
            'client_data': client_data,
            'paper_config': {
                'training': {'target': 45408, 'actual': len(train_set)},
                'validation': {'target': 11342 + (11342 * 6), 'actual': len(val_set)},
                'testing': {'target': 50000 + 1873 + 8022 + (8021 * 5), 'actual': len(test_set)}
            }
        }

    def _create_non_iid_distribution(self, train_data, num_clients=3, alpha=10.0):
        """
        Create realistic non-IID data distribution using Dirichlet distribution
        
        Args:
            train_data: Training data to distribute
            num_clients: Number of federated clients
            alpha: Dirichlet distribution parameter (α=10.0 for moderate heterogeneity)
            
        Returns:
            client_data: Dictionary with data for each client
        """
        logger.info(f"Creating non-IID distribution (α={alpha}) for {num_clients} clients")
        
        num_samples = len(train_data)
        
        # Create Dirichlet distribution for realistic data partitioning
        # α = 10.0 creates moderate heterogeneity (balanced non-IID)
        dirichlet_dist = np.random.dirichlet([alpha] * num_clients)
        logger.info(f"Dirichlet distribution: {dirichlet_dist}")
        
        # Calculate samples per client based on Dirichlet distribution
        samples_per_client = [int(ratio * num_samples) for ratio in dirichlet_dist]
        
        # Adjust to ensure all samples are distributed
        remaining_samples = num_samples - sum(samples_per_client)
        samples_per_client[0] += remaining_samples
        
        logger.info(f"Samples per client:")
        for i, samples in enumerate(samples_per_client):
            percentage = (samples / num_samples) * 100
            logger.info(f"  Client {i+1}: {samples:,} samples ({percentage:.1f}%)")
        
        # Distribute data among clients
        client_data = {}
        start_idx = 0
        
        for i in range(num_clients):
            end_idx = start_idx + samples_per_client[i]
            client_data[f'client_{i+1}'] = {
                'data': train_data[start_idx:end_idx],
                'samples': samples_per_client[i],
                'percentage': (samples_per_client[i] / num_samples) * 100
            }
            start_idx = end_idx
        
        return client_data

    def analyze_data_heterogeneity(self, client_data):
        """
        Analyze the heterogeneity of data distribution across clients
        
        Args:
            client_data: Dictionary with client data distribution
            
        Returns:
            heterogeneity_metrics: Dictionary with heterogeneity analysis
        """
        samples_per_client = [client['samples'] for client in client_data.values()]
        
        # Calculate heterogeneity metrics
        mean_samples = np.mean(samples_per_client)
        std_samples = np.std(samples_per_client)
        cv_samples = std_samples / mean_samples  # Coefficient of variation
        
        # Interpret heterogeneity
        if cv_samples > 0.5:
            heterogeneity_level = "HIGH"
        elif cv_samples > 0.2:
            heterogeneity_level = "MEDIUM"
        else:
            heterogeneity_level = "LOW"
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_samples = np.sort(samples_per_client)
        n = len(sorted_samples)
        cumsum = np.cumsum(sorted_samples)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        heterogeneity_metrics = {
            'mean': mean_samples,
            'std': std_samples,
            'cv': cv_samples,
            'gini': gini,
            'heterogeneity_level': heterogeneity_level
        }
        
        logger.info(f"Heterogeneity Analysis:")
        logger.info(f"  Mean: {mean_samples:.0f} samples")
        logger.info(f"  Std Dev: {std_samples:.0f} samples")
        logger.info(f"  CV: {cv_samples:.3f}")
        logger.info(f"  Gini: {gini:.3f}")
        logger.info(f"  Level: {heterogeneity_level}")
        
        return heterogeneity_metrics
    
    def train_locally(self, epochs=10, batch_size=16):
        """
        Train the GAN-TCN model locally using simplified autoencoder approach
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            training_metrics: Dictionary with training metrics
        """
        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs")
        
        # Convert data to PyTorch tensors
        sequences = torch.FloatTensor(self.local_sequences).to(self.device)
        
        # Training metrics
        epoch_metrics = []
        
        for epoch in range(epochs):
            epoch_losses = {'reconstruction_loss': []}
            
            # Create batches
            num_batches = len(sequences) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_sequences = sequences[start_idx:end_idx]
                
                # Train generator (autoencoder mode)
                self.optimizer_g.zero_grad()
                
                # Generate reconstructed sequences
                reconstructed_sequences, _, _ = self.model.generator(batch_sequences)
                
                # Simple reconstruction loss (MSE)
                reconstruction_loss = torch.mean((batch_sequences - reconstructed_sequences) ** 2)
                
                # Add L1 regularization for sparsity
                l1_loss = 0.0
                for param in self.model.generator.parameters():
                    l1_loss += torch.mean(torch.abs(param))
                
                # Total loss
                total_loss = reconstruction_loss + 0.0001 * l1_loss
                
                total_loss.backward()
                
                # Clear GPU cache after backward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Very aggressive gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=0.01)
                
                self.optimizer_g.step()
                
                # Store losses
                epoch_losses['reconstruction_loss'].append(reconstruction_loss.item())
            
            # Calculate average losses for this epoch
            avg_reconstruction_loss = np.mean(epoch_losses['reconstruction_loss'])
            
            # Apply loss bounds and handle numerical instability
            if np.isnan(avg_reconstruction_loss) or np.isinf(avg_reconstruction_loss):
                logger.warning(f"Client {self.client_id}: Invalid loss detected ({avg_reconstruction_loss}), setting to 1.0")
                avg_reconstruction_loss = 1.0
            elif avg_reconstruction_loss > 10.0:
                logger.warning(f"Client {self.client_id}: Reconstruction loss too high ({avg_reconstruction_loss:.4f}), clipping to 10.0")
                avg_reconstruction_loss = 10.0
            
            epoch_metrics.append({
                'epoch': epoch + 1,
                'reconstruction_loss': avg_reconstruction_loss
            })
            
            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{epochs} - "
                       f"Reconstruction Loss: {avg_reconstruction_loss:.4f}")
            
            # Early stopping if loss is still too high
            if avg_reconstruction_loss > 5.0 and epoch > 1:
                logger.warning(f"Client {self.client_id}: Early stopping due to high reconstruction loss")
                break
        
        # Store training history
        self.training_history.extend(epoch_metrics)
        
        return {
            'client_id': self.client_id,
            'epochs_trained': epochs,
            'final_reconstruction_loss': epoch_metrics[-1]['reconstruction_loss'],
            'training_history': epoch_metrics
        }
    
    def get_model_parameters(self):
        """
        Get current model parameters for aggregation
        
        Returns:
            parameters: Dictionary with model parameters
        """
        parameters = {}
        
        # Get generator parameters
        for name, param in self.model.generator.named_parameters():
            parameters[f'generator.{name}'] = param.data.clone().detach().cpu().numpy()
        
        # Get discriminator parameters
        for name, param in self.model.discriminator.named_parameters():
            parameters[f'discriminator.{name}'] = param.data.clone().detach().cpu().numpy()
        
        return parameters
    
    def update_model_parameters(self, aggregated_parameters):
        """
        Update model with aggregated parameters
        
        Args:
            aggregated_parameters: Dictionary with aggregated parameters
        """
        logger.info(f"Client {self.client_id}: Updating model with aggregated parameters")
        
        with torch.no_grad():
            # Update generator parameters
            for name, param in self.model.generator.named_parameters():
                param_key = f'generator.{name}'
                if param_key in aggregated_parameters:
                    param.copy_(torch.FloatTensor(aggregated_parameters[param_key]).to(self.device))
            
            # Update discriminator parameters
            for name, param in self.model.discriminator.named_parameters():
                param_key = f'discriminator.{name}'
                if param_key in aggregated_parameters:
                    param.copy_(torch.FloatTensor(aggregated_parameters[param_key]).to(self.device))
    
    def evaluate_zero_day_detection(self, test_sequences, test_labels, threshold=None):
        """
        Evaluate zero-day attack detection performance using reconstruction error
        
        Args:
            test_sequences: Test sequences
            test_labels: Test labels
            threshold: Anomaly threshold
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        logger.info(f"Client {self.client_id}: Evaluating zero-day detection...")
        
        # Convert to PyTorch tensors
        test_tensor = torch.FloatTensor(test_sequences).to(self.device)
        
        # Calculate reconstruction errors
        with torch.no_grad():
            reconstructed_sequences, _, _ = self.model.generator(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - reconstructed_sequences) ** 2, dim=(1, 2))
        
        # Convert to numpy
        errors_np = reconstruction_errors.detach().cpu().numpy()
        
        # Set threshold if not provided (use 95th percentile of errors)
        if threshold is None:
            threshold = np.percentile(errors_np, 95)
        
        # Make predictions based on reconstruction error
        predictions_np = (errors_np > threshold).astype(int)
        
        # Calculate metrics
        accuracy = (predictions_np == test_labels).mean()
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions_np, average='binary')
        except:
            precision = recall = f1 = 0.0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(test_labels, predictions_np)
            tn, fp, fn, tp = cm.ravel()
        except:
            tn = fp = fn = tp = 0
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(test_labels, errors_np)
        except:
            roc_auc = 0.0
        
        return {
            'client_id': self.client_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'mean_error': float(np.mean(errors_np)),
            'std_error': float(np.std(errors_np)),
            'threshold': float(threshold),
            'detection_rate': float(np.mean(predictions_np))
        }

class BlockchainFederatedAggregator:
    """
    Blockchain Federated Learning Aggregator
    """
    
    def __init__(self, num_clients=3, device='cpu', blockchain_config=None):
        """
        Initialize the federated learning aggregator with blockchain integration
        
        Args:
            num_clients: Number of federated learning clients
            device: Device to run aggregation on
            blockchain_config: Blockchain configuration dictionary
        """
        self.num_clients = num_clients
        self.device = device
        self.blockchain_config = blockchain_config
        self.clients = []
        
        # Initialize clients with their own blockchain configs
        for i in range(num_clients):
            from blockchain_config import get_client_blockchain_config
            client_config = get_client_blockchain_config(f"client_{i+1}")
            client = BlockchainFederatedClient(f"client_{i+1}", device=device, blockchain_config=client_config)
            self.clients.append(client)
        
        # Initialize global model
        from blockchain_gan_tcn_architecture import GANTCNModel
        self.global_model = GANTCNModel(
            input_dim=40,
            sequence_length=30,
            latent_dim=10,
            hidden_dims=[64, 128, 256],
            kernel_size=3,
            dropout=0.2
        )
        
        # Move global model to device
        self.global_model = self.global_model.to(self.device)
        
        # Blockchain integration for aggregator
        self.web3 = None
        self.contract = None
        self.account = None
        
        # IPFS integration for aggregator
        self.ipfs_client = None
        self.ipfs_available = False  # Disable IPFS to avoid connection issues
        
        if blockchain_config:
            self._initialize_blockchain()
        
        # IPFS integration with simple client
        try:
            from simple_ipfs_client import SimpleIPFSClient
            self.ipfs_client = SimpleIPFSClient()
            if self.ipfs_client.test_connection():
                self.ipfs_available = True
                logger.info(f"Aggregator: IPFS enabled and connected")
            else:
                self.ipfs_available = False
                logger.warning(f"Aggregator: IPFS connection failed, running without IPFS")
        except Exception as e:
            self.ipfs_client = None
            self.ipfs_available = False
            logger.warning(f"Aggregator: IPFS not available: {str(e)}, running without IPFS")
        logger.info(f"Federated aggregator initialized with {num_clients} clients and blockchain integration")
    
    def _initialize_blockchain(self):
        """
        Initialize blockchain connection and smart contract for aggregator
        """
        try:
            # Connect to Ethereum network
            self.web3 = Web3(Web3.HTTPProvider(self.blockchain_config['rpc_url']))
            
            # Add middleware for PoA networks (like Hardhat)
            try:
                from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
                self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                logger.warning("Aggregator: ExtraDataToPOAMiddleware not available, skipping")
            
            # Check connection
            if not self.web3.is_connected():
                logger.error("Aggregator: Failed to connect to blockchain")
                return False
            
            # Load account
            self.account = Account.from_key(self.blockchain_config['private_key'])
            self.web3.eth.default_account = self.account.address
            
            # Load smart contract
            contract_address = self.blockchain_config['contract_address']
            contract_abi = self.blockchain_config['contract_abi']
            self.contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
            
            logger.info("Aggregator: Blockchain initialized successfully")
            logger.info(f"  Account: {self.account.address}")
            logger.info(f"  Contract: {contract_address}")
            return True
            
        except Exception as e:
            logger.error(f"Aggregator: Blockchain initialization failed: {str(e)}")
            return False
    
    def _initialize_ipfs(self):
        """
        Initialize IPFS client connection for aggregator using simple client
        """
        try:
            from simple_ipfs_client import SimpleIPFSClient
            self.ipfs_client = SimpleIPFSClient()
            if self.ipfs_client.test_connection():
                logger.info("Aggregator: IPFS initialized successfully")
                return True
            else:
                logger.warning("Aggregator: Could not connect to IPFS")
                self.ipfs_client = None
                self.ipfs_available = False
                return False
        except Exception as e:
            logger.warning(f"Aggregator: Could not connect to IPFS: {str(e)}")
            self.ipfs_client = None
            self.ipfs_available = False
            return False
    
    def aggregate_models_on_blockchain(self, round_num):
        """
        Aggregate models on blockchain with optional IPFS storage
        
        Args:
            round_num: Current federated learning round
            
        Returns:
            result: Dictionary with aggregation results
        """
        result = {
            'success': False,
            'ipfs_hash': None,
            'blockchain_hash': None,
            'blockchain_tx': None
        }
        
        if not self.contract:
            logger.warning("Aggregator: No blockchain contract available")
            return result
        
        try:
            # Get aggregated model parameters
            aggregated_parameters = self.aggregate_models_fedavg([
                client.get_model_parameters() for client in self.clients
            ])
            
            # Step 1: Upload aggregated model to IPFS if available
            if self.ipfs_client:
                metadata = {
                    'aggregator_id': 'federated_aggregator',
                    'round': round_num,
                    'timestamp': int(time.time()),
                    'model_type': 'gan_tcn_aggregated',
                    'parameters_count': len(aggregated_parameters),
                    'num_clients': len(self.clients)
                }
                
                ipfs_hash = self.upload_aggregated_model_to_ipfs(aggregated_parameters, metadata)
                if ipfs_hash:
                    result['ipfs_hash'] = ipfs_hash
                    logger.info(f"Aggregator: Aggregated model uploaded to IPFS: {ipfs_hash}")
                else:
                    logger.warning("Aggregator: IPFS upload failed, continuing with blockchain-only")
            else:
                logger.info("Aggregator: IPFS not available, using blockchain-only approach")
            
            # Step 2: Submit to blockchain
            if result['ipfs_hash']:
                # Convert IPFS hash (base58) to proper hex format for blockchain
                try:
                    import base58
                    # Decode base58 IPFS hash to bytes
                    ipfs_hash_bytes = base58.b58decode(result['ipfs_hash'])
                    # Convert to hex and pad to 32 bytes (64 hex characters)
                    ipfs_hash_hex = ipfs_hash_bytes.hex().ljust(64, '0')[:64]
                    aggregated_hash = self.web3.to_bytes(hexstr=ipfs_hash_hex)
                    logger.info(f"Aggregator: Using IPFS hash for blockchain: {result['ipfs_hash']} -> {ipfs_hash_hex}")
                except ImportError:
                    # Fallback: use SHA256 of IPFS hash string
                    import hashlib
                    ipfs_hash_bytes = result['ipfs_hash'].encode('utf-8')
                    ipfs_hash_hex = hashlib.sha256(ipfs_hash_bytes).hexdigest()
                    aggregated_hash = self.web3.to_bytes(hexstr=ipfs_hash_hex)
                    logger.info(f"Aggregator: Using SHA256 of IPFS hash for blockchain: {result['ipfs_hash']} -> {ipfs_hash_hex}")
            else:
                # Generate traditional model hash
                param_str = json.dumps(aggregated_parameters, sort_keys=True, default=str)
                hash_object = hashlib.sha256(param_str.encode())
                aggregated_hash = self.web3.to_bytes(hexstr=hash_object.hexdigest())
                logger.info("Aggregator: Using model hash for blockchain")
            
            # Submit aggregation to blockchain
            # Smart contract expects: aggregateModels(round, aggregatedHash)
            tx = self.contract.functions.aggregateModels(round_num, aggregated_hash).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.blockchain_config['private_key'])
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                result['success'] = True
                result['blockchain_hash'] = aggregated_hash.hex()
                result['blockchain_tx'] = tx_hash.hex()
                
                logger.info("Aggregator: Models aggregated successfully")
                logger.info(f"  IPFS Hash: {result['ipfs_hash'] or 'N/A'}")
                logger.info(f"  Blockchain Hash: {result['blockchain_hash']}")
                logger.info(f"  Transaction: {result['blockchain_tx']}")
                logger.info(f"  Round: {round_num}")
            else:
                logger.error("Aggregator: Blockchain aggregation failed")
                
        except Exception as e:
            logger.error(f"Aggregator: Aggregation failed: {str(e)}")
            logger.error(f"Aggregator: Error type: {type(e).__name__}")
            if hasattr(e, 'args') and e.args:
                logger.error(f"Aggregator: Error args: {e.args}")
        
        return result
    
    def upload_aggregated_model_to_ipfs(self, parameters, metadata=None):
        """
        Upload aggregated model parameters to IPFS
        
        Args:
            parameters: Aggregated model parameters dictionary
            metadata: Additional metadata (round, num_clients, etc.)
            
        Returns:
            ipfs_hash: IPFS hash of uploaded model
        """
        if not self.ipfs_client:
            logger.warning("Aggregator: IPFS not available")
            return None
        
        try:
            import pickle
            
            # Move model parameters to CPU before serialization to avoid GPU memory issues
            if isinstance(parameters, torch.Tensor):
                parameters_cpu = parameters.cpu()
            elif isinstance(parameters, dict):
                parameters_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in parameters.items()}
            else:
                parameters_cpu = parameters
            
            # Prepare model data (on CPU)
            model_data = {
                'parameters': parameters_cpu,
                'metadata': metadata or {}
            }
            
            # Clear GPU cache before serialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Serialize model data (now on CPU)
            serialized_data = pickle.dumps(model_data)
            
            # Upload to IPFS using simple client
            ipfs_hash = self.ipfs_client.add_data(model_data)
            
            logger.info(f"Aggregator: Aggregated model uploaded to IPFS: {ipfs_hash}")
            logger.info(f"  Size: {len(serialized_data)} bytes")
            
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Aggregator: Failed to upload aggregated model to IPFS: {str(e)}")
            return None
    
    def download_aggregated_model_from_ipfs(self, ipfs_hash):
        """
        Download aggregated model parameters from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the aggregated model
            
        Returns:
            model_data: Dictionary with parameters and metadata
        """
        if not self.ipfs_client:
            logger.warning("Aggregator: IPFS not available")
            return None
        
        try:
            # Download from IPFS using simple client
            model_data = self.ipfs_client.get_data(ipfs_hash)
            
            # Check if data was retrieved successfully
            if model_data is None:
                logger.error(f"Aggregator: Failed to retrieve data from IPFS")
                return None
            
            logger.info(f"Aggregator: Aggregated model downloaded from IPFS: {ipfs_hash}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Aggregator: Failed to download aggregated model from IPFS: {str(e)}")
            return None
    
    def get_blockchain_round_data(self, round_num):
        """
        Get round data from blockchain
        
        Args:
            round_num: Round number
            
        Returns:
            round_data: Dictionary with round information
        """
        if not self.contract:
            return None
        
        try:
            # Get round data from blockchain
            model_hashes, aggregated_hash, timestamp, is_complete = self.contract.functions.getRoundData(round_num).call()
            
            return {
                'model_hashes': [h.hex() for h in model_hashes],
                'aggregated_hash': aggregated_hash.hex() if aggregated_hash != b'\x00' * 32 else None,
                'timestamp': timestamp,
                'is_complete': is_complete,
                'round_num': round_num
            }
        except Exception as e:
            logger.error(f"Aggregator: Failed to get round data: {str(e)}")
            return None
    
    def prepare_federated_data(self, use_paper_split=True, alpha=10.0):
        """
        Prepare data for all clients using paper's split with non-IID distribution
        
        Args:
            use_paper_split: Whether to use paper's train/validation/test split
            alpha: Dirichlet distribution parameter for non-IID distribution (α=10.0 for moderate heterogeneity)
            
        Returns:
            success: Whether data preparation was successful
        """
        logger.info("Preparing federated data using paper's split with non-IID distribution...")
        
        if use_paper_split:
            # Use paper's split with non-IID distribution
            split_data = self.clients[0].create_paper_split_with_non_iid(
                num_clients=self.num_clients, alpha=alpha
            )
            
            if split_data is None:
                logger.error("Failed to create paper split")
                return False
            
            # Store validation and test data for centralized evaluation
            self.validation_sequences = split_data['val']
            self.validation_labels = split_data['val_labels']
            self.test_sequences = split_data['test']
            self.test_labels = split_data['test_labels']
            
            # Analyze heterogeneity
            heterogeneity_metrics = self.clients[0].analyze_data_heterogeneity(split_data['client_data'])
            logger.info(f"Data heterogeneity level: {heterogeneity_metrics['heterogeneity_level']}")
            
            # Distribute training data to clients using non-IID distribution
            for i, client in enumerate(self.clients):
                client_id = f"client_{i+1}"
                if client_id in split_data['client_data']:
                    client_data = split_data['client_data'][client_id]
                    client.local_sequences = client_data['data']
                    client.local_labels = np.zeros(len(client_data['data']))  # All normal for training
                    logger.info(f"Distributed {len(client.local_sequences)} samples to {client_id} ({client_data['percentage']:.1f}%)")
                else:
                    logger.error(f"No data found for {client_id}")
                    return False
            
            logger.info(f"Paper split completed:")
            logger.info(f"  Training: {len(split_data['train']):,} samples (non-IID distributed)")
            logger.info(f"  Validation: {len(split_data['val']):,} samples (centralized)")
            logger.info(f"  Testing: {len(split_data['test']):,} samples (centralized)")
            
        else:
            # Use original federated data preparation
            preprocessor = BlockchainPreprocessor()
            client_data = preprocessor.prepare_federated_data(num_clients=self.num_clients)
            
            if client_data is None:
                logger.error("Failed to prepare federated data")
                return False
            
            # Distribute data to clients
            for i, client in enumerate(self.clients):
                client_id = f"client_{i+1}"
                if client_id in client_data:
                    client.local_sequences = client_data[client_id]['sequences']
                    client.local_labels = client_data[client_id]['labels']
                    logger.info(f"Distributed {len(client.local_sequences)} samples to {client_id}")
                else:
                    logger.error(f"No data found for {client_id}")
                    return False
        
        return True
    
    def aggregate_models_fedavg(self, client_parameters_list):
        """
        Aggregate client models using FedAvg algorithm
        
        Args:
            client_parameters_list: List of client parameter dictionaries
            
        Returns:
            aggregated_parameters: Dictionary with aggregated parameters
        """
        logger.info("Aggregating models using FedAvg...")
        
        logger.debug(f"Starting FedAvg aggregation with {len(client_parameters_list)} clients")
        
        if not client_parameters_list:
            logger.error("No client parameters provided for aggregation")
            return None
        
        # Initialize aggregated parameters
        aggregated_parameters = {}
        
        # Get parameter names from first client
        param_names = list(client_parameters_list[0].keys())
        logger.debug(f"Parameter names: {param_names}")
        
        # FedAvg aggregation
        for param_name in param_names:
            # Collect parameter from all clients
            client_params = []
            for client_params_dict in client_parameters_list:
                if param_name in client_params_dict:
                    client_params.append(client_params_dict[param_name])
            
            if client_params:
                # Average the parameters
                aggregated_param = np.zeros_like(client_params[0])
                for param in client_params:
                    aggregated_param += param
                aggregated_param /= len(client_params)
                
                aggregated_parameters[param_name] = aggregated_param
        
        logger.info(f"Aggregated {len(aggregated_parameters)} parameters from {len(client_parameters_list)} clients")
        return aggregated_parameters
    
    def run_federated_round(self, epochs=3, batch_size=32, round_num=0):
        """
        Run one federated learning round with blockchain integration
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training
            round_num: Current round number for blockchain
            
        Returns:
            round_results: Dictionary with round results
        """
        logger.info(f"Starting federated learning round {round_num} with blockchain integration...")
        
        # Train all clients locally
        client_results = []
        client_parameters = []
        blockchain_submissions = []
        
        for client in self.clients:
            # Train client locally
            training_result = client.train_locally(epochs=epochs, batch_size=batch_size)
            client_results.append(training_result)
            
                    # Get client parameters
        try:
            parameters = client.get_model_parameters()
            if parameters is None or len(parameters) == 0:
                logger.error(f"Client {client.client_id}: No model parameters returned")
                return None
            client_parameters.append(parameters)
            logger.debug(f"Client {client.client_id}: Got {len(parameters)} parameters")
        except Exception as e:
            logger.error(f"Client {client.client_id}: Failed to get model parameters: {str(e)}")
            return None
            
            # Submit model to blockchain (temporarily disabled due to ABI mismatch)
            blockchain_submissions.append(False)
            logger.info(f"Client {client.client_id} blockchain submission: SKIPPED (ABI mismatch)")
            
            logger.info(f"Client {client.client_id} training completed")
        
        # Aggregate models
        logger.debug(f"Aggregating {len(client_parameters)} client models...")
        aggregated_parameters = self.aggregate_models_fedavg(client_parameters)
        
        if aggregated_parameters is None:
            logger.error("Model aggregation failed")
            return None
        
        logger.debug(f"Successfully aggregated {len(aggregated_parameters)} parameters")
        
        # Update all clients with aggregated parameters
        for client in self.clients:
            client.update_model_parameters(aggregated_parameters)
        
        # Aggregate on blockchain (temporarily disabled due to ABI mismatch)
        blockchain_aggregation = False
        blockchain_aggregation_result = None
        logger.info(f"Blockchain aggregation: SKIPPED (ABI mismatch)")
        
        # Create hash for verification
        model_hash = hashlib.sha256(
            str(aggregated_parameters).encode()
        ).hexdigest()
        
        # Get blockchain round data if available
        blockchain_round_data = None
        if self.blockchain_config:
            try:
                blockchain_round_data = self.get_blockchain_round_data(round_num)
            except Exception as e:
                logger.warning(f"Could not get blockchain round data: {str(e)}")
                blockchain_round_data = None
        
        return {
            'client_results': client_results,
            'aggregated_parameters': aggregated_parameters,
            'model_hash': model_hash,
            'num_clients': len(self.clients),
            'blockchain_submissions': blockchain_submissions,
            'blockchain_aggregation': blockchain_aggregation,
            'blockchain_aggregation_result': blockchain_aggregation_result,
            'blockchain_round_data': blockchain_round_data,
            'round_num': round_num
        }
    
    def evaluate_federated_model(self, test_sequences, test_labels, threshold=0.1):
        """
        Evaluate the federated model on test data
        
        Args:
            test_sequences: Test sequences
            test_labels: Test labels
            threshold: Anomaly threshold
            
        Returns:
            evaluation_results: Dictionary with evaluation results
        """
        logger.info("Evaluating federated model...")
        
        # Evaluate each client
        client_evaluations = []
        for client in self.clients:
            evaluation = client.evaluate_zero_day_detection(test_sequences, test_labels, threshold)
            client_evaluations.append(evaluation)
        
        # Calculate average metrics
        avg_accuracy = np.mean([eval['accuracy'] for eval in client_evaluations])
        avg_precision = np.mean([eval['precision'] for eval in client_evaluations])
        avg_recall = np.mean([eval['recall'] for eval in client_evaluations])
        avg_f1 = np.mean([eval['f1_score'] for eval in client_evaluations])
        avg_detection_rate = np.mean([eval['detection_rate'] for eval in client_evaluations])
        
        return {
            'client_evaluations': client_evaluations,
            'average_accuracy': avg_accuracy,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1_score': avg_f1,
            'average_detection_rate': avg_detection_rate,
            'num_clients': len(self.clients)
        }

if __name__ == "__main__":
    # Test the blockchain federated learning system
    print("Testing blockchain federated learning with real GAN-TCN architecture...")
    
    # Initialize aggregator
    aggregator = BlockchainFederatedAggregator(num_clients=3)
    
    # Prepare federated data
    if aggregator.prepare_federated_data():
        print("✅ Federated data preparation successful!")
        
        # Run one federated round
        round_results = aggregator.run_federated_round(epochs=2)
        
        if round_results:
            print("✅ Federated learning round completed!")
            print(f"   Model Hash: {round_results['model_hash'][:20]}...")
            print(f"   Clients: {round_results['num_clients']}")
            
            # Show client results
            for result in round_results['client_results']:
                print(f"   Client {result['client_id']}: G Loss: {result['final_g_loss']:.4f}, "
                      f"D Loss: {result['final_d_loss']:.4f}")
        else:
            print("❌ Federated learning round failed!")
    else:
        print("❌ Federated data preparation failed!")
    
    print("Blockchain federated learning test completed!")
