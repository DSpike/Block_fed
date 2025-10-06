#!/usr/bin/env python3
"""
Secure Federated Learning Client
Implements privacy-preserving model updates using IPFS-only transmission
"""

import torch
import time
import logging
import hashlib
import pickle
import base64
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from decentralized_fl_system import SecureModelUpdate, SecureEncryptionManager, SecureHashManager, SecureSignatureManager
from real_ipfs_client import RealIPFSClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecureFederatedClient:
    """
    Secure federated learning client that uses IPFS-only transmission
    """
    
    def __init__(self, client_id: str, ipfs_client=None, encryption_password: str = None):
        self.client_id = client_id
        self.ipfs_client = ipfs_client
        
        # Security managers
        self.encryption_manager = SecureEncryptionManager()
        self.hash_manager = SecureHashManager()
        self.signature_manager = SecureSignatureManager()
        
        # Generate client keys
        self.encryption_key = self.encryption_manager.generate_client_key(client_id, encryption_password)
        self.private_key, self.public_key = self.signature_manager.generate_client_keypair(client_id)
        
        logger.info(f"Secure client {client_id} initialized with encryption and signature keys")
    
    def create_secure_model_update(self, 
                                 model_parameters: Dict[str, torch.Tensor],
                                 sample_count: int,
                                 accuracy: float,
                                 loss: float,
                                 round_number: int) -> Optional[SecureModelUpdate]:
        """
        Create a secure model update using IPFS-only transmission
        """
        try:
            # Step 1: Encrypt model parameters
            encrypted_params = self.encryption_manager.encrypt_model_parameters(
                model_parameters, self.client_id
            )
            
            # Step 2: Compute model hash for verification
            model_hash = self.hash_manager.compute_model_hash(model_parameters)
            
            # Step 3: Store encrypted data on IPFS
            if not self.ipfs_client:
                logger.error(f"Client {self.client_id}: No IPFS client available")
                return None
            
            # Prepare secure data for IPFS
            secure_data = {
                'encrypted_parameters': encrypted_params,
                'client_id': self.client_id,
                'timestamp': time.time(),
                'encryption_method': 'fernet',
                'model_hash': model_hash
            }
            
            # Store on IPFS
            ipfs_cid = self.ipfs_client.add_data(secure_data)
            if not ipfs_cid:
                logger.error(f"Client {self.client_id}: Failed to store data on IPFS")
                return None
            
            # Step 4: Create digital signature
            timestamp = time.time()
            signature_data = f"{self.client_id}_{ipfs_cid}_{model_hash}_{timestamp}"
            signature = self.signature_manager.sign_data(signature_data, self.client_id)
            
            # Step 5: Create secure model update
            secure_update = SecureModelUpdate(
                client_id=self.client_id,
                ipfs_cid=ipfs_cid,
                model_hash=model_hash,
                sample_count=sample_count,
                accuracy=accuracy,
                loss=loss,
                timestamp=timestamp,
                signature=signature,
                round_number=round_number,
                encryption_method="fernet"
            )
            
            logger.info(f"Client {self.client_id}: Created secure model update with CID {ipfs_cid}")
            return secure_update
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to create secure model update: {e}")
            return None
    
    def verify_secure_model_update(self, update: SecureModelUpdate) -> bool:
        """
        Verify a secure model update (for testing purposes)
        """
        try:
            # Verify signature
            signature_data = f"{update.client_id}_{update.ipfs_cid}_{update.model_hash}_{update.timestamp}"
            if not self.signature_manager.verify_signature(signature_data, update.signature, update.client_id):
                logger.error(f"Client {self.client_id}: Signature verification failed")
                return False
            
            # Retrieve and verify data from IPFS
            if not self.ipfs_client:
                logger.error(f"Client {self.client_id}: No IPFS client available")
                return False
            
            encrypted_data = self.ipfs_client.get_data(update.ipfs_cid)
            if not encrypted_data:
                logger.error(f"Client {self.client_id}: Failed to retrieve data from IPFS")
                return False
            
            # Decrypt and verify
            decrypted_params = self.encryption_manager.decrypt_model_parameters(
                encrypted_data['encrypted_parameters'], self.client_id
            )
            
            if not self.hash_manager.verify_model_hash(decrypted_params, update.model_hash):
                logger.error(f"Client {self.client_id}: Model hash verification failed")
                return False
            
            logger.info(f"Client {self.client_id}: Secure model update verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Secure model update verification failed: {e}")
            return False
    
    def get_public_key(self) -> str:
        """Get client's public key for signature verification"""
        return self.signature_manager.client_keys[self.client_id]['public']
    
    def get_encryption_key(self) -> str:
        """Get client's encryption key (for miner setup)"""
        return self.encryption_manager.client_keys[self.client_id]

class MockIPFSClient:
    """
    Mock IPFS client for testing purposes
    """
    
    def __init__(self):
        self.storage = {}
        self.cid_counter = 0
    
    def add_data(self, data: Dict) -> str:
        """Mock IPFS add operation"""
        self.cid_counter += 1
        cid = f"QmMock{self.cid_counter:06d}"
        self.storage[cid] = data
        logger.info(f"Mock IPFS: Stored data with CID {cid}")
        return cid
    
    def get_data(self, cid: str) -> Optional[Dict]:
        """Mock IPFS get operation"""
        data = self.storage.get(cid)
        if data:
            logger.info(f"Mock IPFS: Retrieved data with CID {cid}")
        else:
            logger.warning(f"Mock IPFS: No data found for CID {cid}")
        return data

def test_secure_client():
    """Test the secure client functionality"""
    print("🔐 Testing Secure Federated Client with REAL IPFS...")
    
    # Create real IPFS client
    ipfs_client = RealIPFSClient()
    
    # Create secure client
    client = SecureFederatedClient("test_client_1", ipfs_client)
    
    # Create mock model parameters
    mock_params = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(3, 10),
        'layer2.bias': torch.randn(3)
    }
    
    # Create secure model update
    secure_update = client.create_secure_model_update(
        model_parameters=mock_params,
        sample_count=100,
        accuracy=0.85,
        loss=0.3,
        round_number=1
    )
    
    if secure_update:
        print(f"✅ Secure model update created successfully")
        print(f"   - Client ID: {secure_update.client_id}")
        print(f"   - IPFS CID: {secure_update.ipfs_cid}")
        print(f"   - Model Hash: {secure_update.model_hash[:16]}...")
        print(f"   - Signature: {secure_update.signature[:16]}...")
        
        # Verify the update
        if client.verify_secure_model_update(secure_update):
            print("✅ Secure model update verification successful")
        else:
            print("❌ Secure model update verification failed")
    else:
        print("❌ Failed to create secure model update")

if __name__ == "__main__":
    test_secure_client()
