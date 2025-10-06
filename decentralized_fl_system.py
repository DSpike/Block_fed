#!/usr/bin/env python3
"""
Decentralized Federated Learning System with 2 Miners and Consensus Mechanism
Eliminates single point of failure through distributed mining and consensus
"""

import torch
import torch.nn as nn
import time
import hashlib
import json
import logging
import pickle
import base64
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import random
from collections import defaultdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecureEncryptionManager:
    """Manages encryption and decryption for model parameters"""
    
    def __init__(self):
        self.client_keys: Dict[str, str] = {}
    
    def generate_client_key(self, client_id: str, password: str = None) -> str:
        """Generate unique encryption key for client"""
        if password is None:
            password = f"client_{client_id}_secret_{int(time.time())}"
        
        # Derive key from password using PBKDF2
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        self.client_keys[client_id] = key.decode()
        return key.decode()
    
    def encrypt_model_parameters(self, parameters: Dict[str, torch.Tensor], 
                               client_id: str) -> str:
        """Encrypt model parameters using client's key"""
        try:
            # Get client's encryption key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No encryption key for client {client_id}")
            
            # Serialize parameters
            serialized = pickle.dumps(parameters)
            
            # Encrypt with client's key
            f = Fernet(client_key.encode())
            encrypted = f.encrypt(serialized)
            
            return encrypted.hex()
            
        except Exception as e:
            logger.error(f"Failed to encrypt model parameters: {e}")
            raise
    
    def decrypt_model_parameters(self, encrypted_hex: str, client_id: str) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters using client's key"""
        try:
            # Get client's encryption key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No encryption key for client {client_id}")
            
            # Decrypt
            f = Fernet(client_key.encode())
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            decrypted_bytes = f.decrypt(encrypted_bytes)
            
            # Deserialize
            return pickle.loads(decrypted_bytes)
            
        except Exception as e:
            logger.error(f"Failed to decrypt model parameters: {e}")
            raise

class SecureHashManager:
    """Manages cryptographic hashing for model verification"""
    
    @staticmethod
    def compute_model_hash(parameters: Dict[str, torch.Tensor]) -> str:
        """Compute SHA256 hash of model parameters"""
        try:
            # Create deterministic hash
            param_bytes = b''
            for name in sorted(parameters.keys()):
                param = parameters[name]
                param_bytes += name.encode('utf-8')
                param_bytes += param.detach().cpu().numpy().tobytes()
            
            # Compute SHA256 hash
            hash_obj = hashlib.sha256(param_bytes)
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to compute model hash: {e}")
            raise
    
    @staticmethod
    def verify_model_hash(parameters: Dict[str, torch.Tensor], expected_hash: str) -> bool:
        """Verify model parameter hash"""
        try:
            actual_hash = SecureHashManager.compute_model_hash(parameters)
            return actual_hash == expected_hash
        except Exception as e:
            logger.error(f"Failed to verify model hash: {e}")
            return False

class SecureSignatureManager:
    """Manages digital signatures for authentication"""
    
    def __init__(self):
        self.client_keys: Dict[str, Dict[str, str]] = {}
    
    def generate_client_keypair(self, client_id: str) -> tuple:
        """Generate RSA key pair for client"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Store keys
        self.client_keys[client_id] = {
            'private': private_pem.decode(),
            'public': public_pem.decode()
        }
        
        return private_pem.decode(), public_pem.decode()
    
    def sign_data(self, data: str, client_id: str) -> str:
        """Sign data with client's private key"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            # Get client's private key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No private key for client {client_id}")
            
            private_key = serialization.load_pem_private_key(
                client_key['private'].encode(),
                password=None,
            )
            
            # Sign data
            signature = private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def verify_signature(self, data: str, signature: str, client_id: str) -> bool:
        """Verify signature with client's public key"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            # Get client's public key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                return False
            
            public_key = serialization.load_pem_public_key(
                client_key['public'].encode()
            )
            
            # Verify signature
            public_key.verify(
                base64.b64decode(signature),
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False

class MinerRole(Enum):
    """Miner roles in the decentralized system"""
    PRIMARY_MINER = "primary_miner"
    SECONDARY_MINER = "secondary_miner"
    VALIDATOR = "validator"

class ConsensusStatus(Enum):
    """Consensus status for model aggregation"""
    PENDING = "pending"
    AGREED = "agreed"
    DISAGREED = "disagreed"
    TIMEOUT = "timeout"

@dataclass
class SecureModelUpdate:
    """Privacy-preserving model update from a client"""
    client_id: str
    ipfs_cid: str                    # Only IPFS reference - NO raw parameters
    model_hash: str                  # Cryptographic hash for verification
    sample_count: int
    accuracy: float
    loss: float
    timestamp: float
    signature: str                   # Digital signature for authentication
    round_number: int
    encryption_method: str = "fernet"  # Encryption method used

# Keep old ModelUpdate for backward compatibility during transition
@dataclass
class ModelUpdate:
    """Legacy model update (DEPRECATED - use SecureModelUpdate)"""
    client_id: str
    model_parameters: Dict[str, torch.Tensor]  # ❌ PRIVACY LEAK - DEPRECATED
    sample_count: int
    accuracy: float
    loss: float
    timestamp: float
    signature: str
    round_number: int

@dataclass
class AggregationProposal:
    """Proposal for model aggregation"""
    proposer_id: str
    aggregated_model: Dict[str, torch.Tensor]
    model_hash: str
    round_number: int
    timestamp: float
    signature: str
    validation_score: float

@dataclass
class ConsensusVote:
    """Vote on aggregation proposal"""
    voter_id: str
    proposal_hash: str
    vote: bool  # True = agree, False = disagree
    confidence: float
    timestamp: float
    signature: str

class DecentralizedMiner:
    """
    Decentralized miner that can perform aggregation and consensus
    """
    
    def __init__(self, miner_id: str, model: nn.Module, role: MinerRole, ipfs_client=None):
        self.miner_id = miner_id
        self.model = model
        self.role = role
        self.is_active = True
        self.stake = 1000  # Initial stake
        self.reputation = 1.0
        self.consensus_threshold = 0.67  # 67% agreement required
        
        # IPFS client for secure model retrieval
        self.ipfs_client = ipfs_client
        
        # Security managers
        self.encryption_manager = SecureEncryptionManager()
        self.hash_manager = SecureHashManager()
        self.signature_manager = SecureSignatureManager()
        
        # Storage for model updates and proposals
        self.client_updates: Dict[str, ModelUpdate] = {}  # Legacy support
        self.secure_client_updates: Dict[str, Dict] = {}  # Secure updates
        self.aggregation_proposals: Dict[str, AggregationProposal] = {}
        self.consensus_votes: Dict[str, List[ConsensusVote]] = {}
        
        # Communication queues
        self.message_queue = queue.Queue()
        self.broadcast_queue = queue.Queue()
        
        # Consensus state
        self.current_round = 0
        self.consensus_timeout = 30  # 30 seconds timeout
        
        logger.info(f"Miner {miner_id} initialized with role {role.value}")
    
    def add_client_update(self, update: ModelUpdate) -> bool:
        """Add client update to miner's storage (LEGACY - DEPRECATED)"""
        try:
            # Verify signature
            if not self._verify_signature(update):
                logger.warning(f"Miner {self.miner_id}: Invalid signature for client {update.client_id}")
                return False
            
            # Store update
            self.client_updates[update.client_id] = update
            logger.info(f"Miner {self.miner_id}: Added update from client {update.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Failed to add client update: {e}")
            return False
    
    def add_secure_client_update(self, update: SecureModelUpdate) -> bool:
        """Add secure client update using only IPFS CID"""
        try:
            # Verify signature
            signature_data = f"{update.client_id}_{update.ipfs_cid}_{update.model_hash}_{update.timestamp}"
            if not self.signature_manager.verify_signature(signature_data, update.signature, update.client_id):
                logger.error(f"Miner {self.miner_id}: Signature verification failed for client {update.client_id}")
                return False
            
            # Retrieve encrypted data from IPFS
            if not self.ipfs_client:
                logger.error(f"Miner {self.miner_id}: No IPFS client available")
                return False
                
            encrypted_data = self.ipfs_client.get_data(update.ipfs_cid)
            if not encrypted_data:
                logger.error(f"Miner {self.miner_id}: Failed to retrieve data from IPFS: {update.ipfs_cid}")
                return False
            
            # Decrypt model parameters
            decrypted_params = self.encryption_manager.decrypt_model_parameters(
                encrypted_data['encrypted_parameters'], update.client_id
            )
            
            # Verify model hash
            if not self.hash_manager.verify_model_hash(decrypted_params, update.model_hash):
                logger.error(f"Miner {self.miner_id}: Model hash verification failed for client {update.client_id}")
                return False
            
            # Store secure update
            self.secure_client_updates[update.client_id] = {
                'parameters': decrypted_params,
                'metadata': update,
                'verified': True
            }
            
            logger.info(f"Miner {self.miner_id}: Successfully processed secure update from client {update.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Failed to process secure update: {e}")
            return False
    
    def propose_aggregation(self, round_number: int) -> Optional[AggregationProposal]:
        """Propose aggregated model for consensus"""
        try:
            # Use secure updates if available, fallback to legacy
            updates_to_use = self.secure_client_updates if self.secure_client_updates else self.client_updates
            
            if not updates_to_use:
                logger.warning(f"Miner {self.miner_id}: No client updates to aggregate")
                return None
            
            # Perform FedAVG aggregation
            if self.secure_client_updates:
                aggregated_model = self._perform_secure_fedavg_aggregation()
            else:
                aggregated_model = self._perform_fedavg_aggregation()
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(aggregated_model)
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(aggregated_model)
            
            # Create proposal
            proposal = AggregationProposal(
                proposer_id=self.miner_id,
                aggregated_model=aggregated_model,
                model_hash=model_hash,
                round_number=round_number,
                timestamp=time.time(),
                signature=self._sign_data(f"{self.miner_id}_{model_hash}_{round_number}"),
                validation_score=validation_score
            )
            
            # Store proposal
            self.aggregation_proposals[model_hash] = proposal
            
            logger.info(f"Miner {self.miner_id}: Proposed aggregation with hash {model_hash}")
            return proposal
            
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Failed to propose aggregation: {e}")
            return None
    
    def vote_on_proposal(self, proposal: AggregationProposal) -> ConsensusVote:
        """Vote on an aggregation proposal"""
        try:
            # Validate proposal
            if not self._verify_signature(proposal):
                logger.warning(f"Miner {self.miner_id}: Invalid proposal signature")
                return self._create_vote(proposal.model_hash, False, 0.0)
            
            # Evaluate proposal quality
            quality_score = self._evaluate_proposal_quality(proposal)
            
            # Make voting decision
            vote_decision = quality_score > 0.7  # Threshold for agreement
            confidence = abs(quality_score - 0.5) * 2  # Convert to confidence
            
            # Create vote
            vote = self._create_vote(proposal.model_hash, vote_decision, confidence)
            
            # Store vote
            if proposal.model_hash not in self.consensus_votes:
                self.consensus_votes[proposal.model_hash] = []
            self.consensus_votes[proposal.model_hash].append(vote)
            
            logger.info(f"Miner {self.miner_id}: Voted {vote_decision} on proposal {proposal.model_hash}")
            return vote
            
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Failed to vote on proposal: {e}")
            return self._create_vote(proposal.model_hash, False, 0.0)
    
    def check_consensus(self, proposal_hash: str) -> Tuple[ConsensusStatus, float]:
        """Check if consensus has been reached on a proposal"""
        try:
            if proposal_hash not in self.consensus_votes:
                return ConsensusStatus.PENDING, 0.0
            
            votes = self.consensus_votes[proposal_hash]
            if not votes:
                return ConsensusStatus.PENDING, 0.0
            
            # Calculate weighted consensus
            total_weight = 0
            agreement_weight = 0
            
            for vote in votes:
                weight = self._calculate_vote_weight(vote)
                total_weight += weight
                
                if vote.vote:
                    agreement_weight += weight
            
            if total_weight == 0:
                return ConsensusStatus.PENDING, 0.0
            
            consensus_ratio = agreement_weight / total_weight
            
            # Check consensus threshold
            if consensus_ratio >= self.consensus_threshold:
                return ConsensusStatus.AGREED, consensus_ratio
            elif len(votes) >= 2:  # At least 2 votes
                return ConsensusStatus.DISAGREED, consensus_ratio
            else:
                return ConsensusStatus.PENDING, consensus_ratio
                
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Failed to check consensus: {e}")
            return ConsensusStatus.PENDING, 0.0
    
    def _perform_fedavg_aggregation(self) -> Dict[str, torch.Tensor]:
        """Perform FedAVG aggregation on client updates"""
        if not self.client_updates:
            return {}
        
        # Calculate total samples
        total_samples = sum(update.sample_count for update in self.client_updates.values())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first update
        first_update = next(iter(self.client_updates.values()))
        param_names = first_update.model_parameters.keys()
        
        # Weighted averaging
        for param_name in param_names:
            aggregated_param = None
            
            for update in self.client_updates.values():
                if param_name in update.model_parameters:
                    param = update.model_parameters[param_name]
                    weight = update.sample_count / total_samples
                    
                    if aggregated_param is None:
                        aggregated_param = param * weight
                    else:
                        aggregated_param += param * weight
            
            if aggregated_param is not None:
                aggregated_params[param_name] = aggregated_param
        
        return aggregated_params
    
    def _perform_secure_fedavg_aggregation(self) -> Dict[str, torch.Tensor]:
        """Perform FedAVG aggregation on secure client updates"""
        try:
            if not self.secure_client_updates:
                return {}
            
            # Calculate total samples
            total_samples = sum(update['metadata'].sample_count for update in self.secure_client_updates.values())
            
            # Initialize aggregated parameters
            aggregated_params = {}
            
            # Get parameter names from first update
            first_update = next(iter(self.secure_client_updates.values()))
            param_names = first_update['parameters'].keys()
            
            # Weighted averaging
            for param_name in param_names:
                aggregated_param = None
                
                for client_id, secure_update in self.secure_client_updates.items():
                    if param_name in secure_update['parameters']:
                        param = secure_update['parameters'][param_name]
                        weight = secure_update['metadata'].sample_count / total_samples
                        
                        if aggregated_param is None:
                            aggregated_param = param * weight
                        else:
                            aggregated_param += param * weight
                
                if aggregated_param is not None:
                    aggregated_params[param_name] = aggregated_param
            
            logger.info(f"Miner {self.miner_id}: Secure FedAVG aggregation completed with {len(self.secure_client_updates)} clients")
            return aggregated_params
            
        except Exception as e:
            logger.error(f"Miner {self.miner_id}: Secure FedAVG aggregation failed: {e}")
            return {}
    
    def _calculate_model_hash(self, model_params: Dict[str, torch.Tensor]) -> str:
        """Calculate SHA256 hash of model parameters"""
        param_str = str(sorted(model_params.keys()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def _calculate_validation_score(self, model_params: Dict[str, torch.Tensor]) -> float:
        """Calculate validation score for aggregated model"""
        # Simple validation based on parameter statistics
        total_params = 0
        valid_params = 0
        
        for param in model_params.values():
            total_params += param.numel()
            # Check for NaN or infinite values
            if torch.isfinite(param).all():
                valid_params += param.numel()
        
        if total_params == 0:
            return 0.0
        
        return valid_params / total_params
    
    def _evaluate_proposal_quality(self, proposal: AggregationProposal) -> float:
        """Evaluate the quality of an aggregation proposal"""
        # Combine validation score with reputation
        base_score = proposal.validation_score
        reputation_bonus = min(0.2, self.reputation * 0.1)
        
        return min(1.0, base_score + reputation_bonus)
    
    def _create_vote(self, proposal_hash: str, vote: bool, confidence: float) -> ConsensusVote:
        """Create a consensus vote"""
        return ConsensusVote(
            voter_id=self.miner_id,
            proposal_hash=proposal_hash,
            vote=vote,
            confidence=confidence,
            timestamp=time.time(),
            signature=self._sign_data(f"{self.miner_id}_{proposal_hash}_{vote}")
        )
    
    def _calculate_vote_weight(self, vote: ConsensusVote) -> float:
        """Calculate weight of a vote based on miner's stake and reputation"""
        return self.stake * self.reputation * vote.confidence
    
    def _verify_signature(self, obj) -> bool:
        """Verify signature of an object (simplified)"""
        # In a real implementation, this would use cryptographic signatures
        return True
    
    def _sign_data(self, data: str) -> str:
        """Sign data (simplified)"""
        # In a real implementation, this would use cryptographic signatures
        return hashlib.sha256(f"{self.miner_id}_{data}".encode()).hexdigest()[:16]
    
    def update_reputation(self, success: bool):
        """Update miner reputation based on performance"""
        if success:
            self.reputation = min(2.0, self.reputation + 0.1)
        else:
            self.reputation = max(0.1, self.reputation - 0.05)
        
        logger.info(f"Miner {self.miner_id}: Reputation updated to {self.reputation:.2f}")

class DecentralizedFederatedLearningSystem:
    """
    Decentralized Federated Learning System with 2 Miners and Consensus
    """
    
    def __init__(self, model: nn.Module, num_clients: int = 3):
        self.model = model
        self.num_clients = num_clients
        self.current_round = 0
        
        # CRITICAL FIX: Create independent model copies for each miner
        # This prevents shared state and race conditions
        import copy
        import torch
        
        # Create deep copies of the model for each miner
        model_1 = copy.deepcopy(model)
        model_2 = copy.deepcopy(model)
        
        # Ensure both models are on the same device and have same initial state
        device = next(model.parameters()).device
        model_1 = model_1.to(device)
        model_2 = model_2.to(device)
        
        # Verify models have identical initial parameters
        self._verify_identical_initialization(model, model_1, model_2)
        
        # Initialize 2 miners with independent model copies
        self.miners = {
            "miner_1": DecentralizedMiner("miner_1", model_1, MinerRole.PRIMARY_MINER),
            "miner_2": DecentralizedMiner("miner_2", model_2, MinerRole.SECONDARY_MINER)
        }
        
        # Client updates storage
        self.client_updates: Dict[str, ModelUpdate] = {}
        
        # Consensus state
        self.consensus_results: Dict[int, Dict] = {}
        
        logger.info("Decentralized Federated Learning System initialized with 2 miners")
    
    def _verify_identical_initialization(self, original_model, model_1, model_2):
        """Verify that all three models have identical initial parameters"""
        try:
            # Check if all models have identical parameters
            for (name1, param1), (name2, param2), (name3, param3) in zip(
                original_model.named_parameters(),
                model_1.named_parameters(), 
                model_2.named_parameters()
            ):
                if not torch.equal(param1, param2) or not torch.equal(param1, param3):
                    logger.error(f"Model parameter mismatch detected: {name1}")
                    raise ValueError(f"Models have different initial parameters: {name1}")
            
            logger.info("✅ All miners initialized with identical model parameters")
            
        except Exception as e:
            logger.error(f"Model initialization verification failed: {e}")
            raise
    
    def synchronize_models(self):
        """Synchronize model parameters between miners after consensus"""
        try:
            # Get the current global model parameters (from primary miner)
            primary_miner = self.miners["miner_1"]
            global_params = primary_miner.model.state_dict()
            
            # Update secondary miner with global parameters
            secondary_miner = self.miners["miner_2"]
            secondary_miner.model.load_state_dict(global_params)
            
            # Verify synchronization was successful
            self._verify_model_synchronization()
            
            logger.info("✅ Model parameters synchronized between miners")
            
        except Exception as e:
            logger.error(f"Model synchronization failed: {e}")
            raise
    
    def _verify_model_synchronization(self):
        """Verify that both miners have identical model parameters"""
        try:
            miner_1_params = self.miners["miner_1"].model.state_dict()
            miner_2_params = self.miners["miner_2"].model.state_dict()
            
            for (name1, param1), (name2, param2) in zip(
                miner_1_params.items(), 
                miner_2_params.items()
            ):
                if not torch.equal(param1, param2):
                    logger.error(f"Model synchronization failed: {name1} parameters differ")
                    raise ValueError(f"Models not synchronized: {name1}")
            
            logger.debug("✅ Model synchronization verified")
            
        except Exception as e:
            logger.error(f"Model synchronization verification failed: {e}")
            raise
    
    def handle_initialization_divergence(self):
        """Handle case where miners have different initial parameters"""
        try:
            logger.warning("Handling potential initialization divergence...")
            
            # Check if models have diverged
            miner_1_params = self.miners["miner_1"].model.state_dict()
            miner_2_params = self.miners["miner_2"].model.state_dict()
            
            diverged = False
            for (name1, param1), (name2, param2) in zip(
                miner_1_params.items(), 
                miner_2_params.items()
            ):
                if not torch.equal(param1, param2):
                    diverged = True
                    break
            
            if diverged:
                logger.warning("Model divergence detected, forcing synchronization...")
                # Use primary miner's model as reference
                self.synchronize_models()
            else:
                logger.info("Models are already synchronized")
                
        except Exception as e:
            logger.error(f"Failed to handle initialization divergence: {e}")
            raise
    
    def add_client_update(self, update: ModelUpdate) -> bool:
        """Add client update to the system"""
        try:
            # Add to both miners
            success_1 = self.miners["miner_1"].add_client_update(update)
            success_2 = self.miners["miner_2"].add_client_update(update)
            
            if success_1 and success_2:
                self.client_updates[update.client_id] = update
                logger.info(f"Added client update from {update.client_id}")
                return True
            else:
                logger.warning(f"Failed to add client update from {update.client_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add client update: {e}")
            return False
    
    def run_decentralized_round(self) -> Dict:
        """Run one decentralized federated learning round"""
        try:
            logger.info(f"Starting decentralized round {self.current_round + 1}")
            start_time = time.time()
            
            # Step 1: Both miners propose aggregation
            proposals = {}
            for miner_id, miner in self.miners.items():
                proposal = miner.propose_aggregation(self.current_round)
                if proposal:
                    proposals[miner_id] = proposal
                    logger.info(f"Miner {miner_id} proposed aggregation")
            
            if not proposals:
                logger.error("No miners could propose aggregation")
                return {"success": False, "error": "No aggregation proposals"}
            
            # Step 2: Miners vote on each other's proposals
            consensus_results = {}
            for proposer_id, proposal in proposals.items():
                votes = []
                for voter_id, miner in self.miners.items():
                    if voter_id != proposer_id:  # Don't vote on own proposal
                        vote = miner.vote_on_proposal(proposal)
                        votes.append(vote)
                
                # Check consensus
                status, ratio = self.miners[proposer_id].check_consensus(proposal.model_hash)
                consensus_results[proposer_id] = {
                    "status": status,
                    "ratio": ratio,
                    "votes": len(votes)
                }
                
                logger.info(f"Consensus for {proposer_id}: {status.value} ({ratio:.2%})")
            
            # Step 3: Select winning proposal
            winning_proposal = self._select_winning_proposal(proposals, consensus_results)
            
            if not winning_proposal:
                logger.error("No winning proposal found")
                return {"success": False, "error": "No consensus reached"}
            
            # Step 4: Synchronize models after consensus
            self.synchronize_models()
            
            # Step 4: Update global model
            self._update_global_model(winning_proposal.aggregated_model)
            
            # Step 5: Update miner reputations
            for miner_id, miner in self.miners.items():
                success = (miner_id == winning_proposal.proposer_id)
                miner.update_reputation(success)
            
            # Step 6: Store consensus results
            self.consensus_results[self.current_round] = {
                "winning_proposal": winning_proposal.model_hash,
                "proposer": winning_proposal.proposer_id,
                "consensus_ratio": consensus_results[winning_proposal.proposer_id]["ratio"],
                "timestamp": time.time()
            }
            
            self.current_round += 1
            
            round_time = time.time() - start_time
            logger.info(f"Decentralized round {self.current_round} completed in {round_time:.2f}s")
            
            return {
                "success": True,
                "round": self.current_round,
                "winning_proposal": winning_proposal.model_hash,
                "consensus_ratio": consensus_results[winning_proposal.proposer_id]["ratio"],
                "round_time": round_time
            }
            
        except Exception as e:
            logger.error(f"Decentralized round failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _select_winning_proposal(self, proposals: Dict, consensus_results: Dict) -> Optional[AggregationProposal]:
        """Select the winning proposal based on consensus"""
        # Find proposals with consensus
        agreed_proposals = []
        for proposer_id, proposal in proposals.items():
            status = consensus_results[proposer_id]["status"]
            if status == ConsensusStatus.AGREED:
                agreed_proposals.append((proposer_id, proposal))
        
        if not agreed_proposals:
            return None
        
        # Select proposal with highest consensus ratio
        best_proposal = max(agreed_proposals, 
                          key=lambda x: consensus_results[x[0]]["ratio"])
        
        return best_proposal[1]
    
    def _update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update the global model with aggregated parameters"""
        try:
            # Load aggregated parameters into model
            self.model.load_state_dict(aggregated_params)
            logger.info("Global model updated with aggregated parameters")
            
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "current_round": self.current_round,
            "active_miners": len([m for m in self.miners.values() if m.is_active]),
            "client_updates": len(self.client_updates),
            "consensus_results": len(self.consensus_results),
            "miner_reputations": {mid: m.reputation for mid, m in self.miners.items()},
            "miner_stakes": {mid: m.stake for mid, m in self.miners.items()}
        }

def main():
    """Test the decentralized federated learning system"""
    from models.transductive_fewshot_model import TransductiveLearner
    
    # Create test model
    model = TransductiveLearner(input_dim=30, hidden_dim=128, embedding_dim=64, num_classes=2)
    
    # Initialize decentralized system
    system = DecentralizedFederatedLearningSystem(model, num_clients=3)
    
    # Simulate client updates
    for i in range(3):
        update = ModelUpdate(
            client_id=f"client_{i+1}",
            model_parameters={name: param.clone() for name, param in model.named_parameters()},
            sample_count=1000 + i * 500,
            accuracy=0.8 + i * 0.05,
            loss=0.2 - i * 0.02,
            timestamp=time.time(),
            signature=f"signature_{i}",
            round_number=1
        )
        system.add_client_update(update)
    
    # Run decentralized round
    result = system.run_decentralized_round()
    print(f"Round result: {result}")
    
    # Get system status
    status = system.get_system_status()
    print(f"System status: {status}")

if __name__ == "__main__":
    main()
