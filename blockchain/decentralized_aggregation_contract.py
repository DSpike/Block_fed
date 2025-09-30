"""
Smart Contract for Decentralized Federated Learning Aggregation
Implements FedAVG logic directly on the blockchain
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class ModelParameter:
    """Represents a model parameter"""
    name: str
    shape: List[int]
    data: List[float]  # Flattened parameter data
    dtype: str = "float32"

@dataclass
class ClientModelUpdate:
    """Client model update for aggregation"""
    client_id: str
    round_number: int
    parameters: Dict[str, ModelParameter]
    sample_count: int
    training_loss: float
    validation_accuracy: float
    model_hash: str
    timestamp: float
    ipfs_cid: str

@dataclass
class AggregationTask:
    """Aggregation task on blockchain"""
    round_number: int
    task_id: str
    required_clients: int
    submitted_clients: List[str]
    aggregation_status: str  # "pending", "in_progress", "completed", "failed"
    consensus_threshold: float
    created_timestamp: float
    completed_timestamp: Optional[float] = None
    aggregated_model_hash: Optional[str] = None
    ipfs_cid: Optional[str] = None

class DecentralizedAggregationContract:
    """
    Smart Contract for Decentralized Federated Learning Aggregation
    
    This contract implements:
    1. Client model submission
    2. Decentralized aggregation using FedAVG
    3. Consensus-based aggregation validation
    4. Result distribution
    """
    
    def __init__(self, contract_address: str, web3_client=None):
        """
        Initialize the decentralized aggregation contract
        
        Args:
            contract_address: Smart contract address
            web3_client: Web3 client instance
        """
        self.contract_address = contract_address
        self.web3_client = web3_client
        
        # Contract state (simulating blockchain storage)
        self.client_updates: Dict[int, Dict[str, ClientModelUpdate]] = {}  # round -> client_id -> update
        self.aggregation_tasks: Dict[int, AggregationTask] = {}  # round -> task
        self.consensus_votes: Dict[int, Dict[str, str]] = {}  # round -> client_id -> model_hash
        self.aggregation_results: Dict[int, Dict] = {}  # round -> result
        
        # Contract parameters
        self.min_clients_for_aggregation = 2
        self.consensus_threshold = 0.67
        self.max_round_timeout = 3600  # 1 hour in seconds
        
        logger.info(f"Decentralized Aggregation Contract initialized at {contract_address}")
    
    def create_aggregation_task(self, round_number: int, required_clients: int = None) -> str:
        """
        Create a new aggregation task for a round
        
        Args:
            round_number: Round number
            required_clients: Minimum number of clients required
            
        Returns:
            task_id: Unique task identifier
        """
        if required_clients is None:
            required_clients = self.min_clients_for_aggregation
        
        task_id = f"aggregation_task_{round_number}_{int(time.time())}"
        
        task = AggregationTask(
            round_number=round_number,
            task_id=task_id,
            required_clients=required_clients,
            submitted_clients=[],
            aggregation_status="pending",
            consensus_threshold=self.consensus_threshold,
            created_timestamp=time.time()
        )
        
        self.aggregation_tasks[round_number] = task
        self.client_updates[round_number] = {}
        self.consensus_votes[round_number] = {}
        
        logger.info(f"Created aggregation task {task_id} for round {round_number}")
        return task_id
    
    def submit_client_update(self, client_update: ClientModelUpdate) -> bool:
        """
        Submit client model update for aggregation
        
        Args:
            client_update: Client model update
            
        Returns:
            success: Whether submission was successful
        """
        try:
            round_number = client_update.round_number
            
            # Check if aggregation task exists
            if round_number not in self.aggregation_tasks:
                logger.error(f"No aggregation task found for round {round_number}")
                return False
            
            # Check if client already submitted
            if client_update.client_id in self.client_updates[round_number]:
                logger.warning(f"Client {client_update.client_id} already submitted for round {round_number}")
                return False
            
            # Validate model hash
            computed_hash = self._compute_model_hash(client_update.parameters)
            if computed_hash != client_update.model_hash:
                logger.error(f"Model hash mismatch for client {client_update.client_id}")
                return False
            
            # Store client update
            self.client_updates[round_number][client_update.client_id] = client_update
            self.aggregation_tasks[round_number].submitted_clients.append(client_update.client_id)
            
            logger.info(f"Client {client_update.client_id} submitted update for round {round_number}")
            
            # Check if we can start aggregation
            self._check_aggregation_readiness(round_number)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit client update: {str(e)}")
            return False
    
    def submit_consensus_vote(self, round_number: int, client_id: str, model_hash: str) -> bool:
        """
        Submit consensus vote for aggregation
        
        Args:
            round_number: Round number
            client_id: Client identifier
            model_hash: Model hash being voted for
            
        Returns:
            success: Whether vote was submitted
        """
        try:
            if round_number not in self.consensus_votes:
                logger.error(f"No consensus voting for round {round_number}")
                return False
            
            # Check if client has submitted model update
            if client_id not in self.client_updates[round_number]:
                logger.error(f"Client {client_id} has not submitted model update for round {round_number}")
                return False
            
            # Store vote
            self.consensus_votes[round_number][client_id] = model_hash
            
            logger.info(f"Client {client_id} voted for model {model_hash} in round {round_number}")
            
            # Check consensus
            self._check_consensus(round_number)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit consensus vote: {str(e)}")
            return False
    
    def perform_decentralized_aggregation(self, round_number: int) -> Optional[Dict]:
        """
        Perform decentralized aggregation using FedAVG algorithm
        
        Args:
            round_number: Round number
            
        Returns:
            aggregation_result: Aggregation result
        """
        try:
            if round_number not in self.client_updates:
                logger.error(f"No client updates for round {round_number}")
                return None
            
            client_updates = self.client_updates[round_number]
            if len(client_updates) < self.min_clients_for_aggregation:
                logger.error(f"Insufficient clients for aggregation: {len(client_updates)}")
                return None
            
            logger.info(f"Performing decentralized aggregation for round {round_number} with {len(client_updates)} clients")
            
            # Calculate total samples
            total_samples = sum(update.sample_count for update in client_updates.values())
            
            # Initialize aggregated parameters
            aggregated_params = {}
            param_names = set()
            
            # Collect all parameter names
            for update in client_updates.values():
                param_names.update(update.parameters.keys())
            
            # Perform FedAVG aggregation
            for param_name in param_names:
                aggregated_param = None
                
                for client_id, update in client_updates.items():
                    if param_name in update.parameters:
                        param = update.parameters[param_name]
                        
                        # Calculate weight based on sample count
                        weight = update.sample_count / total_samples
                        
                        # Convert parameter data to tensor-like format
                        param_data = param.data
                        
                        if aggregated_param is None:
                            # Initialize with first parameter
                            aggregated_param = [x * weight for x in param_data]
                        else:
                            # Add weighted parameter
                            for i, value in enumerate(param_data):
                                aggregated_param[i] += value * weight
                
                if aggregated_param is not None:
                    aggregated_params[param_name] = ModelParameter(
                        name=param_name,
                        shape=update.parameters[param_name].shape,
                        data=aggregated_param,
                        dtype=update.parameters[param_name].dtype
                    )
            
            # Compute aggregated model hash
            aggregated_hash = self._compute_model_hash(aggregated_params)
            
            # Create aggregation result
            aggregation_result = {
                'round_number': round_number,
                'aggregated_parameters': aggregated_params,
                'model_hash': aggregated_hash,
                'total_samples': total_samples,
                'num_clients': len(client_updates),
                'aggregation_time': time.time(),
                'client_weights': {
                    client_id: update.sample_count / total_samples 
                    for client_id, update in client_updates.items()
                }
            }
            
            # Store result
            self.aggregation_results[round_number] = aggregation_result
            
            # Update task status
            if round_number in self.aggregation_tasks:
                self.aggregation_tasks[round_number].aggregation_status = "completed"
                self.aggregation_tasks[round_number].completed_timestamp = time.time()
                self.aggregation_tasks[round_number].aggregated_model_hash = aggregated_hash
            
            logger.info(f"✅ Decentralized aggregation completed for round {round_number}")
            return aggregation_result
            
        except Exception as e:
            logger.error(f"Failed to perform decentralized aggregation: {str(e)}")
            if round_number in self.aggregation_tasks:
                self.aggregation_tasks[round_number].aggregation_status = "failed"
            return None
    
    def get_aggregation_result(self, round_number: int) -> Optional[Dict]:
        """
        Get aggregation result for a round
        
        Args:
            round_number: Round number
            
        Returns:
            result: Aggregation result
        """
        return self.aggregation_results.get(round_number)
    
    def get_aggregation_status(self, round_number: int) -> Dict:
        """
        Get aggregation status for a round
        
        Args:
            round_number: Round number
            
        Returns:
            status: Aggregation status
        """
        if round_number not in self.aggregation_tasks:
            return {
                'round_number': round_number,
                'status': 'not_found',
                'message': 'No aggregation task found for this round'
            }
        
        task = self.aggregation_tasks[round_number]
        client_updates = self.client_updates.get(round_number, {})
        
        return {
            'round_number': round_number,
            'task_id': task.task_id,
            'status': task.aggregation_status,
            'required_clients': task.required_clients,
            'submitted_clients': len(client_updates),
            'client_list': list(client_updates.keys()),
            'consensus_threshold': task.consensus_threshold,
            'created_timestamp': task.created_timestamp,
            'completed_timestamp': task.completed_timestamp,
            'aggregated_model_hash': task.aggregated_model_hash
        }
    
    def _check_aggregation_readiness(self, round_number: int):
        """
        Check if aggregation can be started
        
        Args:
            round_number: Round number
        """
        task = self.aggregation_tasks[round_number]
        submitted_count = len(self.client_updates[round_number])
        
        if submitted_count >= task.required_clients:
            if task.aggregation_status == "pending":
                task.aggregation_status = "ready"
                logger.info(f"Aggregation ready for round {round_number}: {submitted_count}/{task.required_clients} clients")
    
    def _check_consensus(self, round_number: int):
        """
        Check if consensus has been achieved
        
        Args:
            round_number: Round number
        """
        if round_number not in self.consensus_votes:
            return
        
        votes = self.consensus_votes[round_number]
        if not votes:
            return
        
        # Count votes by model hash
        model_votes = {}
        for client_id, model_hash in votes.items():
            if model_hash not in model_votes:
                model_votes[model_hash] = []
            model_votes[model_hash].append(client_id)
        
        # Check if any model has consensus
        total_votes = len(votes)
        for model_hash, voters in model_votes.items():
            consensus_ratio = len(voters) / total_votes
            if consensus_ratio >= self.consensus_threshold:
                logger.info(f"✅ Consensus achieved for model {model_hash} in round {round_number} "
                           f"({len(voters)}/{total_votes} votes, {consensus_ratio:.2%})")
                
                # Trigger aggregation
                self.perform_decentralized_aggregation(round_number)
                break
    
    def _compute_model_hash(self, parameters: Dict[str, ModelParameter]) -> str:
        """
        Compute hash of model parameters
        
        Args:
            parameters: Model parameters
            
        Returns:
            model_hash: SHA256 hash of parameters
        """
        try:
            # Sort parameters by name for consistent hashing
            sorted_params = sorted(parameters.items())
            
            # Create hashable representation
            param_data = []
            for name, param in sorted_params:
                param_data.append(f"{name}:{param.shape}:{param.dtype}:{param.data}")
            
            # Compute hash
            param_string = "|".join(param_data)
            model_hash = hashlib.sha256(param_string.encode()).hexdigest()
            
            return model_hash
            
        except Exception as e:
            logger.error(f"Failed to compute model hash: {str(e)}")
            return ""
    
    def cleanup_round(self, round_number: int):
        """
        Cleanup data for a completed round
        
        Args:
            round_number: Round number
        """
        try:
            # Keep only recent rounds (last 10)
            if round_number in self.client_updates:
                del self.client_updates[round_number]
            
            if round_number in self.aggregation_tasks:
                del self.aggregation_tasks[round_number]
            
            if round_number in self.consensus_votes:
                del self.consensus_votes[round_number]
            
            # Keep aggregation results for longer
            # They will be cleaned up separately
            
            logger.info(f"Cleaned up data for round {round_number}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup round {round_number}: {str(e)}")

# Enhanced Smart Contract ABI with Decentralized Aggregation
DECENTRALIZED_AGGREGATION_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
            {"internalType": "uint256", "name": "requiredClients", "type": "uint256"}
        ],
        "name": "createAggregationTask",
        "outputs": [
            {"internalType": "string", "name": "taskId", "type": "string"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "clientId", "type": "string"},
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
            {"internalType": "bytes", "name": "modelParameters", "type": "bytes"},
            {"internalType": "uint256", "name": "sampleCount", "type": "uint256"},
            {"internalType": "uint256", "name": "trainingLoss", "type": "uint256"},
            {"internalType": "uint256", "name": "validationAccuracy", "type": "uint256"},
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
            {"internalType": "string", "name": "ipfsCid", "type": "string"}
        ],
        "name": "submitClientUpdate",
        "outputs": [
            {"internalType": "bool", "name": "success", "type": "bool"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
            {"internalType": "string", "name": "clientId", "type": "string"},
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
        ],
        "name": "submitConsensusVote",
        "outputs": [
            {"internalType": "bool", "name": "success", "type": "bool"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "performDecentralizedAggregation",
        "outputs": [
            {"internalType": "bytes32", "name": "aggregatedHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "numClients", "type": "uint256"},
            {"internalType": "uint256", "name": "totalSamples", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "getAggregationResult",
        "outputs": [
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "numClients", "type": "uint256"},
            {"internalType": "uint256", "name": "totalSamples", "type": "uint256"},
            {"internalType": "uint256", "name": "aggregationTime", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "getAggregationStatus",
        "outputs": [
            {"internalType": "string", "name": "status", "type": "string"},
            {"internalType": "uint256", "name": "requiredClients", "type": "uint256"},
            {"internalType": "uint256", "name": "submittedClients", "type": "uint256"},
            {"internalType": "uint256", "name": "consensusThreshold", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Decentralized Aggregation Contract")
    
    # Create contract instance
    contract = DecentralizedAggregationContract("0x1234567890abcdef")
    
    # Create aggregation task
    round_number = 1
    task_id = contract.create_aggregation_task(round_number, required_clients=3)
    logger.info(f"Created task: {task_id}")
    
    # Simulate client submissions
    for i in range(3):
        client_id = f"client_{i+1}"
        
        # Create mock parameters
        parameters = {
            "layer1.weight": ModelParameter(
                name="layer1.weight",
                shape=[64, 25],
                data=[0.1] * (64 * 25),
                dtype="float32"
            ),
            "layer1.bias": ModelParameter(
                name="layer1.bias",
                shape=[64],
                data=[0.01] * 64,
                dtype="float32"
            )
        }
        
        client_update = ClientModelUpdate(
            client_id=client_id,
            round_number=round_number,
            parameters=parameters,
            sample_count=1000 + i * 500,
            training_loss=0.5 - i * 0.1,
            validation_accuracy=0.8 + i * 0.05,
            model_hash=contract._compute_model_hash(parameters),
            timestamp=time.time(),
            ipfs_cid=f"ipfs_cid_{client_id}_{round_number}"
        )
        
        # Submit update
        success = contract.submit_client_update(client_update)
        logger.info(f"Client {client_id} submission: {'✅' if success else '❌'}")
        
        # Submit consensus vote
        vote_success = contract.submit_consensus_vote(
            round_number, client_id, client_update.model_hash
        )
        logger.info(f"Client {client_id} vote: {'✅' if vote_success else '❌'}")
    
    # Check aggregation status
    status = contract.get_aggregation_status(round_number)
    logger.info(f"Aggregation status: {json.dumps(status, indent=2)}")
    
    # Get aggregation result
    result = contract.get_aggregation_result(round_number)
    if result:
        logger.info(f"✅ Aggregation result: {result['model_hash']}")
        logger.info(f"   Clients: {result['num_clients']}")
        logger.info(f"   Total samples: {result['total_samples']}")
    else:
        logger.warning("❌ No aggregation result found")
    
    logger.info("✅ Decentralized aggregation contract test completed")
