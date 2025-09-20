"""
Fully Decentralized Blockchain Federated Learning System
Integrates all four components for 100% decentralized operation
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
import json

from blockchain.consensus_mechanisms import ProofOfContributionConsensus, ConsensusVote
from blockchain.decentralized_aggregation_contract import DecentralizedAggregationContract, ClientModelUpdate
from coordinators.decentralized_coordinator import DecentralizedCoordinator, NodeRole, CoordinationState
from communication.p2p_network import P2PNetwork, MessageType, P2PMessage

logger = logging.getLogger(__name__)

class FullyDecentralizedSystem:
    """
    Fully Decentralized Blockchain Federated Learning System
    
    Integrates:
    1. Decentralized Consensus (Proof of Contribution)
    2. Smart Contract Aggregation (FedAVG on blockchain)
    3. Decentralized Coordinator (No central authority)
    4. P2P Communication (Direct client-to-client)
    """
    
    def __init__(self, node_id: str, host: str = "127.0.0.1", port: int = 8000, 
                 role: NodeRole = NodeRole.CLIENT):
        """
        Initialize fully decentralized system
        
        Args:
            node_id: Unique node identifier
            host: Host address
            port: Port number
            role: Node role
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.role = role
        
        # Initialize all four components
        logger.info("Initializing fully decentralized system components...")
        
        # 1. P2P Communication Network
        self.p2p_network = P2PNetwork(node_id, host, port)
        self._setup_p2p_handlers()
        
        # 2. Decentralized Coordinator
        self.coordinator = DecentralizedCoordinator(node_id, f"{host}:{port}", role)
        
        # 3. Consensus Mechanism
        self.consensus = ProofOfContributionConsensus()
        self.consensus.register_client(node_id, initial_stake=100.0)
        
        # 4. Smart Contract Aggregation
        self.aggregation_contract = DecentralizedAggregationContract(f"contract_{node_id}")
        
        # System state
        self.is_running = False
        self.current_round = None
        self.training_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.round_history: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        logger.info(f"✅ Fully decentralized system initialized: {node_id}")
    
    def start_system(self) -> bool:
        """
        Start the fully decentralized system
        
        Returns:
            success: Whether system started successfully
        """
        try:
            logger.info("Starting fully decentralized system...")
            
            # Start P2P network
            if not self.p2p_network.start_network():
                logger.error("Failed to start P2P network")
                return False
            
            # Start decentralized coordinator
            self.coordinator.start_decentralized_coordination()
            
            # Register with coordinator
            self.coordinator.register_node(
                self.node_id, 
                f"{self.host}:{self.port}", 
                self.role, 
                initial_stake=100.0
            )
            
            self.is_running = True
            
            logger.info("✅ Fully decentralized system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {str(e)}")
            return False
    
    def stop_system(self):
        """
        Stop the fully decentralized system
        """
        logger.info("Stopping fully decentralized system...")
        
        self.is_running = False
        
        # Stop coordinator
        self.coordinator.stop_decentralized_coordination()
        
        # Stop P2P network
        self.p2p_network.stop_network()
        
        logger.info("✅ Fully decentralized system stopped")
    
    def join_network(self, bootstrap_nodes: List[tuple]) -> bool:
        """
        Join the decentralized network
        
        Args:
            bootstrap_nodes: List of bootstrap node addresses (host, port)
            
        Returns:
            success: Whether successfully joined network
        """
        try:
            logger.info(f"Joining network with {len(bootstrap_nodes)} bootstrap nodes...")
            
            # Discover peers through P2P network
            discovered_count = self.p2p_network.discover_peers(bootstrap_nodes)
            
            # Register discovered peers with coordinator
            for peer_id, peer_info in self.p2p_network.peers.items():
                self.coordinator.register_node(
                    peer_id,
                    f"{peer_info.address}:{peer_info.port}",
                    NodeRole.CLIENT,  # Assume all are clients initially
                    initial_stake=100.0
                )
            
            logger.info(f"✅ Joined network: discovered {discovered_count} peers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join network: {str(e)}")
            return False
    
    def start_federated_round(self, round_number: int, required_participants: int = None) -> bool:
        """
        Start a federated learning round
        
        Args:
            round_number: Round number
            required_participants: Minimum participants required
            
        Returns:
            success: Whether round started successfully
        """
        try:
            logger.info(f"Starting federated learning round {round_number}...")
            
            # Start coordination round
            success = self.coordinator.start_coordination_round(round_number, required_participants)
            
            if success:
                # Broadcast round start to all peers
                self.p2p_network.broadcast_message(
                    MessageType.ROUND_START,
                    {
                        'round_number': round_number,
                        'required_participants': required_participants,
                        'coordinator_id': self.node_id
                    },
                    round_number=round_number
                )
                
                self.current_round = round_number
                logger.info(f"✅ Started federated learning round {round_number}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start federated round: {str(e)}")
            return False
    
    def submit_model_update(self, model_parameters: Dict, sample_count: int, 
                          training_loss: float, validation_accuracy: float) -> bool:
        """
        Submit model update for current round
        
        Args:
            model_parameters: Model parameters
            sample_count: Number of training samples
            training_loss: Training loss
            validation_accuracy: Validation accuracy
            
        Returns:
            success: Whether submission was successful
        """
        try:
            if self.current_round is None:
                logger.error("No active federated round")
                return False
            
            logger.info(f"Submitting model update for round {self.current_round}...")
            
            # Submit to coordinator
            success = self.coordinator.submit_model_update(
                self.node_id, model_parameters, sample_count, training_loss, validation_accuracy
            )
            
            if success:
                # Broadcast model update to peers
                self.p2p_network.broadcast_message(
                    MessageType.MODEL_UPDATE,
                    {
                        'client_id': self.node_id,
                        'model_parameters': model_parameters,
                        'sample_count': sample_count,
                        'training_loss': training_loss,
                        'validation_accuracy': validation_accuracy,
                        'model_hash': self._compute_model_hash(model_parameters)
                    },
                    round_number=self.current_round
                )
                
                # Update performance metrics
                self.performance_metrics.update({
                    'accuracy': validation_accuracy,
                    'precision': validation_accuracy * 0.95,  # Simulate metrics
                    'recall': validation_accuracy * 0.98,
                    'f1_score': validation_accuracy * 0.96
                })
                
                # Update consensus contribution score
                self.consensus.update_contribution_score(self.node_id, self.performance_metrics)
                
                logger.info(f"✅ Model update submitted for round {self.current_round}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to submit model update: {str(e)}")
            return False
    
    def submit_consensus_vote(self, model_hash: str) -> bool:
        """
        Submit consensus vote for aggregated model
        
        Args:
            model_hash: Model hash to vote for
            
        Returns:
            success: Whether vote was submitted successfully
        """
        try:
            if self.current_round is None:
                logger.error("No active federated round")
                return False
            
            logger.info(f"Submitting consensus vote for model {model_hash}...")
            
            # Submit to coordinator
            success = self.coordinator.submit_consensus_vote(model_hash)
            
            if success:
                # Broadcast consensus vote to peers
                vote_weight = self.consensus.calculate_vote_weight(self.node_id)
                
                self.p2p_network.broadcast_message(
                    MessageType.CONSENSUS_VOTE,
                    {
                        'voter_id': self.node_id,
                        'model_hash': model_hash,
                        'vote_weight': vote_weight,
                        'round_number': self.current_round
                    },
                    round_number=self.current_round
                )
                
                logger.info(f"✅ Consensus vote submitted for model {model_hash}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to submit consensus vote: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status
        
        Returns:
            status: System status information
        """
        # Get status from all components
        p2p_status = self.p2p_network.get_network_status()
        coordinator_status = self.coordinator.get_coordination_status()
        
        # Get consensus status
        consensus_status = {}
        if self.current_round:
            consensus_status = self.consensus.get_consensus_status(self.current_round)
        
        # Get aggregation status
        aggregation_status = {}
        if self.current_round:
            aggregation_status = self.aggregation_contract.get_aggregation_status(self.current_round)
        
        return {
            'system_info': {
                'node_id': self.node_id,
                'role': self.role.value,
                'is_running': self.is_running,
                'current_round': self.current_round,
                'performance_metrics': self.performance_metrics
            },
            'p2p_network': p2p_status,
            'coordinator': coordinator_status,
            'consensus': consensus_status,
            'aggregation': aggregation_status,
            'round_history_count': len(self.round_history)
        }
    
    def _setup_p2p_handlers(self):
        """
        Setup P2P message handlers
        """
        def handle_model_update(message: P2PMessage):
            """Handle model update from peer"""
            try:
                payload = message.payload
                logger.info(f"Received model update from {message.sender_id}")
                
                # Forward to coordinator if needed
                # (In a real system, this would be handled by the coordinator)
                
            except Exception as e:
                logger.error(f"Error handling model update: {str(e)}")
        
        def handle_consensus_vote(message: P2PMessage):
            """Handle consensus vote from peer"""
            try:
                payload = message.payload
                logger.info(f"Received consensus vote from {message.sender_id}")
                
                # Forward to consensus mechanism if needed
                
            except Exception as e:
                logger.error(f"Error handling consensus vote: {str(e)}")
        
        def handle_round_start(message: P2PMessage):
            """Handle round start message"""
            try:
                payload = message.payload
                logger.info(f"Received round start from {message.sender_id}: round {payload['round_number']}")
                
                # Update current round if this is a new round
                round_number = payload['round_number']
                if self.current_round is None or round_number > self.current_round:
                    self.current_round = round_number
                
            except Exception as e:
                logger.error(f"Error handling round start: {str(e)}")
        
        def handle_round_end(message: P2PMessage):
            """Handle round end message"""
            try:
                payload = message.payload
                logger.info(f"Received round end from {message.sender_id}: round {payload['round_number']}")
                
                # Process round results if needed
                
            except Exception as e:
                logger.error(f"Error handling round end: {str(e)}")
        
        # Register handlers
        self.p2p_network.register_message_handler(MessageType.MODEL_UPDATE, handle_model_update)
        self.p2p_network.register_message_handler(MessageType.CONSENSUS_VOTE, handle_consensus_vote)
        self.p2p_network.register_message_handler(MessageType.ROUND_START, handle_round_start)
        self.p2p_network.register_message_handler(MessageType.ROUND_END, handle_round_end)
    
    def _compute_model_hash(self, parameters: Dict) -> str:
        """
        Compute hash of model parameters
        
        Args:
            parameters: Model parameters
            
        Returns:
            model_hash: SHA256 hash
        """
        try:
            import hashlib
            param_string = json.dumps(parameters, sort_keys=True, default=str)
            model_hash = hashlib.sha256(param_string.encode()).hexdigest()
            return model_hash
        except Exception as e:
            logger.error(f"Failed to compute model hash: {str(e)}")
            return ""
    
    def run_federated_training(self, num_rounds: int = 3, epochs_per_round: int = 5):
        """
        Run federated training for specified number of rounds
        
        Args:
            num_rounds: Number of federated rounds
            epochs_per_round: Number of epochs per round
        """
        logger.info(f"Starting federated training: {num_rounds} rounds, {epochs_per_round} epochs per round")
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"🔄 Starting round {round_num}/{num_rounds}")
            
            # Start federated round
            if not self.start_federated_round(round_num):
                logger.error(f"Failed to start round {round_num}")
                continue
            
            # Simulate local training (in real system, this would be actual training)
            mock_parameters = {
                f"layer{i}.weight": [0.1 * (i + 1)] * 100 for i in range(3)
            }
            
            # Submit model update
            if not self.submit_model_update(
                model_parameters=mock_parameters,
                sample_count=1000 + round_num * 500,
                training_loss=0.5 - round_num * 0.1,
                validation_accuracy=0.8 + round_num * 0.05
            ):
                logger.error(f"Failed to submit model update for round {round_num}")
                continue
            
            # Wait for aggregation and consensus
            time.sleep(5)
            
            # Submit consensus vote (simulate voting for aggregated model)
            model_hash = self._compute_model_hash(mock_parameters)
            self.submit_consensus_vote(model_hash)
            
            # Wait for round completion
            time.sleep(3)
            
            logger.info(f"✅ Completed round {round_num}/{num_rounds}")
            
            # Record round history
            self.round_history.append({
                'round_number': round_num,
                'completed_at': time.time(),
                'performance': self.performance_metrics.copy()
            })
        
        logger.info("🎉 Federated training completed!")
        
        # Print final status
        status = self.get_system_status()
        logger.info(f"Final system status: {json.dumps(status['system_info'], indent=2)}")

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Fully Decentralized System")
    
    # Create system
    system = FullyDecentralizedSystem("node_1", "127.0.0.1", 8001, NodeRole.CLIENT)
    
    # Start system
    if system.start_system():
        logger.info("System started successfully")
        
        # Join network (with empty bootstrap for standalone testing)
        system.join_network([])
        
        # Run federated training
        system.run_federated_training(num_rounds=3, epochs_per_round=5)
        
        # Get final status
        final_status = system.get_system_status()
        logger.info(f"Final status: {json.dumps(final_status['system_info'], indent=2)}")
        
        # Stop system
        system.stop_system()
    
    logger.info("✅ Fully decentralized system test completed")
