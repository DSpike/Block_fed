"""
Decentralized Coordinator for Blockchain Federated Learning
Eliminates central coordinator by implementing peer-to-peer coordination
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from blockchain.consensus_mechanisms import ProofOfContributionConsensus, ConsensusVote, ConsensusResult
from blockchain.decentralized_aggregation_contract import DecentralizedAggregationContract, ClientModelUpdate

logger = logging.getLogger(__name__)

class NodeRole(Enum):
    """Node roles in decentralized system"""
    CLIENT = "client"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"

class CoordinationState(Enum):
    """Coordination states"""
    IDLE = "idle"
    WAITING_FOR_UPDATES = "waiting_for_updates"
    AGGREGATING = "aggregating"
    CONSENSUS_VOTING = "consensus_voting"
    DISTRIBUTING_RESULTS = "distributing_results"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class NodeInfo:
    """Information about a node in the network"""
    node_id: str
    address: str
    role: NodeRole
    stake: float
    last_seen: float
    is_active: bool
    performance_score: float = 1.0

@dataclass
class CoordinationRound:
    """Represents a coordination round"""
    round_number: int
    state: CoordinationState
    required_participants: int
    current_participants: Set[str]
    consensus_threshold: float
    start_time: float
    timeout: float
    aggregated_result: Optional[Dict] = None
    consensus_result: Optional[ConsensusResult] = None

class DecentralizedCoordinator:
    """
    Decentralized coordinator that manages federated learning rounds
    without a central authority
    """
    
    def __init__(self, node_id: str, node_address: str, role: NodeRole = NodeRole.CLIENT):
        """
        Initialize decentralized coordinator
        
        Args:
            node_id: Unique node identifier
            node_address: Node network address
            role: Node role in the network
        """
        self.node_id = node_id
        self.node_address = node_address
        self.role = role
        
        # Network state
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.current_round: Optional[CoordinationRound] = None
        self.round_history: List[CoordinationRound] = []
        
        # Consensus and aggregation
        self.consensus_mechanism = ProofOfContributionConsensus()
        self.aggregation_contract = DecentralizedAggregationContract(f"contract_{node_id}")
        
        # Coordination state
        self.is_coordinating = False
        self.coordination_thread: Optional[threading.Thread] = None
        self.stop_coordination = threading.Event()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'round_started': [],
            'round_completed': [],
            'consensus_achieved': [],
            'aggregation_completed': [],
            'node_joined': [],
            'node_left': []
        }
        
        # Register self as a node
        self.register_node(node_id, node_address, role, initial_stake=100.0)
        
        logger.info(f"Decentralized coordinator initialized: {node_id} ({role.value})")
    
    def register_node(self, node_id: str, address: str, role: NodeRole, initial_stake: float = 100.0) -> bool:
        """
        Register a new node in the network
        
        Args:
            node_id: Node identifier
            address: Node address
            role: Node role
            initial_stake: Initial stake amount
            
        Returns:
            success: Whether registration was successful
        """
        try:
            node_info = NodeInfo(
                node_id=node_id,
                address=address,
                role=role,
                stake=initial_stake,
                last_seen=time.time(),
                is_active=True,
                performance_score=1.0
            )
            
            self.known_nodes[node_id] = node_info
            self.consensus_mechanism.register_client(node_id, initial_stake)
            
            logger.info(f"Registered node {node_id} ({role.value}) with stake {initial_stake}")
            
            # Notify event handlers
            self._notify_event_handlers('node_joined', {'node_id': node_id, 'role': role.value})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {str(e)}")
            return False
    
    def start_coordination_round(self, round_number: int, required_participants: int = None) -> bool:
        """
        Start a new coordination round
        
        Args:
            round_number: Round number
            required_participants: Minimum participants required
            
        Returns:
            success: Whether round was started successfully
        """
        try:
            if self.current_round is not None:
                logger.error(f"Cannot start round {round_number}: round {self.current_round.round_number} is active")
                return False
            
            if required_participants is None:
                required_participants = max(2, len(self.known_nodes) // 2)
            
            # Create coordination round
            self.current_round = CoordinationRound(
                round_number=round_number,
                state=CoordinationState.WAITING_FOR_UPDATES,
                required_participants=required_participants,
                current_participants=set(),
                consensus_threshold=0.67,
                start_time=time.time(),
                timeout=time.time() + 1800  # 30 minutes timeout
            )
            
            # Create aggregation task
            self.aggregation_contract.create_aggregation_task(round_number, required_participants)
            
            logger.info(f"Started coordination round {round_number} with {required_participants} required participants")
            
            # Notify event handlers
            self._notify_event_handlers('round_started', {
                'round_number': round_number,
                'required_participants': required_participants
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start coordination round {round_number}: {str(e)}")
            return False
    
    def submit_model_update(self, client_id: str, model_parameters: Dict, 
                           sample_count: int, training_loss: float, 
                           validation_accuracy: float) -> bool:
        """
        Submit model update for current round
        
        Args:
            client_id: Client identifier
            model_parameters: Model parameters
            sample_count: Number of training samples
            training_loss: Training loss
            validation_accuracy: Validation accuracy
            
        Returns:
            success: Whether submission was successful
        """
        try:
            if self.current_round is None:
                logger.error("No active coordination round")
                return False
            
            if self.current_round.state != CoordinationState.WAITING_FOR_UPDATES:
                logger.error(f"Cannot submit update: round state is {self.current_round.state.value}")
                return False
            
            # Create client update
            client_update = ClientModelUpdate(
                client_id=client_id,
                round_number=self.current_round.round_number,
                parameters=model_parameters,
                sample_count=sample_count,
                training_loss=training_loss,
                validation_accuracy=validation_accuracy,
                model_hash=self._compute_model_hash(model_parameters),
                timestamp=time.time(),
                ipfs_cid=f"ipfs_{client_id}_{self.current_round.round_number}"
            )
            
            # Submit to aggregation contract
            success = self.aggregation_contract.submit_client_update(client_update)
            
            if success:
                self.current_round.current_participants.add(client_id)
                logger.info(f"Model update submitted by {client_id} for round {self.current_round.round_number}")
                
                # Check if we can start aggregation
                self._check_aggregation_readiness()
            
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
            success: Whether vote was submitted
        """
        try:
            if self.current_round is None:
                logger.error("No active coordination round")
                return False
            
            if self.current_round.state != CoordinationState.CONSENSUS_VOTING:
                logger.error(f"Cannot submit vote: round state is {self.current_round.state.value}")
                return False
            
            # Create consensus vote
            vote_weight = self.consensus_mechanism.calculate_vote_weight(self.node_id)
            
            vote = ConsensusVote(
                voter_id=self.node_id,
                round_number=self.current_round.round_number,
                model_hash=model_hash,
                vote_weight=vote_weight,
                timestamp=time.time(),
                signature=f"{self.node_id}_{self.current_round.round_number}_{model_hash}",
                contribution_score=self.consensus_mechanism.contribution_scores.get(self.node_id, 1.0)
            )
            
            # Submit vote
            success = self.consensus_mechanism.submit_vote(vote)
            
            if success:
                # Also submit to aggregation contract
                self.aggregation_contract.submit_consensus_vote(
                    self.current_round.round_number, self.node_id, model_hash
                )
                
                logger.info(f"Consensus vote submitted by {self.node_id} for model {model_hash}")
                
                # Check consensus
                self._check_consensus()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to submit consensus vote: {str(e)}")
            return False
    
    def start_decentralized_coordination(self):
        """
        Start the decentralized coordination process
        """
        if self.is_coordinating:
            logger.warning("Coordination already running")
            return
        
        self.is_coordinating = True
        self.stop_coordination.clear()
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop)
        self.coordination_thread.daemon = True
        self.coordination_thread.start()
        
        logger.info("Started decentralized coordination")
    
    def stop_decentralized_coordination(self):
        """
        Stop the decentralized coordination process
        """
        if not self.is_coordinating:
            return
        
        self.stop_coordination.set()
        
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        
        self.is_coordinating = False
        logger.info("Stopped decentralized coordination")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """
        Add event handler for coordination events
        
        Args:
            event_type: Type of event
            handler: Event handler function
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def get_coordination_status(self) -> Dict:
        """
        Get current coordination status
        
        Returns:
            status: Coordination status information
        """
        status = {
            'node_id': self.node_id,
            'node_role': self.role.value,
            'is_coordinating': self.is_coordinating,
            'known_nodes': len(self.known_nodes),
            'active_nodes': len([n for n in self.known_nodes.values() if n.is_active]),
            'current_round': None,
            'round_history_count': len(self.round_history)
        }
        
        if self.current_round:
            status['current_round'] = {
                'round_number': self.current_round.round_number,
                'state': self.current_round.state.value,
                'required_participants': self.current_round.required_participants,
                'current_participants': len(self.current_round.current_participants),
                'participant_list': list(self.current_round.current_participants),
                'consensus_threshold': self.current_round.consensus_threshold,
                'start_time': self.current_round.start_time,
                'timeout': self.current_round.timeout
            }
        
        return status
    
    def _coordination_loop(self):
        """
        Main coordination loop
        """
        logger.info("Started coordination loop")
        
        while not self.stop_coordination.is_set():
            try:
                if self.current_round is not None:
                    self._process_current_round()
                
                # Check for round timeout
                self._check_round_timeout()
                
                # Update node status
                self._update_node_status()
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {str(e)}")
                time.sleep(5.0)
        
        logger.info("Coordination loop stopped")
    
    def _process_current_round(self):
        """
        Process the current coordination round
        """
        if self.current_round is None:
            return
        
        round_state = self.current_round.state
        
        if round_state == CoordinationState.WAITING_FOR_UPDATES:
            self._check_aggregation_readiness()
        
        elif round_state == CoordinationState.AGGREGATING:
            self._perform_aggregation()
        
        elif round_state == CoordinationState.CONSENSUS_VOTING:
            self._check_consensus()
        
        elif round_state == CoordinationState.DISTRIBUTING_RESULTS:
            self._distribute_results()
    
    def _check_aggregation_readiness(self):
        """
        Check if aggregation can be started
        """
        if self.current_round is None:
            return
        
        submitted_count = len(self.current_round.current_participants)
        required_count = self.current_round.required_participants
        
        if submitted_count >= required_count:
            self.current_round.state = CoordinationState.AGGREGATING
            logger.info(f"Starting aggregation for round {self.current_round.round_number}")
    
    def _perform_aggregation(self):
        """
        Perform decentralized aggregation
        """
        if self.current_round is None:
            return
        
        try:
            # Perform aggregation using the contract
            aggregation_result = self.aggregation_contract.perform_decentralized_aggregation(
                self.current_round.round_number
            )
            
            if aggregation_result:
                self.current_round.aggregated_result = aggregation_result
                self.current_round.state = CoordinationState.CONSENSUS_VOTING
                
                logger.info(f"✅ Aggregation completed for round {self.current_round.round_number}")
                
                # Notify event handlers
                self._notify_event_handlers('aggregation_completed', {
                    'round_number': self.current_round.round_number,
                    'model_hash': aggregation_result['model_hash'],
                    'num_clients': aggregation_result['num_clients']
                })
            else:
                logger.error(f"Aggregation failed for round {self.current_round.round_number}")
                self.current_round.state = CoordinationState.FAILED
        
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            self.current_round.state = CoordinationState.FAILED
    
    def _check_consensus(self):
        """
        Check if consensus has been achieved
        """
        if self.current_round is None:
            return
        
        try:
            consensus_result = self.consensus_mechanism.check_consensus(
                self.current_round.round_number
            )
            
            if consensus_result and consensus_result.achieved_consensus:
                self.current_round.consensus_result = consensus_result
                self.current_round.state = CoordinationState.DISTRIBUTING_RESULTS
                
                logger.info(f"✅ Consensus achieved for round {self.current_round.round_number}")
                
                # Notify event handlers
                self._notify_event_handlers('consensus_achieved', {
                    'round_number': self.current_round.round_number,
                    'winning_model_hash': consensus_result.winning_model_hash,
                    'voters': consensus_result.voters
                })
        
        except Exception as e:
            logger.error(f"Error checking consensus: {str(e)}")
    
    def _distribute_results(self):
        """
        Distribute aggregation results to all participants
        """
        if self.current_round is None:
            return
        
        try:
            # Mark round as completed
            self.current_round.state = CoordinationState.COMPLETED
            
            # Add to history
            self.round_history.append(self.current_round)
            
            logger.info(f"✅ Round {self.current_round.round_number} completed successfully")
            
            # Notify event handlers
            self._notify_event_handlers('round_completed', {
                'round_number': self.current_round.round_number,
                'aggregated_result': self.current_round.aggregated_result,
                'consensus_result': self.current_round.consensus_result
            })
            
            # Clear current round
            self.current_round = None
        
        except Exception as e:
            logger.error(f"Error distributing results: {str(e)}")
    
    def _check_round_timeout(self):
        """
        Check if current round has timed out
        """
        if self.current_round is None:
            return
        
        if time.time() > self.current_round.timeout:
            logger.warning(f"Round {self.current_round.round_number} timed out")
            self.current_round.state = CoordinationState.FAILED
            
            # Clear current round
            self.current_round = None
    
    def _update_node_status(self):
        """
        Update status of known nodes
        """
        current_time = time.time()
        timeout_threshold = 300  # 5 minutes
        
        for node_id, node_info in self.known_nodes.items():
            if current_time - node_info.last_seen > timeout_threshold:
                if node_info.is_active:
                    node_info.is_active = False
                    logger.warning(f"Node {node_id} marked as inactive")
                    
                    # Notify event handlers
                    self._notify_event_handlers('node_left', {'node_id': node_id})
    
    def _notify_event_handlers(self, event_type: str, event_data: Dict):
        """
        Notify event handlers of an event
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {str(e)}")
    
    def _compute_model_hash(self, parameters: Dict) -> str:
        """
        Compute hash of model parameters
        
        Args:
            parameters: Model parameters
            
        Returns:
            model_hash: SHA256 hash
        """
        try:
            # Create hashable representation
            param_string = json.dumps(parameters, sort_keys=True, default=str)
            model_hash = hashlib.sha256(param_string.encode()).hexdigest()
            return model_hash
        except Exception as e:
            logger.error(f"Failed to compute model hash: {str(e)}")
            return ""

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Decentralized Coordinator")
    
    # Create coordinator
    coordinator = DecentralizedCoordinator("node_1", "127.0.0.1:8001", NodeRole.CLIENT)
    
    # Register additional nodes
    coordinator.register_node("node_2", "127.0.0.1:8002", NodeRole.CLIENT, 150.0)
    coordinator.register_node("node_3", "127.0.0.1:8003", NodeRole.CLIENT, 200.0)
    
    # Add event handlers
    def on_round_started(event_data):
        logger.info(f"📢 Round started: {event_data}")
    
    def on_round_completed(event_data):
        logger.info(f"🎉 Round completed: {event_data}")
    
    def on_consensus_achieved(event_data):
        logger.info(f"🤝 Consensus achieved: {event_data}")
    
    coordinator.add_event_handler('round_started', on_round_started)
    coordinator.add_event_handler('round_completed', on_round_completed)
    coordinator.add_event_handler('consensus_achieved', on_consensus_achieved)
    
    # Start coordination
    coordinator.start_decentralized_coordination()
    
    # Start a coordination round
    round_number = 1
    coordinator.start_coordination_round(round_number, required_participants=3)
    
    # Simulate model updates
    for i in range(3):
        client_id = f"node_{i+1}"
        mock_parameters = {
            "layer1.weight": [0.1] * 100,
            "layer1.bias": [0.01] * 10
        }
        
        coordinator.submit_model_update(
            client_id=client_id,
            model_parameters=mock_parameters,
            sample_count=1000 + i * 500,
            training_loss=0.5 - i * 0.1,
            validation_accuracy=0.8 + i * 0.05
        )
    
    # Wait for coordination to complete
    time.sleep(10)
    
    # Check status
    status = coordinator.get_coordination_status()
    logger.info(f"Coordination status: {json.dumps(status, indent=2)}")
    
    # Stop coordination
    coordinator.stop_decentralized_coordination()
    
    logger.info("✅ Decentralized coordinator test completed")
