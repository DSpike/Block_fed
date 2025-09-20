"""
Decentralized Consensus Mechanisms for Blockchain Federated Learning
Implements Proof of Contribution (PoC) consensus for model aggregation
"""

import hashlib
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConsensusType(Enum):
    PROOF_OF_CONTRIBUTION = "proof_of_contribution"
    PROOF_OF_STAKE = "proof_of_stake"
    PROOF_OF_WORK = "proof_of_work"

@dataclass
class ConsensusVote:
    """Represents a consensus vote for model aggregation"""
    voter_id: str
    round_number: int
    model_hash: str
    vote_weight: float
    timestamp: float
    signature: str
    contribution_score: float

@dataclass
class ConsensusResult:
    """Result of consensus mechanism"""
    round_number: int
    winning_model_hash: str
    total_votes: int
    consensus_threshold: float
    achieved_consensus: bool
    consensus_time: float
    voters: List[str]

class ProofOfContributionConsensus:
    """
    Proof of Contribution consensus mechanism for federated learning
    
    Clients vote on model aggregations based on their contribution quality
    and stake in the system.
    """
    
    def __init__(self, consensus_threshold: float = 0.67):
        """
        Initialize PoC consensus
        
        Args:
            consensus_threshold: Minimum percentage of votes needed for consensus
        """
        self.consensus_threshold = consensus_threshold
        self.votes: Dict[int, List[ConsensusVote]] = {}  # round_number -> votes
        self.contribution_scores: Dict[str, float] = {}  # client_id -> score
        self.stakes: Dict[str, float] = {}  # client_id -> stake amount
        
        logger.info(f"Proof of Contribution consensus initialized with threshold {consensus_threshold}")
    
    def register_client(self, client_id: str, initial_stake: float = 100.0):
        """
        Register a new client with initial stake
        
        Args:
            client_id: Unique client identifier
            initial_stake: Initial stake amount
        """
        self.stakes[client_id] = initial_stake
        self.contribution_scores[client_id] = 1.0  # Start with neutral score
        
        logger.info(f"Client {client_id} registered with stake {initial_stake}")
    
    def submit_vote(self, vote: ConsensusVote) -> bool:
        """
        Submit a consensus vote for model aggregation
        
        Args:
            vote: Consensus vote data
            
        Returns:
            success: Whether vote was accepted
        """
        try:
            # Validate vote
            if not self._validate_vote(vote):
                logger.error(f"Invalid vote from {vote.voter_id}")
                return False
            
            # Initialize votes list for round if needed
            if vote.round_number not in self.votes:
                self.votes[vote.round_number] = []
            
            # Check if client already voted in this round
            existing_votes = [v for v in self.votes[vote.round_number] if v.voter_id == vote.voter_id]
            if existing_votes:
                logger.warning(f"Client {vote.voter_id} already voted in round {vote.round_number}")
                return False
            
            # Add vote
            self.votes[vote.round_number].append(vote)
            
            logger.info(f"Vote accepted from {vote.voter_id} for round {vote.round_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit vote: {str(e)}")
            return False
    
    def check_consensus(self, round_number: int) -> Optional[ConsensusResult]:
        """
        Check if consensus has been achieved for a round
        
        Args:
            round_number: Round to check consensus for
            
        Returns:
            consensus_result: Consensus result if achieved, None otherwise
        """
        if round_number not in self.votes:
            logger.warning(f"No votes found for round {round_number}")
            return None
        
        votes = self.votes[round_number]
        if not votes:
            logger.warning(f"Empty votes list for round {round_number}")
            return None
        
        # Count votes by model hash
        model_votes: Dict[str, List[ConsensusVote]] = {}
        for vote in votes:
            if vote.model_hash not in model_votes:
                model_votes[vote.model_hash] = []
            model_votes[vote.model_hash].append(vote)
        
        # Calculate weighted votes
        model_weights: Dict[str, float] = {}
        for model_hash, model_votes_list in model_votes.items():
            total_weight = sum(vote.vote_weight for vote in model_votes_list)
            model_weights[model_hash] = total_weight
        
        # Find model with highest weight
        if not model_weights:
            logger.warning(f"No model weights calculated for round {round_number}")
            return None
        
        winning_model = max(model_weights, key=model_weights.get)
        winning_weight = model_weights[winning_model]
        
        # Calculate total possible weight (all registered clients)
        total_possible_weight = sum(self.stakes.values())
        
        # Check if consensus threshold is met
        consensus_ratio = winning_weight / total_possible_weight if total_possible_weight > 0 else 0
        achieved_consensus = consensus_ratio >= self.consensus_threshold
        
        # Get voters for winning model
        winning_voters = [vote.voter_id for vote in model_votes[winning_model]]
        
        consensus_result = ConsensusResult(
            round_number=round_number,
            winning_model_hash=winning_model,
            total_votes=len(votes),
            consensus_threshold=self.consensus_threshold,
            achieved_consensus=achieved_consensus,
            consensus_time=time.time(),
            voters=winning_voters
        )
        
        if achieved_consensus:
            logger.info(f"✅ CONSENSUS ACHIEVED for round {round_number}: {winning_model} "
                       f"({consensus_ratio:.2%} > {self.consensus_threshold:.2%})")
        else:
            logger.info(f"❌ No consensus for round {round_number}: {winning_model} "
                       f"({consensus_ratio:.2%} < {self.consensus_threshold:.2%})")
        
        return consensus_result
    
    def update_contribution_score(self, client_id: str, performance_metrics: Dict[str, float]):
        """
        Update client's contribution score based on performance
        
        Args:
            client_id: Client identifier
            performance_metrics: Dictionary of performance metrics
        """
        try:
            # Calculate contribution score based on multiple metrics
            accuracy = performance_metrics.get('accuracy', 0.5)
            precision = performance_metrics.get('precision', 0.5)
            recall = performance_metrics.get('recall', 0.5)
            f1_score = performance_metrics.get('f1_score', 0.5)
            
            # Weighted average of metrics
            contribution_score = (
                accuracy * 0.3 +
                precision * 0.2 +
                recall * 0.2 +
                f1_score * 0.3
            )
            
            # Update score with exponential moving average
            alpha = 0.1  # Learning rate
            old_score = self.contribution_scores.get(client_id, 1.0)
            new_score = alpha * contribution_score + (1 - alpha) * old_score
            
            self.contribution_scores[client_id] = new_score
            
            logger.info(f"Updated contribution score for {client_id}: {old_score:.3f} -> {new_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update contribution score for {client_id}: {str(e)}")
    
    def calculate_vote_weight(self, client_id: str) -> float:
        """
        Calculate vote weight for a client based on stake and contribution
        
        Args:
            client_id: Client identifier
            
        Returns:
            vote_weight: Calculated vote weight
        """
        stake = self.stakes.get(client_id, 0.0)
        contribution_score = self.contribution_scores.get(client_id, 1.0)
        
        # Vote weight = stake * contribution_score
        vote_weight = stake * contribution_score
        
        return vote_weight
    
    def _validate_vote(self, vote: ConsensusVote) -> bool:
        """
        Validate a consensus vote
        
        Args:
            vote: Vote to validate
            
        Returns:
            is_valid: Whether vote is valid
        """
        # Check if client is registered
        if vote.voter_id not in self.stakes:
            logger.error(f"Unregistered client {vote.voter_id}")
            return False
        
        # Check vote weight matches client's calculated weight
        expected_weight = self.calculate_vote_weight(vote.voter_id)
        if abs(vote.vote_weight - expected_weight) > 0.01:
            logger.error(f"Vote weight mismatch for {vote.voter_id}: "
                        f"expected {expected_weight}, got {vote.vote_weight}")
            return False
        
        # Check timestamp is reasonable (within last hour)
        current_time = time.time()
        if abs(current_time - vote.timestamp) > 3600:
            logger.error(f"Vote timestamp too old for {vote.voter_id}")
            return False
        
        return True
    
    def get_consensus_status(self, round_number: int) -> Dict:
        """
        Get detailed consensus status for a round
        
        Args:
            round_number: Round number
            
        Returns:
            status: Detailed consensus status
        """
        if round_number not in self.votes:
            return {
                'round_number': round_number,
                'status': 'no_votes',
                'votes_received': 0,
                'consensus_threshold': self.consensus_threshold,
                'achieved_consensus': False
            }
        
        votes = self.votes[round_number]
        
        # Count votes by model
        model_counts = {}
        total_weight = 0
        
        for vote in votes:
            model_hash = vote.model_hash
            if model_hash not in model_counts:
                model_counts[model_hash] = {'count': 0, 'weight': 0.0}
            
            model_counts[model_hash]['count'] += 1
            model_counts[model_hash]['weight'] += vote.vote_weight
            total_weight += vote.vote_weight
        
        # Find leading model
        leading_model = max(model_counts, key=lambda k: model_counts[k]['weight']) if model_counts else None
        leading_weight = model_counts[leading_model]['weight'] if leading_model else 0.0
        
        # Calculate consensus ratio
        total_possible_weight = sum(self.stakes.values())
        consensus_ratio = leading_weight / total_possible_weight if total_possible_weight > 0 else 0.0
        
        return {
            'round_number': round_number,
            'status': 'voting_in_progress',
            'votes_received': len(votes),
            'total_possible_votes': len(self.stakes),
            'consensus_threshold': self.consensus_threshold,
            'achieved_consensus': consensus_ratio >= self.consensus_threshold,
            'consensus_ratio': consensus_ratio,
            'leading_model': leading_model,
            'leading_weight': leading_weight,
            'model_votes': model_counts,
            'total_weight': total_weight
        }

class DecentralizedAggregator:
    """
    Decentralized aggregator that uses consensus mechanisms
    """
    
    def __init__(self, consensus_mechanism: ProofOfContributionConsensus):
        """
        Initialize decentralized aggregator
        
        Args:
            consensus_mechanism: Consensus mechanism to use
        """
        self.consensus = consensus_mechanism
        self.aggregation_history: Dict[int, Dict] = {}
        
        logger.info("Decentralized aggregator initialized")
    
    def propose_aggregation(self, round_number: int, model_hash: str, 
                           proposer_id: str) -> bool:
        """
        Propose an aggregation for consensus
        
        Args:
            round_number: Round number
            model_hash: Hash of proposed aggregated model
            proposer_id: ID of client proposing the aggregation
            
        Returns:
            success: Whether proposal was accepted
        """
        try:
            # Create consensus vote for the proposal
            vote_weight = self.consensus.calculate_vote_weight(proposer_id)
            
            vote = ConsensusVote(
                voter_id=proposer_id,
                round_number=round_number,
                model_hash=model_hash,
                vote_weight=vote_weight,
                timestamp=time.time(),
                signature=f"{proposer_id}_{round_number}_{model_hash}",
                contribution_score=self.consensus.contribution_scores.get(proposer_id, 1.0)
            )
            
            # Submit vote
            success = self.consensus.submit_vote(vote)
            
            if success:
                logger.info(f"Aggregation proposed by {proposer_id} for round {round_number}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to propose aggregation: {str(e)}")
            return False
    
    def check_aggregation_consensus(self, round_number: int) -> Optional[ConsensusResult]:
        """
        Check if aggregation consensus has been achieved
        
        Args:
            round_number: Round number
            
        Returns:
            consensus_result: Consensus result if achieved
        """
        return self.consensus.check_consensus(round_number)
    
    def finalize_aggregation(self, round_number: int, consensus_result: ConsensusResult) -> bool:
        """
        Finalize aggregation based on consensus result
        
        Args:
            round_number: Round number
            consensus_result: Consensus result
            
        Returns:
            success: Whether aggregation was finalized
        """
        try:
            if not consensus_result.achieved_consensus:
                logger.error(f"Cannot finalize aggregation without consensus for round {round_number}")
                return False
            
            # Record aggregation in history
            self.aggregation_history[round_number] = {
                'consensus_result': consensus_result,
                'finalization_time': time.time(),
                'winning_model_hash': consensus_result.winning_model_hash,
                'voters': consensus_result.voters
            }
            
            logger.info(f"✅ Aggregation finalized for round {round_number} with consensus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize aggregation: {str(e)}")
            return False

def create_consensus_mechanism(consensus_type: ConsensusType = ConsensusType.PROOF_OF_CONTRIBUTION) -> ProofOfContributionConsensus:
    """
    Factory function to create consensus mechanism
    
    Args:
        consensus_type: Type of consensus mechanism
        
    Returns:
        consensus: Consensus mechanism instance
    """
    if consensus_type == ConsensusType.PROOF_OF_CONTRIBUTION:
        return ProofOfContributionConsensus()
    else:
        raise ValueError(f"Unsupported consensus type: {consensus_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test the consensus mechanism
    logger.info("Testing Proof of Contribution Consensus")
    
    # Create consensus mechanism
    consensus = create_consensus_mechanism()
    
    # Register clients
    consensus.register_client("client_1", initial_stake=100.0)
    consensus.register_client("client_2", initial_stake=150.0)
    consensus.register_client("client_3", initial_stake=200.0)
    
    # Update contribution scores
    consensus.update_contribution_score("client_1", {
        'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85
    })
    consensus.update_contribution_score("client_2", {
        'accuracy': 0.92, 'precision': 0.90, 'recall': 0.94, 'f1_score': 0.92
    })
    consensus.update_contribution_score("client_3", {
        'accuracy': 0.78, 'precision': 0.75, 'recall': 0.81, 'f1_score': 0.78
    })
    
    # Simulate voting
    round_number = 1
    model_hash = "model_abc123"
    
    # Create votes
    for client_id in ["client_1", "client_2", "client_3"]:
        vote_weight = consensus.calculate_vote_weight(client_id)
        
        vote = ConsensusVote(
            voter_id=client_id,
            round_number=round_number,
            model_hash=model_hash,
            vote_weight=vote_weight,
            timestamp=time.time(),
            signature=f"{client_id}_{round_number}_{model_hash}",
            contribution_score=consensus.contribution_scores[client_id]
        )
        
        consensus.submit_vote(vote)
    
    # Check consensus
    result = consensus.check_consensus(round_number)
    if result:
        logger.info(f"Consensus achieved: {result.achieved_consensus}")
        logger.info(f"Winning model: {result.winning_model_hash}")
        logger.info(f"Voters: {result.voters}")
    
    # Get status
    status = consensus.get_consensus_status(round_number)
    logger.info(f"Consensus status: {json.dumps(status, indent=2)}")
    
    logger.info("✅ Consensus mechanism test completed")
