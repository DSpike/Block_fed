#!/usr/bin/env python3
"""
Incentive Mechanisms and Provenance Tracking System
Implements reward mechanisms, reputation tracking, and comprehensive provenance for federated learning
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from web3 import Web3
from eth_account import Account
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContributionType(Enum):
    """Types of contributions to federated learning"""
    MODEL_UPDATE = "model_update"
    DATA_QUALITY = "data_quality"
    PARTICIPATION = "participation"
    VERIFICATION = "verification"
    AGGREGATION = "aggregation"

class RewardType(Enum):
    """Types of rewards"""
    TOKEN_REWARD = "token_reward"
    REPUTATION_BOOST = "reputation_boost"
    PRIORITY_ACCESS = "priority_access"
    GOVERNANCE_RIGHTS = "governance_rights"

@dataclass
class Contribution:
    """Represents a contribution to federated learning"""
    contributor_id: str
    contribution_type: ContributionType
    round_number: int
    timestamp: float
    value: float
    quality_score: float
    metadata: Dict[str, Any]
    verification_hash: str
    blockchain_tx_hash: Optional[str] = None

@dataclass
class Reward:
    """Represents a reward given to a participant"""
    recipient_id: str
    reward_type: RewardType
    amount: float
    round_number: int
    timestamp: float
    contribution_hash: str
    justification: str
    blockchain_tx_hash: Optional[str] = None

@dataclass
class ProvenanceRecord:
    """Represents a provenance record for model or data"""
    record_id: str
    record_type: str  # 'model', 'data', 'aggregation', 'verification'
    round_number: int
    timestamp: float
    participants: List[str]
    operations: List[Dict[str, Any]]
    verification_hashes: List[str]
    blockchain_tx_hashes: List[str]
    metadata: Dict[str, Any]

class IncentiveCalculator:
    """
    Calculates incentives based on contributions and performance
    """
    
    def __init__(self, base_reward: float = 100.0, quality_weight: float = 0.4, 
                 participation_weight: float = 0.3, verification_weight: float = 0.3):
        """
        Initialize incentive calculator
        
        Args:
            base_reward: Base reward amount
            quality_weight: Weight for quality contribution
            participation_weight: Weight for participation
            verification_weight: Weight for verification
        """
        self.base_reward = base_reward
        self.quality_weight = quality_weight
        self.participation_weight = participation_weight
        self.verification_weight = verification_weight
        
        # Contribution type multipliers
        self.contribution_multipliers = {
            ContributionType.MODEL_UPDATE: 1.0,
            ContributionType.DATA_QUALITY: 0.8,
            ContributionType.PARTICIPATION: 0.6,
            ContributionType.VERIFICATION: 0.9,
            ContributionType.AGGREGATION: 1.2
        }
        
        logger.info("Incentive Calculator initialized")
    
    def calculate_contribution_score(self, contribution: Contribution) -> float:
        """
        Calculate contribution score
        
        Args:
            contribution: Contribution to evaluate
            
        Returns:
            score: Contribution score (0.0 to 1.0)
        """
        # Base score from contribution type
        base_score = self.contribution_multipliers.get(contribution.contribution_type, 0.5)
        
        # Quality adjustment
        quality_adjustment = contribution.quality_score * self.quality_weight
        
        # Value adjustment (normalized)
        value_adjustment = min(contribution.value, 1.0) * 0.2
        
        # Calculate final score
        final_score = (base_score + quality_adjustment + value_adjustment) / 2.0
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def calculate_reward(self, contribution: Contribution, reputation_score: float) -> Reward:
        """
        Calculate reward for contribution
        
        Args:
            contribution: Contribution to reward
            reputation_score: Current reputation score of contributor
            
        Returns:
            reward: Calculated reward
        """
        # Calculate contribution score
        contribution_score = self.calculate_contribution_score(contribution)
        
        # Base reward calculation
        base_amount = self.base_reward * contribution_score
        
        # Reputation multiplier (higher reputation = higher rewards)
        reputation_multiplier = 1.0 + (reputation_score * 0.5)
        
        # Final reward amount
        reward_amount = base_amount * reputation_multiplier
        
        # Determine reward type based on contribution
        if contribution.contribution_type == ContributionType.MODEL_UPDATE:
            reward_type = RewardType.TOKEN_REWARD
        elif contribution.contribution_type == ContributionType.VERIFICATION:
            reward_type = RewardType.REPUTATION_BOOST
        else:
            reward_type = RewardType.TOKEN_REWARD
        
        # Create reward
        reward = Reward(
            recipient_id=contribution.contributor_id,
            reward_type=reward_type,
            amount=reward_amount,
            round_number=contribution.round_number,
            timestamp=time.time(),
            contribution_hash=contribution.verification_hash,
            justification=f"Reward for {contribution.contribution_type.value} with score {contribution_score:.3f}"
        )
        
        logger.info(f"Calculated reward for {contribution.contributor_id}: {reward_amount:.2f} {reward_type.value}")
        return reward
    
    def calculate_penalty(self, contributor_id: str, violation_type: str, 
                         severity: float, current_reputation: float) -> Tuple[float, str]:
        """
        Calculate penalty for violations
        
        Args:
            contributor_id: Contributor ID
            violation_type: Type of violation
            severity: Severity of violation (0.0 to 1.0)
            current_reputation: Current reputation score
            
        Returns:
            penalty_amount: Penalty amount
            justification: Penalty justification
        """
        # Base penalty based on severity
        base_penalty = self.base_reward * severity * 0.5
        
        # Reputation-based adjustment (higher reputation = higher penalty)
        reputation_multiplier = 1.0 + (current_reputation * 0.3)
        
        # Final penalty
        penalty_amount = base_penalty * reputation_multiplier
        
        justification = f"Penalty for {violation_type} (severity: {severity:.2f})"
        
        logger.warning(f"Calculated penalty for {contributor_id}: {penalty_amount:.2f} - {justification}")
        return penalty_amount, justification

class ProvenanceTracker:
    """
    Tracks provenance of models, data, and operations
    """
    
    def __init__(self):
        """Initialize provenance tracker"""
        self.provenance_records = {}
        self.operation_history = []
        self.lock = threading.Lock()
        
        logger.info("Provenance Tracker initialized")
    
    def create_provenance_record(self, record_type: str, round_number: int, 
                               participants: List[str], operations: List[Dict[str, Any]],
                               metadata: Dict[str, Any] = None) -> ProvenanceRecord:
        """
        Create a new provenance record
        
        Args:
            record_type: Type of record
            round_number: Round number
            participants: List of participants
            operations: List of operations
            metadata: Additional metadata
            
        Returns:
            provenance_record: Created provenance record
        """
        record_id = self.generate_record_id(record_type, round_number, participants)
        
        provenance_record = ProvenanceRecord(
            record_id=record_id,
            record_type=record_type,
            round_number=round_number,
            timestamp=time.time(),
            participants=participants,
            operations=operations,
            verification_hashes=[],
            blockchain_tx_hashes=[],
            metadata=metadata or {}
        )
        
        with self.lock:
            self.provenance_records[record_id] = provenance_record
            self.operation_history.append(provenance_record)
        
        logger.info(f"Created provenance record: {record_id}")
        return provenance_record
    
    def add_verification_hash(self, record_id: str, verification_hash: str):
        """Add verification hash to provenance record"""
        with self.lock:
            if record_id in self.provenance_records:
                self.provenance_records[record_id].verification_hashes.append(verification_hash)
                logger.info(f"Added verification hash to {record_id}: {verification_hash}")
    
    def add_blockchain_tx_hash(self, record_id: str, tx_hash: str):
        """Add blockchain transaction hash to provenance record"""
        with self.lock:
            if record_id in self.provenance_records:
                self.provenance_records[record_id].blockchain_tx_hashes.append(tx_hash)
                logger.info(f"Added blockchain TX hash to {record_id}: {tx_hash}")
    
    def generate_record_id(self, record_type: str, round_number: int, participants: List[str]) -> str:
        """Generate unique record ID"""
        data = f"{record_type}:{round_number}:{':'.join(sorted(participants))}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_provenance_chain(self, record_id: str) -> List[ProvenanceRecord]:
        """Get provenance chain for a record"""
        with self.lock:
            if record_id in self.provenance_records:
                record = self.provenance_records[record_id]
                # Find related records
                related_records = []
                for other_record in self.provenance_records.values():
                    if (other_record.round_number == record.round_number and 
                        any(p in other_record.participants for p in record.participants)):
                        related_records.append(other_record)
                
                return sorted(related_records, key=lambda x: x.timestamp)
            return []
    
    def verify_provenance_integrity(self, record_id: str) -> bool:
        """Verify integrity of provenance record"""
        with self.lock:
            if record_id not in self.provenance_records:
                return False
            
            record = self.provenance_records[record_id]
            
            # Check if all verification hashes are present
            if not record.verification_hashes:
                logger.warning(f"No verification hashes for record {record_id}")
                return False
            
            # Check if blockchain transactions are recorded
            if not record.blockchain_tx_hashes:
                logger.warning(f"No blockchain transactions for record {record_id}")
                return False
            
            # Verify timestamp consistency
            if record.timestamp <= 0:
                logger.warning(f"Invalid timestamp for record {record_id}")
                return False
            
            logger.info(f"Provenance integrity verified for record {record_id}")
            return True
    
    def get_participant_provenance(self, participant_id: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a participant"""
        with self.lock:
            participant_records = []
            for record in self.provenance_records.values():
                if participant_id in record.participants:
                    participant_records.append(record)
            
            return sorted(participant_records, key=lambda x: x.timestamp, reverse=True)
    
    def generate_provenance_report(self, round_number: int = None) -> Dict[str, Any]:
        """Generate comprehensive provenance report"""
        with self.lock:
            if round_number is not None:
                records = [r for r in self.provenance_records.values() if r.round_number == round_number]
            else:
                records = list(self.provenance_records.values())
            
            # Analyze records
            total_records = len(records)
            record_types = {}
            participants = set()
            total_operations = 0
            
            for record in records:
                record_types[record.record_type] = record_types.get(record.record_type, 0) + 1
                participants.update(record.participants)
                total_operations += len(record.operations)
            
            # Calculate integrity metrics
            verified_records = sum(1 for r in records if self.verify_provenance_integrity(r.record_id))
            integrity_rate = verified_records / total_records if total_records > 0 else 0
            
            report = {
                'total_records': total_records,
                'record_types': record_types,
                'unique_participants': len(participants),
                'total_operations': total_operations,
                'integrity_rate': integrity_rate,
                'verified_records': verified_records,
                'round_number': round_number,
                'generated_at': time.time()
            }
            
            logger.info(f"Generated provenance report: {total_records} records, {integrity_rate:.2%} integrity")
            return report

class ReputationManager:
    """
    Manages reputation scores and reputation-based incentives
    """
    
    def __init__(self, initial_reputation: float = 0.5, decay_factor: float = 0.95):
        """
        Initialize reputation manager
        
        Args:
            initial_reputation: Initial reputation score for new participants
            decay_factor: Reputation decay factor over time
        """
        self.initial_reputation = initial_reputation
        self.decay_factor = decay_factor
        self.reputation_scores = {}
        self.reputation_history = {}
        self.lock = threading.Lock()
        
        logger.info("Reputation Manager initialized")
    
    def get_reputation(self, participant_id: str) -> float:
        """Get current reputation score"""
        with self.lock:
            return self.reputation_scores.get(participant_id, self.initial_reputation)
    
    def update_reputation(self, participant_id: str, contribution_score: float, 
                         reward_amount: float, penalty_amount: float = 0.0):
        """
        Update reputation score based on contribution and rewards
        
        Args:
            participant_id: Participant ID
            contribution_score: Contribution score (0.0 to 1.0)
            reward_amount: Reward amount received
            penalty_amount: Penalty amount (if any)
        """
        with self.lock:
            current_reputation = self.reputation_scores.get(participant_id, self.initial_reputation)
            
            # Calculate reputation change
            contribution_impact = contribution_score * 0.1  # Positive impact
            reward_impact = (reward_amount / 100.0) * 0.05  # Small positive impact
            penalty_impact = -(penalty_amount / 100.0) * 0.1  # Negative impact
            
            reputation_change = contribution_impact + reward_impact + penalty_impact
            
            # Apply decay
            decayed_reputation = current_reputation * self.decay_factor
            
            # Calculate new reputation
            new_reputation = decayed_reputation + reputation_change
            
            # Ensure reputation stays within bounds
            new_reputation = max(0.0, min(1.0, new_reputation))
            
            # Update reputation
            self.reputation_scores[participant_id] = new_reputation
            
            # Record in history
            if participant_id not in self.reputation_history:
                self.reputation_history[participant_id] = []
            
            self.reputation_history[participant_id].append({
                'timestamp': time.time(),
                'old_reputation': current_reputation,
                'new_reputation': new_reputation,
                'change': reputation_change,
                'contribution_score': contribution_score,
                'reward_amount': reward_amount,
                'penalty_amount': penalty_amount
            })
            
            logger.info(f"Updated reputation for {participant_id}: {current_reputation:.3f} -> {new_reputation:.3f}")
    
    def get_reputation_history(self, participant_id: str) -> List[Dict[str, Any]]:
        """Get reputation history for participant"""
        with self.lock:
            return self.reputation_history.get(participant_id, []).copy()
    
    def get_top_participants(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top participants by reputation"""
        with self.lock:
            sorted_participants = sorted(
                self.reputation_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_participants[:limit]
    
    def calculate_reputation_based_reward_multiplier(self, participant_id: str) -> float:
        """Calculate reward multiplier based on reputation"""
        reputation = self.get_reputation(participant_id)
        
        # Higher reputation = higher multiplier (1.0 to 2.0)
        multiplier = 1.0 + reputation
        
        return multiplier

class IncentiveProvenanceSystem:
    """
    Main system for managing incentives and provenance
    """
    
    def __init__(self, ethereum_config: Dict = None):
        """
        Initialize incentive and provenance system
        
        Args:
            ethereum_config: Ethereum configuration for blockchain integration
        """
        self.ethereum_config = ethereum_config
        
        # Initialize components
        self.incentive_calculator = IncentiveCalculator()
        self.provenance_tracker = ProvenanceTracker()
        self.reputation_manager = ReputationManager()
        
        # Storage
        self.contributions = {}
        self.rewards = {}
        self.penalties = {}
        
        # Threading
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Incentive and Provenance System initialized")
    
    def record_contribution(self, contribution: Contribution) -> str:
        """
        Record a contribution and calculate reward
        
        Args:
            contribution: Contribution to record
            
        Returns:
            contribution_id: Unique contribution ID
        """
        contribution_id = hashlib.sha256(
            f"{contribution.contributor_id}:{contribution.round_number}:{contribution.timestamp}".encode()
        ).hexdigest()[:16]
        
        with self.lock:
            self.contributions[contribution_id] = contribution
        
        # Calculate reward
        reputation_score = self.reputation_manager.get_reputation(contribution.contributor_id)
        reward = self.incentive_calculator.calculate_reward(contribution, reputation_score)
        
        # Store reward
        reward_id = hashlib.sha256(f"{contribution_id}:{reward.timestamp}".encode()).hexdigest()[:16]
        self.rewards[reward_id] = reward
        
        # Update reputation
        self.reputation_manager.update_reputation(
            contribution.contributor_id,
            self.incentive_calculator.calculate_contribution_score(contribution),
            reward.amount
        )
        
        # Create provenance record
        provenance_record = self.provenance_tracker.create_provenance_record(
            record_type='contribution',
            round_number=contribution.round_number,
            participants=[contribution.contributor_id],
            operations=[{
                'type': 'contribution',
                'contribution_type': contribution.contribution_type.value,
                'value': contribution.value,
                'quality_score': contribution.quality_score
            }],
            metadata=contribution.metadata
        )
        
        # Add verification hash
        self.provenance_tracker.add_verification_hash(provenance_record.record_id, contribution.verification_hash)
        
        logger.info(f"Recorded contribution {contribution_id} with reward {reward.amount:.2f}")
        return contribution_id
    
    def record_aggregation(self, round_number: int, participants: List[str], 
                          aggregation_data: Dict[str, Any]) -> str:
        """
        Record model aggregation
        
        Args:
            round_number: Round number
            participants: List of participants
            aggregation_data: Aggregation data
            
        Returns:
            aggregation_id: Unique aggregation ID
        """
        aggregation_id = hashlib.sha256(
            f"aggregation:{round_number}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Create provenance record
        provenance_record = self.provenance_tracker.create_provenance_record(
            record_type='aggregation',
            round_number=round_number,
            participants=participants,
            operations=[{
                'type': 'aggregation',
                'participants': participants,
                'aggregation_data': aggregation_data
            }],
            metadata={'aggregation_id': aggregation_id}
        )
        
        # Record contributions for all participants
        for participant_id in participants:
            contribution = Contribution(
                contributor_id=participant_id,
                contribution_type=ContributionType.AGGREGATION,
                round_number=round_number,
                timestamp=time.time(),
                value=1.0,  # Full participation
                quality_score=0.8,  # Default quality for aggregation
                metadata={'aggregation_id': aggregation_id},
                verification_hash=hashlib.sha256(f"{participant_id}:{round_number}".encode()).hexdigest()
            )
            
            self.record_contribution(contribution)
        
        logger.info(f"Recorded aggregation {aggregation_id} for round {round_number}")
        return aggregation_id
    
    def apply_penalty(self, participant_id: str, violation_type: str, 
                     severity: float, justification: str) -> str:
        """
        Apply penalty to participant
        
        Args:
            participant_id: Participant ID
            violation_type: Type of violation
            severity: Severity (0.0 to 1.0)
            justification: Justification for penalty
            
        Returns:
            penalty_id: Unique penalty ID
        """
        current_reputation = self.reputation_manager.get_reputation(participant_id)
        penalty_amount, calc_justification = self.incentive_calculator.calculate_penalty(
            participant_id, violation_type, severity, current_reputation
        )
        
        penalty_id = hashlib.sha256(
            f"penalty:{participant_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        with self.lock:
            self.penalties[penalty_id] = {
                'participant_id': participant_id,
                'violation_type': violation_type,
                'severity': severity,
                'penalty_amount': penalty_amount,
                'justification': f"{justification} - {calc_justification}",
                'timestamp': time.time()
            }
        
        # Update reputation with penalty
        self.reputation_manager.update_reputation(
            participant_id, 0.0, 0.0, penalty_amount
        )
        
        logger.warning(f"Applied penalty {penalty_id} to {participant_id}: {penalty_amount:.2f}")
        return penalty_id
    
    def get_participant_summary(self, participant_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for participant"""
        with self.lock:
            # Get contributions
            contributions = [c for c in self.contributions.values() if c.contributor_id == participant_id]
            
            # Get rewards
            rewards = [r for r in self.rewards.values() if r.recipient_id == participant_id]
            
            # Get penalties
            penalties = [p for p in self.penalties.values() if p['participant_id'] == participant_id]
            
            # Get reputation history
            reputation_history = self.reputation_manager.get_reputation_history(participant_id)
            
            # Get provenance records
            provenance_records = self.provenance_tracker.get_participant_provenance(participant_id)
            
            # Calculate statistics
            total_contributions = len(contributions)
            total_rewards = sum(r.amount for r in rewards)
            total_penalties = sum(p['penalty_amount'] for p in penalties)
            current_reputation = self.reputation_manager.get_reputation(participant_id)
            
            return {
                'participant_id': participant_id,
                'current_reputation': current_reputation,
                'total_contributions': total_contributions,
                'total_rewards': total_rewards,
                'total_penalties': total_penalties,
                'net_rewards': total_rewards - total_penalties,
                'reputation_history': reputation_history[-10:],  # Last 10 changes
                'recent_contributions': contributions[-5:],  # Last 5 contributions
                'recent_rewards': rewards[-5:],  # Last 5 rewards
                'provenance_records': len(provenance_records)
            }
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        with self.lock:
            # Get top participants
            top_participants = self.reputation_manager.get_top_participants(10)
            
            # Calculate system statistics
            total_contributions = len(self.contributions)
            total_rewards = len(self.rewards)
            total_penalties = len(self.penalties)
            total_participants = len(self.reputation_manager.reputation_scores)
            
            # Get provenance report
            provenance_report = self.provenance_tracker.generate_provenance_report()
            
            # Calculate average reputation
            avg_reputation = np.mean(list(self.reputation_manager.reputation_scores.values())) if self.reputation_manager.reputation_scores else 0.0
            
            report = {
                'system_statistics': {
                    'total_participants': total_participants,
                    'total_contributions': total_contributions,
                    'total_rewards': total_rewards,
                    'total_penalties': total_penalties,
                    'average_reputation': avg_reputation
                },
                'top_participants': [
                    {'participant_id': pid, 'reputation': rep} 
                    for pid, rep in top_participants
                ],
                'provenance_report': provenance_report,
                'generated_at': time.time()
            }
            
            logger.info(f"Generated system report: {total_participants} participants, {total_contributions} contributions")
            return report
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Incentive and Provenance System cleanup completed")

def main():
    """Test the incentive and provenance system"""
    logger.info("Testing Incentive and Provenance System")
    
    try:
        # Initialize system
        system = IncentiveProvenanceSystem()
        
        # Test participants
        participants = ['client_1', 'client_2', 'client_3']
        
        # Simulate contributions
        for round_num in range(1, 4):
            logger.info(f"Simulating round {round_num}")
            
            for participant in participants:
                # Create contribution
                contribution = Contribution(
                    contributor_id=participant,
                    contribution_type=ContributionType.MODEL_UPDATE,
                    round_number=round_num,
                    timestamp=time.time(),
                    value=0.8 + np.random.random() * 0.2,  # Random value between 0.8-1.0
                    quality_score=0.7 + np.random.random() * 0.3,  # Random quality between 0.7-1.0
                    metadata={'model_type': 'transductive_fewshot'},
                    verification_hash=hashlib.sha256(f"{participant}:{round_num}".encode()).hexdigest()
                )
                
                # Record contribution
                contribution_id = system.record_contribution(contribution)
                logger.info(f"Recorded contribution {contribution_id} for {participant}")
            
            # Record aggregation
            aggregation_id = system.record_aggregation(
                round_number=round_num,
                participants=participants,
                aggregation_data={'method': 'fedavg', 'clients': len(participants)}
            )
            logger.info(f"Recorded aggregation {aggregation_id}")
        
        # Test penalty
        penalty_id = system.apply_penalty(
            participant_id='client_2',
            violation_type='low_quality_model',
            severity=0.3,
            justification='Model quality below threshold'
        )
        logger.info(f"Applied penalty {penalty_id}")
        
        # Generate reports
        for participant in participants:
            summary = system.get_participant_summary(participant)
            logger.info(f"Participant {participant} summary: Reputation={summary['current_reputation']:.3f}, "
                       f"Rewards={summary['total_rewards']:.2f}, Contributions={summary['total_contributions']}")
        
        system_report = system.generate_system_report()
        logger.info(f"System report: {system_report['system_statistics']}")
        
        # Cleanup
        system.cleanup()
        
        logger.info("✅ Incentive and Provenance System test completed!")
        
    except Exception as e:
        logger.error(f"❌ System test failed: {str(e)}")

if __name__ == "__main__":
    main()
