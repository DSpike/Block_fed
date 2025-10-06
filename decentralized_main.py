#!/usr/bin/env python3
"""
Main Decentralized Federated Learning System
Integrates 2 miners with blockchain consensus mechanism
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Optional, Tuple
from decentralized_fl_system import DecentralizedFederatedLearningSystem, ModelUpdate
from blockchain.decentralized_blockchain_client import DecentralizedBlockchainClient, DecentralizedConsensusManager
from models.transductive_fewshot_model import TransductiveLearner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecentralizedBlockchainFLSystem:
    """
    Complete decentralized federated learning system with blockchain consensus
    """
    
    def __init__(self, model: nn.Module, num_clients: int = 3, 
                 rpc_url: str = "http://localhost:8545",
                 contract_address: str = "0x1234567890123456789012345678901234567890"):
        
        self.model = model
        self.num_clients = num_clients
        self.current_round = 0
        
        # Initialize decentralized FL system
        self.fl_system = DecentralizedFederatedLearningSystem(model, num_clients)
        
        # Initialize blockchain clients for 2 miners
        self.miner_clients = {
            "miner_1": DecentralizedBlockchainClient(
                rpc_url=rpc_url,
                private_key="0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
                contract_address=contract_address,
                contract_abi=self._load_contract_abi()
            ),
            "miner_2": DecentralizedBlockchainClient(
                rpc_url=rpc_url,
                private_key="0x6cbed15c793ce57650b9877cf6fa156fbef513c4e6134f022a85b1ffdd5b2c1",
                contract_address=contract_address,
                contract_abi=self._load_contract_abi()
            )
        }
        
        # Initialize consensus managers
        self.consensus_managers = {
            miner_id: DecentralizedConsensusManager(client)
            for miner_id, client in self.miner_clients.items()
        }
        
        # Register miners on blockchain
        self._register_miners()
        
        logger.info("Decentralized Blockchain FL System initialized")
    
    def _load_contract_abi(self) -> List[Dict]:
        """Load smart contract ABI"""
        # Complete ABI for testing
        return [
            {
                "inputs": [{"name": "stake", "type": "uint256"}],
                "name": "registerMiner",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [{"name": "modelHash", "type": "bytes32"}, {"name": "validationScore", "type": "uint256"}],
                "name": "submitProposal",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "proposalHash", "type": "bytes32"}, {"name": "vote", "type": "bool"}, {"name": "confidence", "type": "uint256"}],
                "name": "voteOnProposal",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "minerAddress", "type": "address"}],
                "name": "getMinerInfo",
                "outputs": [
                    {"name": "stake", "type": "uint256"},
                    {"name": "reputation", "type": "uint256"},
                    {"name": "isActive", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getSystemStats",
                "outputs": [
                    {"name": "totalMiners", "type": "uint256"},
                    {"name": "totalStakeAmount", "type": "uint256"},
                    {"name": "currentRoundNumber", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _register_miners(self):
        """Register both miners on blockchain"""
        for miner_id, client in self.miner_clients.items():
            try:
                success = client.register_miner(stake=1000)  # 1000 ETH stake
                if success:
                    logger.info(f"Miner {miner_id} registered successfully")
                else:
                    logger.error(f"Failed to register miner {miner_id}")
            except Exception as e:
                logger.error(f"Error registering miner {miner_id}: {e}")
    
    def add_client_update(self, update: ModelUpdate) -> bool:
        """Add client update to the system"""
        return self.fl_system.add_client_update(update)
    
    def run_decentralized_round(self) -> Dict:
        """Run one decentralized federated learning round with blockchain consensus"""
        try:
            logger.info(f"Starting decentralized round {self.current_round + 1}")
            start_time = time.time()
            
            # Step 1: Run local FL aggregation
            fl_result = self.fl_system.run_decentralized_round()
            
            if not fl_result.get("success", False):
                logger.error("FL aggregation failed")
                return {"success": False, "error": "FL aggregation failed"}
            
            # Step 2: Submit proposals to blockchain
            proposals = {}
            for miner_id, consensus_manager in self.consensus_managers.items():
                try:
                    # Get aggregated model from miner
                    miner = self.fl_system.miners[miner_id]
                    if miner.aggregation_proposals:
                        latest_proposal = max(miner.aggregation_proposals.values(), 
                                           key=lambda p: p.timestamp)
                        
                        # Submit to blockchain
                        model_hash = consensus_manager.submit_aggregation_proposal(
                            latest_proposal.aggregated_model,
                            int(latest_proposal.validation_score * 100)
                        )
                        
                        if model_hash:
                            proposals[miner_id] = {
                                "model_hash": model_hash,
                                "proposal": latest_proposal
                            }
                            logger.info(f"Miner {miner_id} submitted proposal: {model_hash}")
                
                except Exception as e:
                    logger.error(f"Failed to submit proposal for miner {miner_id}: {e}")
            
            if not proposals:
                logger.error("No proposals submitted to blockchain")
                return {"success": False, "error": "No blockchain proposals"}
            
            # Step 3: Cross-vote on proposals
            voting_results = {}
            for proposer_id, proposal_info in proposals.items():
                for voter_id, consensus_manager in self.consensus_managers.items():
                    if voter_id != proposer_id:  # Don't vote on own proposal
                        try:
                            # Evaluate proposal quality
                            proposal = proposal_info["proposal"]
                            quality_score = self._evaluate_proposal_quality(proposal)
                            
                            # Vote based on quality
                            vote = quality_score > 0.7
                            confidence = int(abs(quality_score - 0.5) * 200)  # 0-100 scale
                            
                            # Submit vote to blockchain
                            success = consensus_manager.vote_on_proposal(
                                proposal_info["model_hash"],
                                vote,
                                confidence
                            )
                            
                            if success:
                                voting_results[f"{voter_id}_votes_{proposer_id}"] = {
                                    "vote": vote,
                                    "confidence": confidence
                                }
                                logger.info(f"Miner {voter_id} voted {vote} on {proposer_id}'s proposal")
                        
                        except Exception as e:
                            logger.error(f"Failed to vote for miner {voter_id}: {e}")
            
            # Step 4: Wait for consensus
            consensus_reached = False
            winning_proposal = None
            
            for proposer_id, proposal_info in proposals.items():
                try:
                    consensus_manager = self.consensus_managers[proposer_id]
                    
                    # Wait for consensus
                    if consensus_manager.wait_for_consensus(proposal_info["model_hash"], timeout=60):
                        consensus_reached = True
                        winning_proposal = proposal_info
                        logger.info(f"Consensus reached for miner {proposer_id}'s proposal")
                        break
                
                except Exception as e:
                    logger.error(f"Failed to check consensus for miner {proposer_id}: {e}")
            
            if not consensus_reached:
                logger.warning("No consensus reached on blockchain, using local FL result")
                winning_proposal = {"proposal": fl_result.get("winning_proposal")}
            
            # Step 5: Update global model
            if winning_proposal and "proposal" in winning_proposal:
                self._update_global_model(winning_proposal["proposal"].aggregated_model)
            
            # Step 6: Update miner reputations
            self._update_miner_reputations(proposals, voting_results)
            
            round_time = time.time() - start_time
            self.current_round += 1
            
            logger.info(f"Decentralized round {self.current_round} completed in {round_time:.2f}s")
            
            return {
                "success": True,
                "round": self.current_round,
                "consensus_reached": consensus_reached,
                "winning_proposal": winning_proposal["model_hash"] if winning_proposal else None,
                "round_time": round_time,
                "voting_results": voting_results
            }
            
        except Exception as e:
            logger.error(f"Decentralized round failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _evaluate_proposal_quality(self, proposal) -> float:
        """Evaluate the quality of an aggregation proposal"""
        # Simple quality evaluation based on validation score
        base_score = proposal.validation_score
        
        # Add some randomness to simulate different miner perspectives
        import random
        variation = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score + variation))
    
    def _update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update the global model with aggregated parameters"""
        try:
            self.model.load_state_dict(aggregated_params)
            logger.info("Global model updated with consensus-agreed parameters")
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
    
    def _update_miner_reputations(self, proposals: Dict, voting_results: Dict):
        """Update miner reputations based on consensus results"""
        try:
            # Update reputations in FL system
            for miner_id, miner in self.fl_system.miners.items():
                # Check if this miner's proposal won
                won_proposal = any(
                    miner_id in proposal_info.get("model_hash", "") 
                    for proposal_info in proposals.values()
                )
                
                # Update reputation
                miner.update_reputation(won_proposal)
                
                logger.info(f"Miner {miner_id} reputation: {miner.reputation:.2f}")
        
        except Exception as e:
            logger.error(f"Failed to update miner reputations: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        fl_status = self.fl_system.get_system_status()
        
        # Add blockchain status
        blockchain_status = {}
        for miner_id, client in self.miner_clients.items():
            try:
                blockchain_status[miner_id] = {
                    "balance": client.get_balance(),
                    "miner_info": client.get_miner_info(client.account.address)
                }
            except Exception as e:
                blockchain_status[miner_id] = {"error": str(e)}
        
        return {
            "fl_system": fl_status,
            "blockchain": blockchain_status,
            "current_round": self.current_round
        }

def main():
    """Test the decentralized blockchain federated learning system"""
    # Create test model
    model = TransductiveLearner(input_dim=30, hidden_dim=128, embedding_dim=64, num_classes=2)
    
    # Initialize decentralized system
    system = DecentralizedBlockchainFLSystem(model, num_clients=3)
    
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
