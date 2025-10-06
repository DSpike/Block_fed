#!/usr/bin/env python3
"""
Decentralized Blockchain Client for Federated Learning
Integrates with smart contracts for consensus and model management
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from web3 import Web3
from eth_account import Account
import hashlib

logger = logging.getLogger(__name__)

class DecentralizedBlockchainClient:
    """
    Blockchain client for decentralized federated learning
    Handles smart contract interactions and consensus
    """
    
    def __init__(self, rpc_url: str, private_key: str, contract_address: str, contract_abi: Dict):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = Account.from_key(private_key)
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        
        # Load contract
        self.contract = self.web3.eth.contract(
            address=contract_address,
            abi=contract_abi
        )
        
        # Gas settings
        self.gas_price = self.web3.eth.gas_price
        self.gas_limit = 500000
        
        logger.info(f"Decentralized blockchain client initialized for {self.account.address}")
    
    def register_miner(self, stake: int) -> bool:
        """Register miner with stake"""
        try:
            # Convert stake to wei
            stake_wei = self.web3.to_wei(stake, 'ether')
            
            # Build transaction
            transaction = self.contract.functions.registerMiner(stake_wei).build_transaction({
                'from': self.account.address,
                'value': stake_wei,
                'gas': self.gas_limit,
                'gasPrice': self.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Miner registered successfully: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Miner registration failed: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register miner: {e}")
            return False
    
    def submit_proposal(self, model_hash: str, validation_score: int) -> Optional[str]:
        """Submit model aggregation proposal"""
        try:
            # Convert model hash to bytes32
            model_hash_bytes = self.web3.to_bytes(hexstr=model_hash)
            
            # Build transaction
            transaction = self.contract.functions.submitProposal(
                model_hash_bytes,
                validation_score
            ).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': self.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Proposal submitted successfully: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                logger.error(f"Proposal submission failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to submit proposal: {e}")
            return None
    
    def vote_on_proposal(self, proposal_hash: str, vote: bool, confidence: int) -> bool:
        """Vote on a proposal"""
        try:
            # Convert proposal hash to bytes32
            proposal_hash_bytes = self.web3.to_bytes(hexstr=proposal_hash)
            
            # Build transaction
            transaction = self.contract.functions.voteOnProposal(
                proposal_hash_bytes,
                vote,
                confidence
            ).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': self.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Vote cast successfully: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Vote casting failed: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to vote on proposal: {e}")
            return False
    
    def check_consensus(self, proposal_hash: str) -> Tuple[int, int, bool]:
        """Check consensus status for a proposal"""
        try:
            # Convert proposal hash to bytes32
            proposal_hash_bytes = self.web3.to_bytes(hexstr=proposal_hash)
            
            # Call contract function
            result = self.contract.functions.getConsensusStatus(proposal_hash_bytes).call()
            
            consensus_ratio, total_votes, is_consensus_reached = result
            
            logger.info(f"Consensus status: {consensus_ratio}% agreement, {total_votes} votes, consensus: {is_consensus_reached}")
            
            return consensus_ratio, total_votes, is_consensus_reached
            
        except Exception as e:
            logger.error(f"Failed to check consensus: {e}")
            return 0, 0, False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            result = self.contract.functions.getSystemStats().call()
            
            total_miners, total_stake, current_round = result
            
            return {
                "total_miners": total_miners,
                "total_stake": self.web3.from_wei(total_stake, 'ether'),
                "current_round": current_round
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_miner_info(self, miner_address: str) -> Dict[str, Any]:
        """Get miner information"""
        try:
            result = self.contract.functions.getMinerInfo(miner_address).call()
            
            stake, reputation, is_active = result
            
            return {
                "stake": self.web3.from_wei(stake, 'ether'),
                "reputation": reputation,
                "is_active": is_active
            }
            
        except Exception as e:
            logger.error(f"Failed to get miner info: {e}")
            return {}
    
    def listen_for_events(self, event_name: str, callback):
        """Listen for blockchain events"""
        try:
            # Get event filter
            event_filter = self.contract.events[event_name].create_filter(
                fromBlock='latest'
            )
            
            # Process events
            for event in event_filter.get_new_entries():
                callback(event)
                
        except Exception as e:
            logger.error(f"Failed to listen for events: {e}")
    
    def calculate_model_hash(self, model_params: Dict[str, Any]) -> str:
        """Calculate hash of model parameters"""
        # Serialize model parameters
        param_str = json.dumps(model_params, sort_keys=True, default=str)
        
        # Calculate SHA256 hash
        return hashlib.sha256(param_str.encode()).hexdigest()
    
    def get_balance(self) -> float:
        """Get account balance in ETH"""
        try:
            balance_wei = self.web3.eth.get_balance(self.account.address)
            return self.web3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

class DecentralizedConsensusManager:
    """
    Manages consensus process for decentralized federated learning
    """
    
    def __init__(self, blockchain_client: DecentralizedBlockchainClient):
        self.blockchain_client = blockchain_client
        self.active_proposals: Dict[str, Dict] = {}
        self.consensus_timeout = 300  # 5 minutes
        
    def submit_aggregation_proposal(self, model_params: Dict[str, Any], validation_score: int) -> Optional[str]:
        """Submit aggregation proposal to blockchain"""
        try:
            # Calculate model hash
            model_hash = self.blockchain_client.calculate_model_hash(model_params)
            
            # Submit proposal
            tx_hash = self.blockchain_client.submit_proposal(model_hash, validation_score)
            
            if tx_hash:
                # Store proposal info
                self.active_proposals[model_hash] = {
                    "model_params": model_params,
                    "validation_score": validation_score,
                    "tx_hash": tx_hash,
                    "timestamp": time.time(),
                    "votes": []
                }
                
                logger.info(f"Proposal submitted: {model_hash}")
                return model_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to submit aggregation proposal: {e}")
            return None
    
    def vote_on_proposal(self, proposal_hash: str, vote: bool, confidence: int) -> bool:
        """Vote on a proposal"""
        try:
            success = self.blockchain_client.vote_on_proposal(proposal_hash, vote, confidence)
            
            if success:
                # Update local proposal info
                if proposal_hash in self.active_proposals:
                    self.active_proposals[proposal_hash]["votes"].append({
                        "vote": vote,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })
                
                logger.info(f"Vote cast on proposal {proposal_hash}: {vote}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to vote on proposal: {e}")
            return False
    
    def check_consensus_status(self, proposal_hash: str) -> Tuple[int, int, bool]:
        """Check consensus status for a proposal"""
        try:
            consensus_ratio, total_votes, is_consensus_reached = self.blockchain_client.check_consensus(proposal_hash)
            
            logger.info(f"Consensus status for {proposal_hash}: {consensus_ratio}% agreement, {total_votes} votes")
            
            return consensus_ratio, total_votes, is_consensus_reached
            
        except Exception as e:
            logger.error(f"Failed to check consensus status: {e}")
            return 0, 0, False
    
    def wait_for_consensus(self, proposal_hash: str, timeout: int = None) -> bool:
        """Wait for consensus to be reached on a proposal"""
        if timeout is None:
            timeout = self.consensus_timeout
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            consensus_ratio, total_votes, is_consensus_reached = self.check_consensus_status(proposal_hash)
            
            if is_consensus_reached:
                logger.info(f"Consensus reached on proposal {proposal_hash}")
                return True
            
            time.sleep(5)  # Check every 5 seconds
        
        logger.warning(f"Consensus timeout for proposal {proposal_hash}")
        return False
    
    def get_winning_proposal(self, round_number: int) -> Optional[Dict]:
        """Get winning proposal for a round"""
        try:
            # This would need to be implemented in the smart contract
            # For now, return the proposal with highest consensus
            best_proposal = None
            best_ratio = 0
            
            for proposal_hash, proposal_info in self.active_proposals.items():
                consensus_ratio, _, is_consensus_reached = self.check_consensus_status(proposal_hash)
                
                if is_consensus_reached and consensus_ratio > best_ratio:
                    best_ratio = consensus_ratio
                    best_proposal = proposal_info
            
            return best_proposal
            
        except Exception as e:
            logger.error(f"Failed to get winning proposal: {e}")
            return None

def main():
    """Test the decentralized blockchain client"""
    # Mock contract ABI (in real implementation, load from file)
    contract_abi = [
        {
            "inputs": [{"name": "stake", "type": "uint256"}],
            "name": "registerMiner",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        }
    ]
    
    # Initialize client
    client = DecentralizedBlockchainClient(
        rpc_url="http://localhost:8545",
        private_key="0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
        contract_address="0x1234567890123456789012345678901234567890",
        contract_abi=contract_abi
    )
    
    # Test functions
    print(f"Account balance: {client.get_balance()} ETH")
    print(f"System stats: {client.get_system_stats()}")

if __name__ == "__main__":
    main()
