#!/usr/bin/env python3
"""
Deployment Script for Blockchain-Enabled Federated Learning with Incentive Mechanisms
Handles smart contract deployment, participant registration, and system initialization
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from web3 import Web3
from eth_account import Account
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainSystemDeployer:
    """
    Deployer for blockchain-enabled federated learning system
    """
    
    def __init__(self, rpc_url: str, private_key: str):
        """
        Initialize deployer
        
        Args:
            rpc_url: Ethereum RPC URL
            private_key: Private key for deployment
        """
        self.rpc_url = rpc_url
        self.private_key = private_key
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add middleware for PoA networks
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
            self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            logger.warning("ExtraDataToPOAMiddleware not available")
        
        # Check connection
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum network: {rpc_url}")
        
        # Load account
        self.account = Account.from_key(private_key)
        self.web3.eth.default_account = self.account.address
        
        logger.info(f"Deployer initialized")
        logger.info(f"Account: {self.account.address}")
        logger.info(f"Balance: {self.web3.eth.get_balance(self.account.address)} wei")
    
    def deploy_federated_learning_contract(self, contract_bytecode: str, contract_abi: List[Dict]) -> str:
        """
        Deploy federated learning smart contract
        
        Args:
            contract_bytecode: Contract bytecode
            contract_abi: Contract ABI
            
        Returns:
            contract_address: Deployed contract address
        """
        try:
            logger.info("Deploying Federated Learning contract...")
            
            # Build deployment transaction
            contract = self.web3.eth.contract(bytecode=contract_bytecode, abi=contract_abi)
            
            # Deploy contract
            tx_hash = contract.constructor().transact({
                'from': self.account.address,
                'gas': 2000000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Wait for deployment
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                contract_address = tx_receipt.contractAddress
                logger.info(f"✅ Federated Learning contract deployed: {contract_address}")
                return contract_address
            else:
                logger.error("❌ Contract deployment failed")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Contract deployment error: {str(e)}")
            return ""
    
    def deploy_incentive_contract(self, contract_bytecode: str, contract_abi: List[Dict], 
                                aggregator_address: str) -> str:
        """
        Deploy incentive smart contract
        
        Args:
            contract_bytecode: Contract bytecode
            contract_abi: Contract ABI
            aggregator_address: Aggregator address
            
        Returns:
            contract_address: Deployed contract address
        """
        try:
            logger.info("Deploying Incentive contract...")
            
            # Build deployment transaction
            contract = self.web3.eth.contract(bytecode=contract_bytecode, abi=contract_abi)
            
            # Deploy contract with aggregator address
            tx_hash = contract.constructor(aggregator_address).transact({
                'from': self.account.address,
                'gas': 3000000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Wait for deployment
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                contract_address = tx_receipt.contractAddress
                logger.info(f"✅ Incentive contract deployed: {contract_address}")
                return contract_address
            else:
                logger.error("❌ Incentive contract deployment failed")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Incentive contract deployment error: {str(e)}")
            return ""
    
    def register_participants(self, contract_address: str, contract_abi: List[Dict], 
                            participant_addresses: List[str]) -> bool:
        """
        Register participants in the incentive contract
        
        Args:
            contract_address: Contract address
            contract_abi: Contract ABI
            participant_addresses: List of participant addresses
            
        Returns:
            success: Whether registration was successful
        """
        try:
            logger.info(f"Registering {len(participant_addresses)} participants...")
            
            # Load contract
            contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)
            
            success_count = 0
            
            for address in participant_addresses:
                try:
                    # Build transaction
                    tx = contract.functions.registerParticipant(address).build_transaction({
                        'from': self.account.address,
                        'gas': 100000,
                        'gasPrice': self.web3.eth.gas_price,
                        'nonce': self.web3.eth.get_transaction_count(self.account.address)
                    })
                    
                    # Sign and send transaction
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    
                    # Wait for transaction receipt
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    
                    if receipt.status == 1:
                        success_count += 1
                        logger.info(f"✅ Registered participant: {address}")
                    else:
                        logger.error(f"❌ Failed to register participant: {address}")
                
                except Exception as e:
                    logger.error(f"❌ Error registering {address}: {str(e)}")
            
            logger.info(f"Registration completed: {success_count}/{len(participant_addresses)} successful")
            return success_count == len(participant_addresses)
            
        except Exception as e:
            logger.error(f"❌ Participant registration error: {str(e)}")
            return False
    
    def fund_participants(self, participant_addresses: List[str], amount_wei: int) -> bool:
        """
        Fund participant addresses with ETH for gas fees
        
        Args:
            participant_addresses: List of participant addresses
            amount_wei: Amount to send in wei
            
        Returns:
            success: Whether funding was successful
        """
        try:
            logger.info(f"Funding {len(participant_addresses)} participants with {amount_wei} wei each...")
            
            success_count = 0
            
            for address in participant_addresses:
                try:
                    # Build transaction
                    tx = {
                        'to': address,
                        'value': amount_wei,
                        'gas': 21000,
                        'gasPrice': self.web3.eth.gas_price,
                        'nonce': self.web3.eth.get_transaction_count(self.account.address)
                    }
                    
                    # Sign and send transaction
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    
                    # Wait for transaction receipt
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    
                    if receipt.status == 1:
                        success_count += 1
                        logger.info(f"✅ Funded participant: {address}")
                    else:
                        logger.error(f"❌ Failed to fund participant: {address}")
                
                except Exception as e:
                    logger.error(f"❌ Error funding {address}: {str(e)}")
            
            logger.info(f"Funding completed: {success_count}/{len(participant_addresses)} successful")
            return success_count == len(participant_addresses)
            
        except Exception as e:
            logger.error(f"❌ Participant funding error: {str(e)}")
            return False
    
    def save_deployment_info(self, deployment_info: Dict[str, Any], filepath: str):
        """Save deployment information to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(deployment_info, f, indent=2, default=str)
            
            logger.info(f"Deployment info saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment info: {str(e)}")

def load_contract_artifacts(contract_name: str) -> tuple:
    """
    Load contract artifacts (bytecode and ABI)
    
    Args:
        contract_name: Name of the contract
        
    Returns:
        bytecode: Contract bytecode
        abi: Contract ABI
    """
    try:
        # In a real deployment, these would be loaded from compiled artifacts
        # For now, return dummy values
        if contract_name == "FederatedLearning":
            bytecode = "0x608060405234801561001057600080fd5b50..."  # Dummy bytecode
            abi = [
                {
                    "inputs": [],
                    "name": "constructor",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                }
            ]
        elif contract_name == "FederatedLearningIncentive":
            bytecode = "0x608060405234801561001057600080fd5b50..."  # Dummy bytecode
            abi = [
                {
                    "inputs": [{"internalType": "address", "name": "_aggregator", "type": "address"}],
                    "name": "constructor",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                }
            ]
        else:
            raise ValueError(f"Unknown contract: {contract_name}")
        
        return bytecode, abi
        
    except Exception as e:
        logger.error(f"Error loading contract artifacts: {str(e)}")
        return "", []

def main():
    """Main deployment function"""
    logger.info("🚀 Blockchain Federated Learning System Deployment")
    logger.info("=" * 60)
    
    # Configuration
    rpc_url = "http://localhost:8545"
    private_key = "0x" + "0" * 64  # Dummy private key
    
    # Participant addresses (in production, these would be real MetaMask addresses)
    participant_addresses = [
        "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
        "0x8ba1f109551bD432803012645Hac136c4c8b4d8b7",
        "0x9ca2f209661bE543904123756Hac247d5d9c5e9c8"
    ]
    
    try:
        # Initialize deployer
        deployer = BlockchainSystemDeployer(rpc_url, private_key)
        
        # Load contract artifacts
        fl_bytecode, fl_abi = load_contract_artifacts("FederatedLearning")
        incentive_bytecode, incentive_abi = load_contract_artifacts("FederatedLearningIncentive")
        
        # Deploy contracts
        fl_contract_address = deployer.deploy_federated_learning_contract(fl_bytecode, fl_abi)
        if not fl_contract_address:
            logger.error("Failed to deploy Federated Learning contract")
            return
        
        incentive_contract_address = deployer.deploy_incentive_contract(
            incentive_bytecode, incentive_abi, deployer.account.address
        )
        if not incentive_contract_address:
            logger.error("Failed to deploy Incentive contract")
            return
        
        # Fund participants (optional, for gas fees)
        funding_amount = 1000000000000000000  # 1 ETH in wei
        deployer.fund_participants(participant_addresses, funding_amount)
        
        # Register participants in incentive contract
        deployer.register_participants(incentive_contract_address, incentive_abi, participant_addresses)
        
        # Save deployment information
        deployment_info = {
            'network': {
                'rpc_url': rpc_url,
                'chain_id': deployer.web3.eth.chain_id
            },
            'contracts': {
                'federated_learning': {
                    'address': fl_contract_address,
                    'abi': fl_abi
                },
                'incentive': {
                    'address': incentive_contract_address,
                    'abi': incentive_abi
                }
            },
            'participants': participant_addresses,
            'deployer': {
                'address': deployer.account.address,
                'balance': str(deployer.web3.eth.get_balance(deployer.account.address))
            },
            'deployment_time': time.time()
        }
        
        deployer.save_deployment_info(deployment_info, 'deployment_info.json')
        
        logger.info("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Federated Learning Contract: {fl_contract_address}")
        logger.info(f"Incentive Contract: {incentive_contract_address}")
        logger.info(f"Participants: {len(participant_addresses)}")
        logger.info(f"Deployment info saved to: deployment_info.json")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")

if __name__ == "__main__":
    main()
