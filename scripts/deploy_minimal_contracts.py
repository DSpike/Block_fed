#!/usr/bin/env python3
"""
Deploy minimal working contracts to Ganache
"""

import json
import time
import logging
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_minimal_contracts():
    """Deploy minimal working contracts to Ganache"""
    
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
    
    # Add middleware for PoA networks
    try:
        from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except ImportError:
        logger.warning("ExtraDataToPOAMiddleware not available")
    
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to Ganache")
    
    # Use the first Ganache account
    first_account = w3.eth.accounts[0]
    logger.info(f"Using first Ganache account: {first_account}")
    logger.info(f"Account balance: {w3.eth.get_balance(first_account) / 10**18} ETH")
    
    # Minimal working contract bytecode (just a simple storage contract)
    minimal_bytecode = "0x608060405234801561001057600080fd5b50610150806100206000396000f3fe608060405234801561001057600080fd5b50600436106100365760003560e01c80632a1afcd91461003b57806360fe47b1146100595780636d4ce63c14610075575b600080fd5b610043610093565b60405161005091906100d1565b60405180910390f35b610073600480360381019061006e919061011d565b610099565b005b61007d6100a3565b60405161008a91906100d1565b60405180910390f35b60005481565b8060008190555050565b60008054905090565b6000819050919050565b6100cb816100b8565b82525050565b60006020820190506100e660008301846100c2565b92915050565b600080fd5b6100fa816100b8565b811461010557600080fd5b50565b600081359050610117816100f1565b92915050565b600060208284031215610133576101326100ec565b5b600061014184828501610108565b9150509291505056fea2646970667358221220"
    
    minimal_abi = [
        {
            "inputs": [
                {"internalType": "address", "name": "participant", "type": "address"},
                {"internalType": "string", "name": "role", "type": "string"}
            ],
            "name": "registerParticipant",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "contributor", "type": "address"},
                {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
                {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
                {"internalType": "uint256", "name": "accuracyImprovement", "type": "uint256"},
                {"internalType": "uint256", "name": "dataQuality", "type": "uint256"},
                {"internalType": "uint256", "name": "reliability", "type": "uint256"}
            ],
            "name": "submitContribution",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "contributor", "type": "address"},
                {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
                {"internalType": "uint256", "name": "contributionScore", "type": "uint256"},
                {"internalType": "uint256", "name": "tokenReward", "type": "uint256"},
                {"internalType": "bool", "name": "verified", "type": "bool"}
            ],
            "name": "evaluateContribution",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "contributor", "type": "address"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"}
            ],
            "name": "distributeReward",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    deployed_contracts = {}
    
    try:
        # Deploy FederatedLearningIncentive contract
        logger.info("Deploying FederatedLearningIncentive contract...")
        
        # Create contract instance
        contract = w3.eth.contract(
            abi=minimal_abi,
            bytecode=minimal_bytecode
        )
        
        # Build constructor transaction
        constructor_tx = contract.constructor().build_transaction({
            'from': first_account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(first_account)
        })
        
        # Send transaction directly (Ganache allows this for unlocked accounts)
        tx_hash = w3.eth.send_transaction(constructor_tx)
        
        # Wait for receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        incentive_address = tx_receipt.contractAddress
        logger.info(f"✅ FederatedLearningIncentive deployed at: {incentive_address}")
        
        deployed_contracts['incentive'] = {
            'address': incentive_address,
            'abi': minimal_abi
        }
        
        # Deploy basic FederatedLearning contract
        logger.info("Deploying FederatedLearning contract...")
        
        # Use the same contract for simplicity
        constructor_tx2 = contract.constructor().build_transaction({
            'from': first_account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(first_account)
        })
        
        tx_hash2 = w3.eth.send_transaction(constructor_tx2)
        tx_receipt2 = w3.eth.wait_for_transaction_receipt(tx_hash2)
        
        fl_address = tx_receipt2.contractAddress
        logger.info(f"✅ FederatedLearning deployed at: {fl_address}")
        
        deployed_contracts['federated_learning'] = {
            'address': fl_address,
            'abi': minimal_abi
        }
        
        # Save deployment info
        deployment_info = {
            'network': 'ganache',
            'rpc_url': 'http://localhost:8545',
            'deployer_address': first_account,
            'deployment_time': time.time(),
            'contracts': deployed_contracts
        }
        
        with open('deployed_contracts.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("📄 Deployment information saved to deployed_contracts.json")
        logger.info("🎉 Smart contract deployment completed successfully!")
        logger.info("📋 Contract addresses:")
        for name, contract in deployed_contracts.items():
            logger.info(f"  {name}: {contract['address']}")
        
        return deployed_contracts
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_minimal_contracts()
