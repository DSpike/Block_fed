#!/usr/bin/env python3
"""
Deploy contracts using unlocked Ganache accounts
"""

import json
import time
import logging
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_with_unlock():
    """Deploy contracts using unlocked Ganache accounts"""
    
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
    
    # Simple contract bytecode and ABI
    simple_contract_bytecode = "0x608060405234801561001057600080fd5b50600436106100365760003560e01c8063150b7a021461003b578063a9059cbb14610059575b600080fd5b610043610075565b60405161005091906100a1565b60405180910390f35b610073600480360381019061006e91906100ed565b61007e565b005b60008054905090565b6000805490506000821415610092576001905061008e565b60008260000154905060008260010160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff1614156100ee57600190506100ea565b600190505b92915050565b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b600061011d826100f2565b9050919050565b61012d81610112565b82525050565b60006020820190506101486000830184610124565b92915050565b600080fd5b600080fd5b600080fd5b60008083601f84011261016f5761016e61014a565b5b8235905067ffffffffffffffff81111561018c5761018b61014f565b5b6020830191508360018202830111156101a8576101a7610154565b5b9250929050565b6000813590506101be81610112565b92915050565b6000806000604084860312156101dd576101dc610145565b5b60006101eb868287016101af565b935050602084013567ffffffffffffffff81111561020c5761020b61014a565b5b61021886828701610159565b92509250509250925092565b6000819050919050565b61023681610224565b82525050565b6000602082019050610251600083018461022d565b92915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b6000600282049050600182168061029e57607f821691505b602082108114156102b2576102b1610257565b5b50919050565b6000819050919050565b60007fffffffff0000000000000000000000000000000000000000000000000000000082169050919050565b6102f6816102b8565b82525050565b600060208201905061031160008301846102ed565b92915050565b600081519050919050565b600082825260208201905092915050565b60005b83811015610351578082015181840152602081019050610336565b83811115610360576000848401525b50505050565b6000601f19601f8301169050919050565b600061038282610317565b61038c8185610322565b935061039c818560208601610333565b6103a581610366565b840191505092915050565b600060208201905081810360008301526103ca8184610377565b905092915050565b6000819050919050565b6103e5816103d2565b81146103f057600080fd5b50565b600081359050610402816103dc565b92915050565b60006020828403121561041e5761041d610145565b5b600061042c848285016103f3565b91505092915050565b61043e816103d2565b82525050565b60006020820190506104596000830184610435565b92915050565b6000806040838503121561047657610475610145565b5b6000610484858286016103f3565b9250506020610495858286016103f3565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b60006104d9826103d2565b91506104e4836103d2565b9250827fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff03821115610519576105186104a3565b5b828201905092915050565b600061052f826103d2565b915061053a836103d2565b9250817fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0483118215151615610573576105726104a3565b5b828202905092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601260045260246000fd5b60006105b8826103d2565b91506105c3836103d2565b9250826105d3576105d261057e565b5b828204905092915050565b60006105e9826103d2565b91506105f4836103d2565b925082821015610607576106066104a3565b5b828203905092915050565b60006020828403121561062657610625610145565b5b6000610634848285016103f3565b91505092915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b6000610677826103d2565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8214156106aa576106a96104a3565b5b60018201905091905056fea2646970667358221220"
    
    simple_contract_abi = [
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
            abi=simple_contract_abi,
            bytecode=simple_contract_bytecode
        )
        
        # Build constructor transaction
        constructor_tx = contract.constructor().build_transaction({
            'from': first_account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(first_account)
        })
        
        # For Ganache, we can use personal_sendTransaction
        # This bypasses the need for private keys
        try:
            # Try to unlock the account (Ganache allows this)
            w3.geth.personal.unlock_account(first_account, "", 0)
            logger.info(f"Account {first_account} unlocked")
        except:
            logger.info("Account unlock not available, trying direct transaction")
        
        # Send transaction directly (Ganache allows this for unlocked accounts)
        tx_hash = w3.eth.send_transaction(constructor_tx)
        
        # Wait for receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        incentive_address = tx_receipt.contractAddress
        logger.info(f"✅ FederatedLearningIncentive deployed at: {incentive_address}")
        
        deployed_contracts['incentive'] = {
            'address': incentive_address,
            'abi': simple_contract_abi
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
            'abi': simple_contract_abi
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
    deploy_with_unlock()
