#!/usr/bin/env python3
"""
Deploy complete contracts with all required functions
"""

import json
import time
import logging
import os
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compile_contract(contract_source):
    """Compile Solidity contract"""
    try:
        # Install and set Solidity compiler version
        install_solc('0.8.19')
        set_solc_version('0.8.19')
        
        # Compile contract
        compiled_sol = compile_source(contract_source, output_values=['abi', 'bin'])
        
        # Get contract interface
        contract_id, contract_interface = compiled_sol.popitem()
        
        return contract_interface
    except Exception as e:
        logger.error(f"Contract compilation failed: {str(e)}")
        raise

def deploy_complete_contracts():
    """Deploy complete contracts with all required functions"""
    
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
    
    # Read contract source
    contract_path = 'contracts/CompleteFederatedLearning.sol'
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Contract file not found: {contract_path}")
    
    with open(contract_path, 'r') as f:
        contract_source = f.read()
    
    # Compile contract
    logger.info("Compiling CompleteFederatedLearning contract...")
    contract_interface = compile_contract(contract_source)
    
    logger.info("✅ Contract compiled successfully")
    logger.info(f"Contract ABI has {len(contract_interface['abi'])} functions")
    
    deployed_contracts = {}
    
    try:
        # Deploy FederatedLearningIncentive contract
        logger.info("Deploying CompleteFederatedLearning contract...")
        
        # Create contract instance
        contract = w3.eth.contract(
            abi=contract_interface['abi'],
            bytecode=contract_interface['bin']
        )
        
        # Build constructor transaction
        constructor_tx = contract.constructor().build_transaction({
            'from': first_account,
            'gas': 3000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(first_account)
        })
        
        # Send transaction directly (Ganache allows this for unlocked accounts)
        tx_hash = w3.eth.send_transaction(constructor_tx)
        
        # Wait for receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status == 1:
            incentive_address = tx_receipt.contractAddress
            logger.info(f"✅ CompleteFederatedLearning deployed at: {incentive_address}")
            
            deployed_contracts['incentive'] = {
                'address': incentive_address,
                'abi': contract_interface['abi']
            }
            
            # Deploy second instance for federated_learning
            logger.info("Deploying second CompleteFederatedLearning contract...")
            
            constructor_tx2 = contract.constructor().build_transaction({
                'from': first_account,
                'gas': 3000000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(first_account)
            })
            
            tx_hash2 = w3.eth.send_transaction(constructor_tx2)
            tx_receipt2 = w3.eth.wait_for_transaction_receipt(tx_hash2)
            
            if tx_receipt2.status == 1:
                fl_address = tx_receipt2.contractAddress
                logger.info(f"✅ Second CompleteFederatedLearning deployed at: {fl_address}")
                
                deployed_contracts['federated_learning'] = {
                    'address': fl_address,
                    'abi': contract_interface['abi']
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
                logger.info("🎉 Complete smart contract deployment completed successfully!")
                logger.info("📋 Contract addresses:")
                for name, contract in deployed_contracts.items():
                    logger.info(f"  {name}: {contract['address']}")
                
                # Test the deployed contracts
                logger.info("🧪 Testing deployed contracts...")
                test_deployed_contracts(w3, deployed_contracts, first_account)
                
                return deployed_contracts
            else:
                logger.error("❌ Second contract deployment failed")
                return None
        else:
            logger.error("❌ Contract deployment failed")
            return None
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        raise

def test_deployed_contracts(w3, deployed_contracts, test_account):
    """Test the deployed contracts"""
    try:
        # Test incentive contract
        incentive_contract = w3.eth.contract(
            address=deployed_contracts['incentive']['address'],
            abi=deployed_contracts['incentive']['abi']
        )
        
        # Test registerParticipant function
        tx = incentive_contract.functions.registerParticipant(test_account, "test").build_transaction({
            'from': test_account,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(test_account)
        })
        
        tx_hash = w3.eth.send_transaction(tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            logger.info("✅ registerParticipant function test passed")
        else:
            logger.error("❌ registerParticipant function test failed")
        
        # Test submitModelUpdate function
        test_hash = b'\x00' * 32
        test_cid = b'\x01' * 32
        
        tx2 = incentive_contract.functions.submitModelUpdate(test_hash, test_cid, 1).build_transaction({
            'from': test_account,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(test_account)
        })
        
        tx_hash2 = w3.eth.send_transaction(tx2)
        receipt2 = w3.eth.wait_for_transaction_receipt(tx_hash2)
        
        if receipt2.status == 1:
            logger.info("✅ submitModelUpdate function test passed")
        else:
            logger.error("❌ submitModelUpdate function test failed")
        
        logger.info("🎉 Contract testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Contract testing failed: {str(e)}")

if __name__ == "__main__":
    deploy_complete_contracts()