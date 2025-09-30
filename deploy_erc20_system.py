#!/usr/bin/env python3
"""
Deploy ERC20 Token and Incentive Contracts for Federated Learning System
"""

import json
import time
from web3 import Web3
from solcx import compile_source, install_solc

# Install Solidity compiler
install_solc('0.8.19')

# Ganache connection
GANACHE_URL = "http://localhost:8545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

def compile_contract(contract_source, contract_name):
    """Compile Solidity contract"""
    from solcx import set_solc_version
    set_solc_version('0.8.19')
    compiled_sol = compile_source(contract_source, output_values=['abi', 'bin'], solc_version='0.8.19', allow_paths='.')
    contract_interface = compiled_sol[f'<stdin>:{contract_name}']
    return contract_interface

def deploy_contract(w3, contract_interface, constructor_args=None):
    """Deploy contract to blockchain"""
    # Get the first account (should have funds in Ganache)
    account = w3.eth.accounts[0]
    
    # Build contract
    contract = w3.eth.contract(
        abi=contract_interface['abi'],
        bytecode=contract_interface['bin']
    )
    
    # Build constructor transaction
    if constructor_args:
        constructor_tx = contract.constructor(*constructor_args).build_transaction({
            'from': account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
    else:
        constructor_tx = contract.constructor().build_transaction({
            'from': account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
    
    # Deploy contract
    tx_hash = w3.eth.send_transaction(constructor_tx)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    return tx_receipt.contractAddress, contract_interface['abi']

def main():
    print("🚀 Deploying ERC20 Token and Incentive Contracts...")
    
    # Check connection
    if not w3.is_connected():
        print("❌ Failed to connect to Ganache. Make sure it's running on port 8545")
        return
    
    print(f"✅ Connected to Ganache at {GANACHE_URL}")
    print(f"📊 Network ID: {w3.eth.chain_id}")
    print(f"💰 Account balance: {w3.eth.get_balance(w3.eth.accounts[0]) / 10**18:.2f} ETH")
    
    try:
        # Read ERC20 Token contract
        with open('contracts/FederatedLearningToken.sol', 'r') as f:
            token_source = f.read()
        
        # Compile ERC20 Token contract
        print("\n📝 Compiling ERC20 Token contract...")
        token_interface = compile_contract(token_source, 'FederatedLearningToken')
        
        # Deploy ERC20 Token contract
        print("🚀 Deploying ERC20 Token contract...")
        initial_supply = 1000000  # 1 million tokens
        token_address, token_abi = deploy_contract(w3, token_interface, [initial_supply])
        print(f"✅ ERC20 Token deployed at: {token_address}")
        
        # Read Incentive contract
        with open('contracts/FederatedLearningIncentive.sol', 'r') as f:
            incentive_source = f.read()
        
        # Compile Incentive contract
        print("\n📝 Compiling Incentive contract...")
        incentive_interface = compile_contract(incentive_source, 'FederatedLearningIncentive')
        
        # Deploy Incentive contract
        print("🚀 Deploying Incentive contract...")
        aggregator_address = w3.eth.accounts[0]  # Use first account as aggregator
        incentive_address, incentive_abi = deploy_contract(
            w3, incentive_interface, [aggregator_address, token_address]
        )
        print(f"✅ Incentive contract deployed at: {token_address}")
        
        # Set incentive contract in token contract
        print("\n🔗 Setting incentive contract in token contract...")
        token_contract = w3.eth.contract(address=token_address, abi=token_abi)
        tx_hash = token_contract.functions.setIncentiveContract(incentive_address).transact({
            'from': w3.eth.accounts[0],
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0])
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Incentive contract set in token contract")
        
        # Save deployment info
        deployment_info = {
            "network": "ganache",
            "chain_id": w3.eth.chain_id,
            "deployment_time": time.time(),
            "contracts": {
                "FederatedLearningToken": {
                    "address": token_address,
                    "abi": token_abi
                },
                "FederatedLearningIncentive": {
                    "address": incentive_address,
                    "abi": incentive_abi
                }
            },
            "accounts": {
                "deployer": w3.eth.accounts[0],
                "aggregator": aggregator_address
            },
            "token_info": {
                "name": "Federated Learning Token",
                "symbol": "FLT",
                "decimals": 18,
                "initial_supply": initial_supply
            }
        }
        
        with open('erc20_deployment.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\n📄 Deployment info saved to: erc20_deployment.json")
        
        # Display summary
        print("\n🎉 DEPLOYMENT COMPLETE!")
        print("=" * 50)
        print(f"ERC20 Token Address: {token_address}")
        print(f"Incentive Contract Address: {incentive_address}")
        print(f"Initial Token Supply: {initial_supply:,} FLT")
        print(f"Deployer Account: {w3.eth.accounts[0]}")
        print("=" * 50)
        
        # Test token balance
        token_balance = token_contract.functions.balanceOf(w3.eth.accounts[0]).call()
        print(f"💰 Deployer token balance: {token_balance / 10**18:,.2f} FLT")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
