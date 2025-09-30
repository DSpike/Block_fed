#!/usr/bin/env python3
"""
Test ERC20 token distribution
"""

import json
from web3 import Web3

def test_erc20_distribution():
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Ganache")
        return
    
    # Load deployment info
    with open('erc20_deployment.json', 'r') as f:
        deployment = json.load(f)
    
    token_address = deployment['contracts']['FederatedLearningToken']['address']
    token_abi = deployment['contracts']['FederatedLearningToken']['abi']
    
    # Create contract instance
    token_contract = w3.eth.contract(address=token_address, abi=token_abi)
    
    # Test simple token transfer instead
    recipient = w3.eth.accounts[1]
    amount = 1000 * 10**18  # 1000 tokens (with 18 decimals)
    
    print(f"🚀 Testing ERC20 token transfer...")
    print(f"From: {w3.eth.accounts[0]}")
    print(f"To: {recipient}")
    print(f"Amount: {amount / 10**18:.2f} FLT")
    
    try:
        # Check initial balance
        initial_balance = token_contract.functions.balanceOf(recipient).call()
        print(f"Initial balance: {initial_balance / 10**18:.2f} FLT")
        
        # Build transaction
        tx = token_contract.functions.transfer(
            recipient,
            amount
        ).build_transaction({
            'from': w3.eth.accounts[0],
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0])
        })
        
        # Send transaction
        tx_hash = w3.eth.send_transaction(tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            print(f"✅ Token transfer successful!")
            print(f"Gas used: {receipt.gasUsed}")
            print(f"Block: {receipt.blockNumber}")
            
            # Check final balance
            final_balance = token_contract.functions.balanceOf(recipient).call()
            print(f"Final balance: {final_balance / 10**18:.2f} FLT")
            print(f"Transfer amount: {(final_balance - initial_balance) / 10**18:.2f} FLT")
        else:
            print("❌ Transaction failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_erc20_distribution()
