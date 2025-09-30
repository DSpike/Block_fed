#!/usr/bin/env python3
"""
Update the federated learning system to use the deployed ERC20 token
"""

import json
import os

def update_main_py():
    """Update main.py to use the deployed ERC20 token"""
    
    # Read deployment info
    with open('erc20_deployment.json', 'r') as f:
        deployment = json.load(f)
    
    token_address = deployment['contracts']['FederatedLearningToken']['address']
    token_abi = deployment['contracts']['FederatedLearningToken']['abi']
    
    print(f"🔗 Updating system to use ERC20 token at: {token_address}")
    
    # Update the blockchain incentive contract to use the deployed token
    incentive_file = 'blockchain/blockchain_incentive_contract.py'
    
    if os.path.exists(incentive_file):
        with open(incentive_file, 'r') as f:
            content = f.read()
        
        # Add token contract initialization
        token_init = f'''
    def __init__(self, web3_url: str = "http://localhost:8545", 
                 contract_address: str = None, 
                 private_key: str = None,
                 token_contract_address: str = "{token_address}"):
        """
        Initialize blockchain incentive contract
        
        Args:
            web3_url: Web3 provider URL
            contract_address: Incentive contract address
            private_key: Private key for signing transactions
            token_contract_address: ERC20 token contract address
        """
        self.web3 = Web3(Web3.HTTPProvider(web3_url))
        self.token_contract_address = token_contract_address
        
        # Load ERC20 token contract
        with open('erc20_deployment.json', 'r') as f:
            token_deployment = json.load(f)
        
        self.token_abi = token_deployment['contracts']['FederatedLearningToken']['abi']
        self.token_contract = self.web3.eth.contract(
            address=token_contract_address,
            abi=self.token_abi
        )
        
        # Initialize incentive contract (if available)
        if contract_address:
            self.contract_address = contract_address
            # Load incentive contract ABI (simplified for now)
            self.contract_abi = []  # Will be updated when incentive contract is deployed
            self.contract = self.web3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )
        else:
            self.contract = None
        
        # Account setup
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)
        else:
            # Use first Ganache account
            self.account = self.web3.eth.accounts[0]
        
        # Aggregator address (for funded account mode)
        self.aggregator_address = self.web3.eth.accounts[0]
        self.use_funded_account_mode = True
'''
        
        # Replace the existing __init__ method
        import re
        pattern = r'def __init__\(self[^)]*\):[^}]*}'
        content = re.sub(pattern, token_init, content, flags=re.DOTALL)
        
        # Add direct ERC20 token distribution method
        erc20_method = '''
    def distribute_erc20_tokens(self, recipients: list, amounts: list) -> bool:
        """
        Distribute ERC20 tokens directly to recipients
        
        Args:
            recipients: List of recipient addresses
            amounts: List of token amounts
            
        Returns:
            success: Whether distribution was successful
        """
        try:
            if not recipients or not amounts:
                logger.warning("No recipients or amounts to distribute")
                return False
            
            # Use the funded account for transactions
            from_address = self.aggregator_address if self.use_funded_account_mode else self.account
            
            # Call ERC20 distributeRewards function
            tx = self.token_contract.functions.distributeRewards(
                recipients,
                amounts
            ).build_transaction({
                'from': from_address,
                'gas': 500000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(from_address)
            })
            
            # Send transaction
            tx_hash = self.web3.eth.send_transaction(tx)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                total_tokens = sum(amounts)
                logger.info(f"ERC20 tokens distributed: {len(recipients)} recipients, {total_tokens} total tokens")
                logger.info(f"Gas used: {receipt.gasUsed}, Block: {receipt.blockNumber}")
                return True
            else:
                logger.error("ERC20 token distribution transaction failed")
                return False
                
        except Exception as e:
            logger.error(f"Error distributing ERC20 tokens: {str(e)}")
            return False
'''
        
        # Add the new method before the last class method
        content = content.replace(
            '    def complete_aggregation_round(self, round_number: int, average_accuracy: float) -> bool:',
            erc20_method + '\n    def complete_aggregation_round(self, round_number: int, average_accuracy: float) -> bool:'
        )
        
        # Write updated content
        with open(incentive_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {incentive_file}")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
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
    
    # Test token distribution
    recipients = [w3.eth.accounts[1], w3.eth.accounts[2]]
    amounts = [1000, 2000]  # 1000 and 2000 tokens
    
    print(f"🚀 Testing ERC20 token distribution...")
    print(f"Recipients: {recipients}")
    print(f"Amounts: {amounts}")
    
    try:
        # Build transaction
        tx = token_contract.functions.distributeRewards(
            recipients,
            amounts
        ).build_transaction({
            'from': w3.eth.accounts[0],
            'gas': 500000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0])
        })
        
        # Send transaction
        tx_hash = w3.eth.send_transaction(tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            print(f"✅ Token distribution successful!")
            print(f"Gas used: {receipt.gasUsed}")
            print(f"Block: {receipt.blockNumber}")
            
            # Check balances
            for i, recipient in enumerate(recipients):
                balance = token_contract.functions.balanceOf(recipient).call()
                print(f"💰 {recipient}: {balance / 10**18:.2f} FLT")
        else:
            print("❌ Transaction failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_erc20_distribution()
'''
    
    with open('test_erc20_distribution.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_erc20_distribution.py")
    print("\n🎉 ERC20 Integration Update Complete!")
    print("=" * 50)
    print(f"Token Address: {token_address}")
    print("Next steps:")
    print("1. Run: python test_erc20_distribution.py")
    print("2. Update main.py to use the new ERC20 distribution")
    print("=" * 50)

if __name__ == "__main__":
    update_main_py()

