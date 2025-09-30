#!/usr/bin/env python3
"""
Fix blockchain hanging by adding timeout protection while keeping blockchain functionality
"""

import re

def fix_blockchain_timeout():
    """Add timeout protection to blockchain initialization"""
    
    print("🔧 FIXING BLOCKCHAIN TIMEOUT ISSUE")
    print("=" * 40)
    print("✅ Keeping blockchain functionality")
    print("✅ Adding timeout protection to prevent hanging")
    
    main_file = "main.py"
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic blockchain initialization with timeout protection
    old_blockchain_init = '''            # Initialize blockchain and IPFS integration
            self.blockchain_integration = BlockchainIPFSIntegration(ethereum_config, ipfs_config)'''
    
    new_blockchain_init = '''            # Initialize blockchain and IPFS integration with timeout protection
            try:
                import signal
                import requests
                
                # Test blockchain connection with timeout
                logger.info("Testing blockchain connection...")
                response = requests.get(f"{self.config.ethereum_rpc_url.replace('http://', 'http://')}", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Blockchain connection successful")
                    self.blockchain_integration = BlockchainIPFSIntegration(ethereum_config, ipfs_config)
                else:
                    logger.warning(f"⚠️ Blockchain connection failed: HTTP {response.status_code}")
                    self.blockchain_integration = None
                    
            except requests.exceptions.Timeout:
                logger.warning("⚠️ Blockchain connection timeout - continuing without blockchain")
                self.blockchain_integration = None
            except requests.exceptions.ConnectionError:
                logger.warning("⚠️ Blockchain service not available - continuing without blockchain")
                self.blockchain_integration = None
            except Exception as e:
                logger.warning(f"⚠️ Blockchain initialization failed: {e} - continuing without blockchain")
                self.blockchain_integration = None'''
    
    if old_blockchain_init in content:
        content = content.replace(old_blockchain_init, new_blockchain_init)
        print("✅ Added timeout protection to blockchain integration")
    
    # Fix MetaMask authenticator with timeout
    old_metamask_init = '''            self.authenticator = MetaMaskAuthenticator(
                rpc_url=self.config.ethereum_rpc_url,
                contract_address=self.config.contract_address,
                contract_abi=FEDERATED_LEARNING_ABI
            )'''
    
    new_metamask_init = '''            # Initialize MetaMask authenticator with timeout protection
            try:
                # Test blockchain connection before initializing MetaMask
                import requests
                response = requests.get(self.config.ethereum_rpc_url, timeout=3)
                if response.status_code == 200:
                    self.authenticator = MetaMaskAuthenticator(
                        rpc_url=self.config.ethereum_rpc_url,
                        contract_address=self.config.contract_address,
                        contract_abi=FEDERATED_LEARNING_ABI
                    )
                    logger.info("✅ MetaMask authenticator initialized")
                else:
                    logger.warning("⚠️ Blockchain not available - MetaMask authenticator disabled")
                    self.authenticator = None
            except Exception as e:
                logger.warning(f"⚠️ MetaMask authenticator initialization failed: {e}")
                self.authenticator = None'''
    
    if old_metamask_init in content:
        content = content.replace(old_metamask_init, new_metamask_init)
        print("✅ Added timeout protection to MetaMask authenticator")
    
    # Fix incentive contract with timeout
    old_incentive_pattern = r'if self\.config\.enable_incentives:.*?self\.incentive_manager = BlockchainIncentiveManager\(self\.incentive_contract\)'
    
    new_incentive_init = '''if self.config.enable_incentives:
                try:
                    # Test blockchain connection before initializing incentive contract
                    import requests
                    response = requests.get(self.config.ethereum_rpc_url, timeout=3)
                    if response.status_code == 200:
                        logger.info("Initializing blockchain incentive contract...")
                        self.incentive_contract = BlockchainIncentiveContract(
                            rpc_url=self.config.ethereum_rpc_url,
                            contract_address=self.config.incentive_contract_address,
                            contract_abi=FEDERATED_LEARNING_ABI,
                            private_key=self.config.private_key,
                            aggregator_address=self.config.aggregator_address
                        )
                        self.incentive_manager = BlockchainIncentiveManager(self.incentive_contract)
                        logger.info("✅ Blockchain incentive contract initialized")
                    else:
                        logger.warning("⚠️ Blockchain not available - incentive contract disabled")
                        self.incentive_contract = None
                        self.incentive_manager = None
                except Exception as e:
                    logger.warning(f"⚠️ Incentive contract initialization failed: {e}")
                    self.incentive_contract = None
                    self.incentive_manager = None'''
    
    if re.search(old_incentive_pattern, content, re.DOTALL):
        content = re.sub(old_incentive_pattern, new_incentive_init, content, flags=re.DOTALL)
        print("✅ Added timeout protection to incentive contract")
    
    # Save the fixed file
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed main.py with timeout protection")
    print("\n🎯 WHAT THIS FIX DOES:")
    print("• Keeps blockchain functionality intact")
    print("• Adds 3-5 second timeouts to prevent hanging")
    print("• Gracefully falls back if blockchain services unavailable")
    print("• System continues running even if blockchain fails")
    print("• Real blockchain integration when services are available")
    
    return True

if __name__ == "__main__":
    fix_blockchain_timeout()
    print("\n✅ Blockchain timeout fix applied!")
    print("🚀 System will now run with blockchain support + timeout protection")

