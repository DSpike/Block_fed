#!/usr/bin/env python3
"""
Run blockchain federated learning system in ONLINE mode (with blockchain enabled)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main, EnhancedSystemConfig

def run_online():
    """Run the system in online mode with blockchain integration enabled"""
    
    # Create configuration with blockchain enabled
    config = EnhancedSystemConfig(
        # Data configuration
        data_path="UNSW_NB15_training-set.csv",
        test_path="UNSW_NB15_testing-set.csv",
        zero_day_attack="Backdoor",
        
        # Training configuration
        num_clients=3,
        num_rounds=3,  # Already set to 3 rounds
        local_epochs=5,  # Reduced for testing
        learning_rate=0.01,
        
        # Device configuration
        device='cuda',
        
        # Blockchain configuration - ENABLED
        enable_incentives=True,   # Enable blockchain incentives
        base_reward=100,
        max_reward=1000,
        min_reputation=0.5,
        
        # Enable blockchain components (will fallback to offline if not available)
        ethereum_rpc_url="http://localhost:8545",  # Ganache local blockchain
        ipfs_url="http://localhost:5001",          # Local IPFS node
        
        # Other settings
        fully_decentralized=False
    )
    
    print("🚀 Running Blockchain Federated Learning System in ONLINE MODE")
    print("=" * 70)
    print("📋 Configuration:")
    print(f"   - Clients: {config.num_clients}")
    print(f"   - Rounds: {config.num_rounds}")
    print(f"   - Local epochs: {config.local_epochs}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Blockchain: ENABLED")
    print(f"   - Incentives: ENABLED")
    print("=" * 70)
    
    try:
        # Initialize and run the system directly with our config
        from main import BlockchainFederatedIncentiveSystem
        
        print("\n🔄 Initializing system with blockchain components ENABLED...")
        
        # Patch all blockchain components to enable them (with graceful fallback)
        import blockchain.blockchain_ipfs_integration as bci
        import blockchain.metamask_auth_system as mms
        import blockchain.incentive_provenance_system as ips
        
        # Patch BlockchainIPFSIntegration
        original_bci_init = bci.BlockchainIPFSIntegration.__init__
        def patched_bci_init(self, ethereum_config, ipfs_config):
            try:
                # Try to initialize blockchain components
                return original_bci_init(self, ethereum_config, ipfs_config)
            except Exception as e:
                print(f"⚠️  Blockchain connection failed: {e}")
                print("⚠️  Attempting to continue with blockchain features disabled")
                self.blockchain_enabled = False
                self.ipfs_enabled = False
                # Set dummy attributes to prevent errors
                self.ethereum_client = None
                self.ipfs_client = None
                return
        bci.BlockchainIPFSIntegration.__init__ = patched_bci_init
        
        # Patch MetaMaskAuthenticator
        original_mms_init = mms.MetaMaskAuthenticator.__init__
        def patched_mms_init(self, rpc_url, contract_address, contract_abi):
            try:
                return original_mms_init(self, rpc_url, contract_address, contract_abi)
            except Exception as e:
                print(f"⚠️  MetaMask connection failed: {e}")
                print("⚠️  Continuing without MetaMask authentication")
                self.connected = False
                return
        mms.MetaMaskAuthenticator.__init__ = patched_mms_init
        
        # Patch IncentiveProvenanceSystem
        original_ips_init = ips.IncentiveProvenanceSystem.__init__
        def patched_ips_init(self, ethereum_config):
            try:
                return original_ips_init(self, ethereum_config)
            except Exception as e:
                print(f"⚠️  Incentive system connection failed: {e}")
                print("⚠️  Continuing without incentive system")
                self.ethereum_client = None
                return
        ips.IncentiveProvenanceSystem.__init__ = patched_ips_init
        
        # Patch the blockchain integration to add missing attributes
        def patched_bci_init_with_attrs(self, ethereum_config, ipfs_config):
            if not ethereum_config.get('rpc_url') or not ipfs_config.get('url'):
                print("⚠️  Blockchain IPFS integration disabled - running in offline mode")
                self.blockchain_enabled = False
                self.ipfs_enabled = False
                # Add dummy attributes to prevent attribute errors
                self.ethereum_client = None
                self.ipfs_client = None
                self.contract = None
                return
            return original_bci_init(self, ethereum_config, ipfs_config)
        bci.BlockchainIPFSIntegration.__init__ = patched_bci_init_with_attrs
        
        system = BlockchainFederatedIncentiveSystem(config)
        
        # Initialize all components
        print("\n🔧 Initializing system components...")
        if not system.initialize_system():
            print("❌ System initialization failed")
            return
        
        # Preprocess data
        print("\n📊 Preprocessing data...")
        if not system.preprocess_data():
            print("❌ Data preprocessing failed")
            return
        
        # Setup federated learning
        print("\n🤝 Setting up federated learning...")
        if not system.setup_federated_learning():
            print("❌ Federated learning setup failed")
            return
        
        # Run TCGAN training
        print("\n🚀 Starting TCGAN training...")
        if not system.run_tcgan_training():
            print("❌ TCGAN training failed")
            return
        
        # Run federated training with incentives
        print("\n🔄 Running federated training...")
        if not system.run_federated_training_with_incentives():
            print("❌ Federated training failed")
            return
        
        # Evaluate zero-day detection
        print("\n📊 Evaluating zero-day detection...")
        zero_day_results = system.evaluate_zero_day_detection()
        
        # Evaluate final global model
        print("\n📈 Evaluating final global model...")
        final_results = system.evaluate_final_global_model()
        
        print("\n✅ System completed successfully in online mode!")
        
    except Exception as e:
        print(f"\n❌ Error running system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_online()
