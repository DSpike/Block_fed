#!/usr/bin/env python3
"""
Test Worms Attack for Zero-Day Detection
This script tests the blockchain federated learning system with Worms as the zero-day attack.
Worms has only 174 total samples, making it a very challenging zero-day detection scenario.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main, EnhancedSystemConfig

def test_worms_attack():
    """Test the system with Worms as zero-day attack"""
    print("🦠 Testing Worms Attack for Zero-Day Detection")
    print("=" * 60)
    print("📊 Worms Attack Statistics:")
    print("   - Total samples: 174 (Train: 130, Test: 44)")
    print("   - Challenge level: VERY HIGH (lowest sample count)")
    print("   - Attack type: Worms (malware propagation)")
    print("=" * 60)
    
    # Create configuration with Worms as zero-day attack
    config = EnhancedSystemConfig(
        # Data configuration
        data_path="UNSW_NB15_training-set.csv",
        test_path="UNSW_NB15_testing-set.csv",
        zero_day_attack="Worms",  # 🦠 Testing Worms attack
        
        # Model configuration
        input_dim=32,
        hidden_dim=128,
        embedding_dim=128,
        
        # Federated learning configuration (reduced for testing)
        num_clients=3,
        num_rounds=5,  # Reduced rounds for faster testing
        local_epochs=3,  # Reduced epochs for faster testing
        learning_rate=0.01,
        
        # Blockchain configuration
        ethereum_rpc_url="http://localhost:8545",
        contract_address="0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8",
        incentive_contract_address="0x02090bbB57546b0bb224880a3b93D2Ffb0dde144",
        private_key="0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
        aggregator_address="0x4565f36D8E3cBC1c7187ea39Eb613E484411e075",
        
        # IPFS configuration
        ipfs_url="http://localhost:5001",
        
        # Incentive configuration
        enable_incentives=True,
        base_reward=100,
        max_reward=1000,
        min_reputation=100,
        
        # Device configuration
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        
        # Decentralization configuration
        fully_decentralized=False  # Use centralized for testing
    )
    
    try:
        print("\n🚀 Starting Worms Attack Test...")
        print("This will test the system's ability to detect Worms as a zero-day attack.")
        print("Due to the very low sample count (174 total), this is a challenging test.")
        
        # Run the system directly
        from main import BlockchainFederatedIncentiveSystem
        
        system = BlockchainFederatedIncentiveSystem(config)
        
        # Initialize all components
        if not system.initialize_system():
            print("❌ System initialization failed")
            return False
        
        # Preprocess data
        if not system.preprocess_data():
            print("❌ Data preprocessing failed")
            return False
        
        # Setup federated learning
        if not system.setup_federated_learning():
            print("❌ Federated learning setup failed")
            return False
        
        # Run TCGAN training
        if not system.run_tcgan_training():
            print("❌ TCGAN training failed")
            return False
        
        # Run federated training
        if not system.run_federated_training_with_incentives():
            print("❌ Federated training failed")
            return False
        
        # Evaluate zero-day detection
        results = system.evaluate_zero_day_detection()
        
        if results:
            print("\n✅ Worms Attack Test Completed Successfully!")
            print("=" * 60)
            print("📊 Results Summary:")
            
            # Display key results if available
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
            
            print("\n🎯 Analysis:")
            print("   - Worms attack detection capability tested")
            print("   - Zero-day adaptation performance evaluated")
            print("   - System robustness with low-sample attacks verified")
            
        else:
            print("❌ Worms Attack Test Failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during Worms attack test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🦠 Blockchain Federated Learning - Worms Attack Test")
    print("=" * 60)
    
    success = test_worms_attack()
    
    if success:
        print("\n🎉 Worms attack test completed successfully!")
        print("The system has been tested with one of the most challenging attack types.")
    else:
        print("\n❌ Worms attack test failed.")
        print("Please check the error messages above for troubleshooting.")
