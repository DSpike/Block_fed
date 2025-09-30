#!/usr/bin/env python3
"""
Simple offline runner for testing preprocessing fixes
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from main import EnhancedSystemConfig, BlockchainFederatedIncentiveSystem
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from models.tcgan_ttt_model import TCGANTTTModel

def run_offline():
    """Run the system in offline mode for testing"""
    print("🚀 Running Blockchain Federated Learning System in OFFLINE MODE")
    print("=" * 70)
    
    # Create offline configuration
    config = EnhancedSystemConfig(
        # Disable blockchain components
        enable_incentives=False,
        ethereum_rpc_url="",
        ipfs_url="",
        fully_decentralized=False,
        
        # Keep other settings
        num_clients=3,
        num_rounds=3,
        zero_day_attack="Backdoor"
    )
    
    print("📊 Configuration:")
    print(f"   - Blockchain: DISABLED")
    print(f"   - Incentives: DISABLED") 
    print(f"   - Clients: {config.num_clients}")
    print(f"   - Rounds: {config.num_rounds}")
    print(f"   - Zero-day attack: {config.zero_day_attack}")
    print()
    
    try:
        # Initialize system
        system = BlockchainFederatedIncentiveSystem(config)
        
        print("🔧 Initializing system (offline mode)...")
        
        # Skip blockchain initialization for offline mode
        print("📊 Skipping blockchain components for offline testing...")
        system.is_initialized = True  # Mark as initialized to bypass blockchain checks
        
        # Initialize only the essential components
        print("🔧 Initializing essential components...")
        system.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        system.preprocessor = UNSWPreprocessor()
        system.model = TCGANTTTModel(
            input_dim=25,
            sequence_length=10,
            latent_dim=64,
            hidden_dim=128,
            num_classes=2,
            noise_dim=64
        ).to(system.device)
        
        # Initialize a simple coordinator for offline mode
        from federated_learning.blockchain_fedavg_coordinator import BlockchainFedAVGCoordinator
        system.coordinator = BlockchainFedAVGCoordinator(
            model=system.model,
            num_clients=config.num_clients,
            device=system.device
        )
        
        print("✅ Essential components initialized successfully")
        
        print("📊 Preprocessing data...")
        if not system.preprocess_data():
            print("❌ Data preprocessing failed")
            return False
        
        print("🤝 Setting up federated learning...")
        if not system.setup_federated_learning():
            print("❌ Federated learning setup failed")
            return False
        
        print("🚀 Starting TCGAN training...")
        if not system.run_tcgan_training():
            print("❌ TCGAN training failed")
            return False
        
        print("🔄 Running federated training...")
        if not system.run_federated_training_with_incentives():
            print("❌ Federated training failed")
            return False
        
        print("🎯 Evaluating zero-day detection...")
        results = system.evaluate_zero_day_detection()
        
        if results:
            print("\n✅ OFFLINE TEST COMPLETED SUCCESSFULLY!")
            print("🎯 Zero-day detection results:")
            for metric, value in results.items():
                print(f"   {metric}: {value}")
        else:
            print("❌ Zero-day detection evaluation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during offline test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_offline()
    if success:
        print("\n🎉 All tests passed! The preprocessing fix is working.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
