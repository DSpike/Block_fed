#!/usr/bin/env python3
"""
Test Different Attack Types for Zero-Day Detection
This script allows easy testing of the blockchain federated learning system
with different attack types as zero-day attacks.
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main, EnhancedSystemConfig

def get_available_attack_types():
    """Get all available attack types from the UNSW-NB15 dataset"""
    print("🔍 Analyzing available attack types in UNSW-NB15 dataset...")
    
    # Load datasets
    train_df = pd.read_csv("UNSW_NB15_training-set.csv")
    test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
    
    # Get attack types from both datasets
    train_attacks = train_df['attack_cat'].value_counts()
    test_attacks = test_df['attack_cat'].value_counts()
    
    print("\n📊 Attack types available in TRAINING data:")
    for attack, count in train_attacks.items():
        if attack != 'Normal':
            print(f"   {attack}: {count:,} samples")
    
    print("\n📊 Attack types available in TEST data:")
    for attack, count in test_attacks.items():
        if attack != 'Normal':
            print(f"   {attack}: {count:,} samples")
    
    # Calculate total samples per attack type
    print("\n🎯 Total samples per attack type (Training + Test):")
    all_attacks = set(train_attacks.index) | set(test_attacks.index)
    attack_totals = {}
    
    for attack in all_attacks:
        if attack != 'Normal':
            train_count = train_attacks.get(attack, 0)
            test_count = test_attacks.get(attack, 0)
            total_count = train_count + test_count
            attack_totals[attack] = total_count
            print(f"   {attack}: {total_count:,} samples (Train: {train_count:,}, Test: {test_count:,})")
    
    return attack_totals

def run_experiment_with_attack_type(attack_type: str):
    """Run the blockchain federated learning experiment with specified attack type"""
    print(f"\n🚀 Running blockchain federated learning with '{attack_type}' as zero-day attack")
    print("=" * 80)
    
    # Create configuration with specified attack type
    config = EnhancedSystemConfig(
        num_clients=3,
        num_rounds=3,
        local_epochs=5,
        learning_rate=0.01,
        
        # ✅ ALL BLOCKCHAIN COMPONENTS ENABLED
        enable_incentives=True,
        base_reward=100,
        max_reward=1000,
        min_reputation=0.5,
        
        # ✅ Blockchain services enabled
        ethereum_rpc_url="http://localhost:8545",
        ipfs_url="http://localhost:5001",
        
        # ✅ Fully decentralized mode enabled
        fully_decentralized=True,
        
        # 🎯 SPECIFIED ATTACK TYPE
        zero_day_attack=attack_type
    )
    
    try:
        # Run the main system
        from main import BlockchainFederatedIncentiveSystem
        
        system = BlockchainFederatedIncentiveSystem(config)
        
        # Initialize system
        print(f"\n🔧 Initializing system for {attack_type} zero-day detection...")
        if not system.initialize_system():
            print("❌ System initialization failed")
            return False
        
        # Preprocess data
        print(f"\n📊 Preprocessing data with {attack_type} holdout...")
        if not system.preprocess_data():
            print("❌ Data preprocessing failed")
            return False
        
        # Setup federated learning
        print(f"\n🤝 Setting up federated learning...")
        if not system.setup_federated_learning():
            print("❌ Federated learning setup failed")
            return False
        
        # Run TCGAN training
        print(f"\n🚀 Starting TCGAN training...")
        if not system.run_tcgan_training():
            print("❌ TCGAN training failed")
            return False
        
        # Run federated training
        print(f"\n🔄 Running federated training...")
        if not system.run_federated_training_with_incentives():
            print("❌ Federated training failed")
            return False
        
        # Evaluate zero-day detection
        print(f"\n🎯 Evaluating zero-day detection for {attack_type}...")
        evaluation_results = system.evaluate_zero_day_detection()
        
        if evaluation_results:
            print(f"\n✅ {attack_type} Zero-Day Detection Results:")
            if 'base_model' in evaluation_results and 'ttt_model' in evaluation_results:
                base = evaluation_results['base_model']
                ttt = evaluation_results['ttt_model']
                print(f"   Base Model - Accuracy: {base.get('accuracy', 0):.4f}, F1: {base.get('f1_score', 0):.4f}")
                print(f"   TTT Model  - Accuracy: {ttt.get('accuracy', 0):.4f}, F1: {ttt.get('f1_score', 0):.4f}")
                print(f"   Improvement - Accuracy: +{ttt.get('accuracy', 0) - base.get('accuracy', 0):.4f}")
        
        print(f"\n🎉 {attack_type} experiment completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running {attack_type} experiment: {e}")
        return False

def main_menu():
    """Main menu for attack type selection"""
    print("🎯 Blockchain Federated Learning - Attack Type Testing")
    print("=" * 60)
    
    # Get available attack types
    attack_totals = get_available_attack_types()
    
    # Sort by sample count (descending)
    sorted_attacks = sorted(attack_totals.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Available attack types for zero-day detection:")
    for i, (attack, count) in enumerate(sorted_attacks, 1):
        print(f"   {i}. {attack}: {count:,} total samples")
    
    print(f"\n💡 Recommendations:")
    print(f"   - High sample count: Better statistical significance")
    print(f"   - Low sample count: More challenging zero-day detection")
    
    # Get user choice
    try:
        choice = input(f"\n🔧 Select attack type (1-{len(sorted_attacks)}) or 'all' to test all: ").strip()
        
        if choice.lower() == 'all':
            print(f"\n🚀 Testing ALL attack types...")
            results = {}
            for attack, count in sorted_attacks:
                print(f"\n{'='*60}")
                print(f"Testing {attack} ({count:,} samples)")
                print(f"{'='*60}")
                success = run_experiment_with_attack_type(attack)
                results[attack] = success
                if not success:
                    print(f"❌ {attack} experiment failed, continuing with next...")
            
            # Summary
            print(f"\n🎯 SUMMARY - All Attack Types:")
            for attack, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"   {attack}: {status}")
        
        elif choice.isdigit() and 1 <= int(choice) <= len(sorted_attacks):
            attack_type = sorted_attacks[int(choice) - 1][0]
            run_experiment_with_attack_type(attack_type)
        
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print(f"\n👋 Experiment cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main_menu()


