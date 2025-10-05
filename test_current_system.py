#!/usr/bin/env python3
"""
Test the current system's incentive processing to see what data it produces
"""

import sys
import os
import time
from unittest.mock import MagicMock

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import BlockchainFederatedIncentiveSystem
from coordinators.blockchain_fedavg_coordinator import ClientUpdate

def test_current_system():
    """Test what the current system produces for incentives"""
    
    # Mock the configuration
    config = MagicMock()
    config.num_clients = 3
    config.epochs = 1
    config.rounds = 1
    config.log_level = "INFO"
    config.attack_type = "DoS"
    config.device = "cpu"
    
    # Create a mock system
    system = BlockchainFederatedIncentiveSystem(config)
    
    # Mock the logger
    system.logger = MagicMock()
    
    # Initialize incentive history
    system.incentive_history = []
    
    # Create mock client updates with different performances
    client_updates = [
        ClientUpdate(
            client_id='client_1',
            model_parameters={'param1': 0.1, 'param2': 0.2},
            sample_count=100,
            training_loss=0.05,
            validation_accuracy=0.9350  # Medium performance
        ),
        ClientUpdate(
            client_id='client_2', 
            model_parameters={'param1': 0.3, 'param2': 0.4},
            sample_count=100,
            training_loss=0.03,
            validation_accuracy=0.9650  # Best performance
        ),
        ClientUpdate(
            client_id='client_3',
            model_parameters={'param1': 0.5, 'param2': 0.6},
            sample_count=100,
            training_loss=0.07,
            validation_accuracy=0.9250  # Lowest performance
        )
    ]
    
    # Test the incentive processing
    current_accuracy = 0.95
    previous_accuracy = 0.85
    
    print("=== TESTING CURRENT SYSTEM INCENTIVE PROCESSING ===")
    print(f"Client performances:")
    for update in client_updates:
        print(f"  {update.client_id}: {update.validation_accuracy}")
    
    print(f"\nCurrent accuracy: {current_accuracy}")
    print(f"Previous accuracy: {previous_accuracy}")
    
    # Call the incentive processing method
    try:
        system._process_round_incentives(client_updates, current_accuracy, previous_accuracy, 1)
        
        print(f"\nIncentive history after processing:")
        for record in system.incentive_history:
            print(f"  Round {record['round_number']}: {record['total_rewards']} tokens")
            if 'shapley_values' in record:
                print(f"    Shapley values: {record['shapley_values']}")
            else:
                print("    No Shapley values found!")
        
        # Test the incentive summary
        summary = system.get_incentive_summary()
        print(f"\nIncentive summary:")
        print(f"  Total rewards: {summary.get('total_rewards_distributed', 0)}")
        print(f"  Participant rewards: {summary.get('participant_rewards', {})}")
        
        # Check if rewards are different
        participant_rewards = summary.get('participant_rewards', {})
        if participant_rewards:
            rewards_list = list(participant_rewards.values())
            if len(set(rewards_list)) > 1:
                print("✅ SUCCESS: Different token amounts for each client!")
                print(f"   Range: {min(rewards_list):.2f} - {max(rewards_list):.2f}")
            else:
                print("❌ FAILURE: All clients have the same token amount!")
        else:
            print("❌ FAILURE: No participant rewards found!")
            
    except Exception as e:
        print(f"❌ Error during incentive processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_current_system()
