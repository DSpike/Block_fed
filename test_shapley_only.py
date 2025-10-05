#!/usr/bin/env python3
"""
Test script to verify Shapley values are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from incentives.shapley_value_calculator import ShapleyValueCalculator

def test_shapley_with_real_data():
    """Test Shapley value calculation with real client performance data"""
    
    # Create calculator
    calculator = ShapleyValueCalculator(num_clients=3, evaluation_metric='accuracy')
    
    # Use the real performance values from the logs (Round 6)
    global_performance = 0.95
    individual_performances = {
        'client_1': 0.9350,  # Medium performing client (from Round 6 logs)
        'client_2': 0.9650,  # Best performing client (from Round 6 logs)
        'client_3': 0.9250   # Lower performing client (from Round 6 logs)
    }
    
    client_data_quality = {
        'client_1': 87.5,
        'client_2': 90.0,
        'client_3': 92.5
    }
    
    client_participation = {
        'client_1': 95.0,
        'client_2': 94.0,
        'client_3': 93.0
    }
    
    # Calculate Shapley values
    contributions = calculator.calculate_shapley_values(
        global_performance=global_performance,
        individual_performances=individual_performances,
        client_data_quality=client_data_quality,
        client_participation=client_participation
    )
    
    print("Shapley Value Results with Real Data:")
    for contrib in contributions:
        print(f"  {contrib.client_id}: Shapley={contrib.shapley_value:.4f}")
    
    # Check if values are different
    shapley_values = [c.shapley_value for c in contributions]
    if len(set(shapley_values)) > 1:
        print("✅ SUCCESS: Shapley values are different!")
        print(f"   Range: {min(shapley_values):.4f} - {max(shapley_values):.4f}")
        
        # Calculate token rewards
        token_rewards = calculator.calculate_token_rewards(contributions, total_tokens=1000)
        print("\nToken Rewards (1000 total tokens):")
        for client_id, tokens in token_rewards.items():
            print(f"  {client_id}: {tokens} tokens")
        
        # Check if token rewards are different
        token_values = list(token_rewards.values())
        if len(set(token_values)) > 1:
            print("✅ SUCCESS: Token rewards are different!")
            print(f"   Range: {min(token_values)} - {max(token_values)} tokens")
        else:
            print("❌ FAILED: All token rewards are the same!")
    else:
        print("❌ FAILED: All Shapley values are the same!")
    
    return contributions, token_rewards

if __name__ == "__main__":
    contributions, token_rewards = test_shapley_with_real_data()
