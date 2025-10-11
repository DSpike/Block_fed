#!/usr/bin/env python3
"""
Test script for enhanced reward calculation in ThresholdAgent
"""

import torch
import numpy as np
from models.transductive_fewshot_model import ThresholdAgent

def test_enhanced_reward_calculation():
    """Test the enhanced reward calculation with various scenarios"""
    
    print("🧪 Testing Enhanced Reward Calculation in ThresholdAgent...")
    
    # Create agent
    agent = ThresholdAgent()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Perfect Performance",
            "params": {
                "adaptation_success_rate": 1.0,
                "accuracy_improvement": 0.3,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 50,
                "true_negatives": 50,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "samples_selected": 20,
                "total_samples": 100,
                "threshold": 0.5
            }
        },
        {
            "name": "High False Positives",
            "params": {
                "adaptation_success_rate": 0.8,
                "accuracy_improvement": 0.2,
                "false_positives": 20,
                "false_negatives": 5,
                "true_positives": 30,
                "true_negatives": 45,
                "precision": 0.6,
                "recall": 0.86,
                "f1_score": 0.71,
                "samples_selected": 15,
                "total_samples": 100,
                "threshold": 0.3
            }
        },
        {
            "name": "High False Negatives",
            "params": {
                "adaptation_success_rate": 0.6,
                "accuracy_improvement": 0.1,
                "false_positives": 5,
                "false_negatives": 25,
                "true_positives": 20,
                "true_negatives": 50,
                "precision": 0.8,
                "recall": 0.44,
                "f1_score": 0.57,
                "samples_selected": 60,
                "total_samples": 100,
                "threshold": 0.8
            }
        },
        {
            "name": "Balanced Performance",
            "params": {
                "adaptation_success_rate": 0.75,
                "accuracy_improvement": 0.15,
                "false_positives": 8,
                "false_negatives": 7,
                "true_positives": 35,
                "true_negatives": 50,
                "precision": 0.81,
                "recall": 0.83,
                "f1_score": 0.82,
                "samples_selected": 30,
                "total_samples": 100,
                "threshold": 0.5
            }
        },
        {
            "name": "Poor Selection Efficiency",
            "params": {
                "adaptation_success_rate": 0.5,
                "accuracy_improvement": 0.05,
                "false_positives": 15,
                "false_negatives": 10,
                "true_positives": 25,
                "true_negatives": 50,
                "precision": 0.63,
                "recall": 0.71,
                "f1_score": 0.67,
                "samples_selected": 80,
                "total_samples": 100,
                "threshold": 0.1
            }
        }
    ]
    
    print("\n📊 Reward Calculation Results:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        
        # Calculate reward
        reward = agent._calculate_reward(**test_case['params'])
        
        # Get detailed breakdown
        breakdown = agent.get_reward_breakdown(**test_case['params'])
        
        # Display results
        print(f"Total Reward: {reward:.2f}")
        print(f"Selection Ratio: {breakdown['selection_ratio']:.2f}")
        print(f"Precision: {test_case['params']['precision']:.2f}")
        print(f"Recall: {test_case['params']['recall']:.2f}")
        print(f"F1-Score: {test_case['params']['f1_score']:.2f}")
        print(f"False Positives: {test_case['params']['false_positives']}")
        print(f"False Negatives: {test_case['params']['false_negatives']}")
        
        print("\nReward Breakdown:")
        for component, value in breakdown.items():
            if component not in ['total_reward', 'selection_ratio']:
                print(f"  {component}: {value:.2f}")
    
    print("\n✅ Enhanced reward calculation test completed!")
    
    # Test the update method
    print("\n🔄 Testing update method with enhanced metrics...")
    
    state = torch.tensor([0.5, 0.7], dtype=torch.float32)
    threshold = 0.4
    
    # Test with comprehensive metrics
    agent.update(
        state, threshold, 0.8, 0.2,
        false_positives=5, false_negatives=3,
        true_positives=40, true_negatives=52,
        precision=0.89, recall=0.93, f1_score=0.91,
        samples_selected=25, total_samples=100
    )
    
    print(f"Memory size: {len(agent.memory)}")
    print(f"Adaptation history size: {len(agent.adaptation_history)}")
    print("✅ Update method test completed!")

if __name__ == "__main__":
    test_enhanced_reward_calculation()
