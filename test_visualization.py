#!/usr/bin/env python3
"""
Test Visualization Data
This script tests the visualization with proper data structure
"""

import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def test_visualization():
    """Test visualization with proper data structure"""
    
    # Create visualizer
    visualizer = PerformanceVisualizer(
        output_dir="test_plots", 
        attack_name="Generic"
    )
    
    # Create sample confusion matrix data
    confusion_matrix_data = np.array([[15000, 3871], [100, 18771]])
    
    print("🧪 Testing Confusion Matrix Visualization...")
    
    # Test confusion matrix
    try:
        plot_path = visualizer.plot_confusion_matrix(
            confusion_matrix_data, 
            class_names=['Normal', 'Attack'],
            save=True,
            show=True
        )
        print(f"✅ Confusion matrix test successful: {plot_path}")
    except Exception as e:
        print(f"❌ Confusion matrix test failed: {e}")
    
    # Test performance comparison with proper data structure
    print("\n🧪 Testing Performance Comparison Visualization...")
    
    base_results = {
        'binary_metrics': {
            'accuracy': 0.5390,
            'precision': 0.5204,
            'recall': 0.9950,
            'f1_score': 0.6834,
            'roc_auc': 0.3469
        }
    }
    
    ttt_results = {
        'binary_metrics': {
            'accuracy': 0.9647,
            'precision': 0.9749,
            'recall': 0.9541,
            'f1_score': 0.9644,
            'roc_auc': 0.8500  # Mock ROC-AUC for TTT
        }
    }
    
    try:
        comparison_path = visualizer.plot_performance_comparison_with_annotations(
            base_results,
            ttt_results,
            scenario_names=['Binary Classification', 'Test-Time Training'],
            save=True
        )
        print(f"✅ Performance comparison test successful: {comparison_path}")
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
    
    print("\n🎉 Visualization testing completed!")

if __name__ == "__main__":
    print("🧪 Testing Visualization Data Structure")
    print("=" * 50)
    test_visualization()

