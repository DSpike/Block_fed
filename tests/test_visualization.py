#!/usr/bin/env python3
"""
Test script for performance visualization functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.performance_visualization import PerformanceVisualizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_visualization():
    """Test the performance visualization functionality"""
    
    # Initialize visualizer
    visualizer = PerformanceVisualizer(output_dir="test_plots")
    
    # Create sample data
    sample_data = {
        'training_history': {
            'epoch_losses': [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
            'epoch_accuracies': [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
        },
        'round_results': [
            {'round': 1, 'accuracy': 0.8, 'avg_loss': 0.2, 'num_clients': 3, 'model_hash': 'abc12345'},
            {'round': 2, 'accuracy': 0.85, 'avg_loss': 0.15, 'num_clients': 3, 'model_hash': 'def67890'},
            {'round': 3, 'accuracy': 0.9, 'avg_loss': 0.1, 'num_clients': 3, 'model_hash': 'ghi13579'},
            {'round': 4, 'accuracy': 0.92, 'avg_loss': 0.08, 'num_clients': 3, 'model_hash': 'jkl24680'},
            {'round': 5, 'accuracy': 0.95, 'avg_loss': 0.05, 'num_clients': 3, 'model_hash': 'mno97531'}
        ],
        'evaluation_results': {
            'accuracy': 0.5,
            'precision': 0.4,
            'recall': 0.3,
            'f1_score': 0.35,
            'roc_auc': 0.6,
            'zero_day_detection_rate': 0.0,
            'avg_confidence': 0.0,
            'num_zero_day_samples': 0,
            'total_samples': 74000
        },
        'client_results': [
            {'client_id': 'client_1', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3},
            {'client_id': 'client_2', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3},
            {'client_id': 'client_3', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3}
        ],
        'blockchain_data': {
            'transactions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'ipfs_cids': ['Qm1', 'Qm2', 'Qm3', 'Qm4', 'Qm5', 'Qm6', 'Qm7', 'Qm8', 'Qm9', 'Qm10'],
            'gas_used': [21000, 25000, 23000, 24000, 22000, 26000, 28000, 27000, 25000, 24000],
            'block_numbers': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }
    }
    
    logger.info("Testing performance visualization functionality...")
    
    # Test individual plots
    logger.info("1. Testing training history plot...")
    visualizer.plot_training_history(sample_data['training_history'])
    
    logger.info("2. Testing federated rounds plot...")
    visualizer.plot_federated_rounds(sample_data['round_results'])
    
    logger.info("3. Testing zero-day detection metrics plot...")
    visualizer.plot_zero_day_detection_metrics(sample_data['evaluation_results'])
    
    logger.info("4. Testing client performance plot...")
    visualizer.plot_client_performance(sample_data['client_results'])
    
    logger.info("5. Testing blockchain metrics plot...")
    visualizer.plot_blockchain_metrics(sample_data['blockchain_data'])
    
    logger.info("6. Testing comprehensive report...")
    visualizer.create_comprehensive_report(sample_data)
    
    logger.info("7. Testing JSON export...")
    visualizer.save_metrics_to_json(sample_data)
    
    logger.info("8. Testing enhanced annotations...")
    visualizer.plot_training_trends_with_annotations(sample_data['training_history'])
    
    # Create mock base results for comparison
    base_results = {
        'accuracy': 0.4,
        'precision': 0.35,
        'recall': 0.25,
        'f1_score': 0.29
    }
    
    visualizer.plot_performance_comparison_with_annotations(
        base_results, sample_data['evaluation_results']
    )
    
    logger.info("✅ All visualization tests completed successfully!")
    logger.info("Check the 'test_plots' directory for generated plots.")

if __name__ == "__main__":
    test_visualization()
