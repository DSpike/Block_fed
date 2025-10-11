#!/usr/bin/env python3
"""
Test IEEE plots with real data from latest run
"""

from ieee_statistical_plots import IEEEStatisticalVisualizer
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ieee_with_real_data():
    # Load real results from the latest run
    with open('performance_plots/performance_metrics_latest.json', 'r') as f:
        real_results = json.load(f)

    print('📊 Testing IEEE plots with REAL data from latest run...')
    print('Real Base Model Results:')
    base_model = real_results.get('evaluation_results', {}).get('base_model', {})
    print(f'  Accuracy: {base_model.get("accuracy", "N/A")}')
    print(f'  F1-Score: {base_model.get("f1_score", "N/A")}')
    print(f'  MCC: {base_model.get("mcc", "N/A")}')

    print('\nReal TTT Model Results:')
    ttt_model = real_results.get('evaluation_results', {}).get('ttt_model', {})
    print(f'  Accuracy: {ttt_model.get("accuracy_mean", ttt_model.get("accuracy", "N/A"))}')
    print(f'  F1-Score: {ttt_model.get("macro_f1_mean", ttt_model.get("f1_score", "N/A"))}')
    print(f'  MCC: {ttt_model.get("mcc_mean", ttt_model.get("mcc", "N/A"))}')

    # Create visualizer and test with real data
    visualizer = IEEEStatisticalVisualizer()
    evaluation_results = real_results.get('evaluation_results', {})

    # Generate plot with real data
    plot_path = visualizer.plot_statistical_comparison(evaluation_results)
    print(f'\n✅ IEEE plot generated with REAL data: {plot_path}')

if __name__ == "__main__":
    test_ieee_with_real_data()

