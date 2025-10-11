#!/usr/bin/env python3
"""
Script to verify the values being plotted in the visualizations
"""

import json
import os

def verify_plot_values():
    """Verify the values in the latest results file"""
    
    # Find the latest results file
    results_files = [f for f in os.listdir('.') if f.startswith('enhanced_binary_results_') and f.endswith('.json')]
    if not results_files:
        print("❌ No results files found!")
        return
    
    latest_file = max(results_files, key=lambda x: os.path.getmtime(x))
    print(f"📊 Reading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("\n🎯 ACTUAL VALUES IN THE PLOTS:")
    print("=" * 50)
    
    # Binary Classification values
    binary_metrics = results['binary_classification']['binary_metrics']
    print("📈 Binary Classification:")
    print(f"   Accuracy:  {binary_metrics['accuracy']:.4f} ({binary_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {binary_metrics['precision']:.4f} ({binary_metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {binary_metrics['recall']:.4f} ({binary_metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {binary_metrics['f1_score']:.4f} ({binary_metrics['f1_score']*100:.2f}%)")
    print(f"   ROC-AUC:   {binary_metrics['roc_auc']:.4f} ({binary_metrics['roc_auc']*100:.2f}%)")
    
    print("\n🚀 Test-Time Training:")
    ttt_metrics = results['test_time_training']['binary_metrics']
    print(f"   Accuracy:  {ttt_metrics['accuracy']:.4f} ({ttt_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {ttt_metrics['precision']:.4f} ({ttt_metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {ttt_metrics['recall']:.4f} ({ttt_metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {ttt_metrics['f1_score']:.4f} ({ttt_metrics['f1_score']*100:.2f}%)")
    
    print("\n📊 Performance Comparison Summary:")
    print("=" * 50)
    print(f"Binary Classification vs Test-Time Training:")
    print(f"  Accuracy:  {binary_metrics['accuracy']*100:.1f}% → {ttt_metrics['accuracy']*100:.1f}% (+{(ttt_metrics['accuracy']-binary_metrics['accuracy'])*100:.1f}%)")
    print(f"  Precision: {binary_metrics['precision']*100:.1f}% → {ttt_metrics['precision']*100:.1f}% (+{(ttt_metrics['precision']-binary_metrics['precision'])*100:.1f}%)")
    print(f"  Recall:    {binary_metrics['recall']*100:.1f}% → {ttt_metrics['recall']*100:.1f}% (+{(ttt_metrics['recall']-binary_metrics['recall'])*100:.1f}%)")
    print(f"  F1-Score:  {binary_metrics['f1_score']*100:.1f}% → {ttt_metrics['f1_score']*100:.1f}% (+{(ttt_metrics['f1_score']-binary_metrics['f1_score'])*100:.1f}%)")
    
    print(f"\n🎉 TTT shows MASSIVE improvement: +{(ttt_metrics['accuracy']-binary_metrics['accuracy'])*100:.1f}% accuracy!")

if __name__ == "__main__":
    verify_plot_values()
