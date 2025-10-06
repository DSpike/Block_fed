#!/usr/bin/env python3
"""
Visual Demonstration of Identical Initialization
Creates visual proof that miners start with identical parameters
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from decentralized_fl_system import DecentralizedFederatedLearningSystem
from models.transductive_fewshot_model import TransductiveFewShotModel

def visualize_parameter_identity():
    """Create visual proof of identical initialization"""
    
    print("🎨 Creating Visual Proof of Identical Initialization")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create original model
    original_model = TransductiveFewShotModel(30, 64, 32, 2)
    
    # Initialize decentralized system
    fl_system = DecentralizedFederatedLearningSystem(original_model, num_clients=3)
    
    # Get models
    miner_1_model = fl_system.miners["miner_1"].model
    miner_2_model = fl_system.miners["miner_2"].model
    
    # Extract first layer weights for visualization
    original_weights = original_model.meta_learner.transductive_net.feature_extractors[0][0].weight.data.cpu().numpy()
    miner_1_weights = miner_1_model.meta_learner.transductive_net.feature_extractors[0][0].weight.data.cpu().numpy()
    miner_2_weights = miner_2_model.meta_learner.transductive_net.feature_extractors[0][0].weight.data.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Visual Proof: Miners Start with Identical Parameters', fontsize=16, fontweight='bold')
    
    # Plot 1: Original Model Weights
    im1 = axes[0, 0].imshow(original_weights, cmap='RdBu', aspect='auto')
    axes[0, 0].set_title('Original Model\nFirst Layer Weights')
    axes[0, 0].set_xlabel('Output Features')
    axes[0, 0].set_ylabel('Input Features')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Miner 1 Weights
    im2 = axes[0, 1].imshow(miner_1_weights, cmap='RdBu', aspect='auto')
    axes[0, 1].set_title('Miner 1\nFirst Layer Weights')
    axes[0, 1].set_xlabel('Output Features')
    axes[0, 1].set_ylabel('Input Features')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Miner 2 Weights
    im3 = axes[0, 2].imshow(miner_2_weights, cmap='RdBu', aspect='auto')
    axes[0, 2].set_title('Miner 2\nFirst Layer Weights')
    axes[0, 2].set_xlabel('Output Features')
    axes[0, 2].set_ylabel('Input Features')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Difference between Original and Miner 1
    diff_orig_1 = np.abs(original_weights - miner_1_weights)
    im4 = axes[1, 0].imshow(diff_orig_1, cmap='Reds', aspect='auto')
    axes[1, 0].set_title('|Original - Miner 1|\nShould be ZERO')
    axes[1, 0].set_xlabel('Output Features')
    axes[1, 0].set_ylabel('Input Features')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Difference between Original and Miner 2
    diff_orig_2 = np.abs(original_weights - miner_2_weights)
    im5 = axes[1, 1].imshow(diff_orig_2, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('|Original - Miner 2|\nShould be ZERO')
    axes[1, 1].set_xlabel('Output Features')
    axes[1, 1].set_ylabel('Input Features')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: Difference between Miner 1 and Miner 2
    diff_1_2 = np.abs(miner_1_weights - miner_2_weights)
    im6 = axes[1, 2].imshow(diff_1_2, cmap='Reds', aspect='auto')
    axes[1, 2].set_title('|Miner 1 - Miner 2|\nShould be ZERO')
    axes[1, 2].set_xlabel('Output Features')
    axes[1, 2].set_ylabel('Input Features')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Add text annotations
    max_diff_orig_1 = np.max(diff_orig_1)
    max_diff_orig_2 = np.max(diff_orig_2)
    max_diff_1_2 = np.max(diff_1_2)
    
    axes[1, 0].text(0.5, 0.95, f'Max Diff: {max_diff_orig_1:.2e}', 
                    transform=axes[1, 0].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 1].text(0.5, 0.95, f'Max Diff: {max_diff_orig_2:.2e}', 
                    transform=axes[1, 1].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[1, 2].text(0.5, 0.95, f'Max Diff: {max_diff_1_2:.2e}', 
                    transform=axes[1, 2].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('miner_initialization_proof.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'miner_initialization_proof.png'")
    
    # Print numerical proof
    print(f"\n📊 NUMERICAL PROOF:")
    print(f"   Max difference (Original - Miner 1): {max_diff_orig_1:.2e}")
    print(f"   Max difference (Original - Miner 2): {max_diff_orig_2:.2e}")
    print(f"   Max difference (Miner 1 - Miner 2): {max_diff_1_2:.2e}")
    
    if max_diff_orig_1 < 1e-10 and max_diff_orig_2 < 1e-10 and max_diff_1_2 < 1e-10:
        print("   ✅ ALL DIFFERENCES ARE EFFECTIVELY ZERO!")
        print("   ✅ MINERS TRULY START WITH IDENTICAL PARAMETERS!")
    else:
        print("   ❌ Differences detected - parameters are not identical")
    
    return max_diff_1_2 < 1e-10

def create_parameter_comparison_table():
    """Create a detailed parameter comparison table"""
    
    print("\n📋 Creating Detailed Parameter Comparison Table")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models
    original_model = TransductiveFewShotModel(30, 64, 32, 2)
    fl_system = DecentralizedFederatedLearningSystem(original_model, num_clients=3)
    
    miner_1_model = fl_system.miners["miner_1"].model
    miner_2_model = fl_system.miners["miner_2"].model
    
    print("Parameter Name".ljust(50) + "Original".ljust(15) + "Miner 1".ljust(15) + "Miner 2".ljust(15) + "Identical".ljust(10))
    print("-" * 110)
    
    identical_count = 0
    total_count = 0
    
    for (name_orig, param_orig), (name1, param1), (name2, param2) in zip(
        original_model.named_parameters(),
        miner_1_model.named_parameters(),
        miner_2_model.named_parameters()
    ):
        total_count += 1
        
        # Get first few values for comparison (handle different tensor dimensions)
        try:
            if param_orig.dim() >= 2:
                orig_val = param_orig[0, 0].item()
                miner1_val = param1[0, 0].item()
                miner2_val = param2[0, 0].item()
            elif param_orig.dim() == 1:
                orig_val = param_orig[0].item()
                miner1_val = param1[0].item()
                miner2_val = param2[0].item()
            else:
                orig_val = param_orig.item()
                miner1_val = param1.item()
                miner2_val = param2.item()
        except:
            orig_val = param_orig.flatten()[0].item()
            miner1_val = param1.flatten()[0].item()
            miner2_val = param2.flatten()[0].item()
        
        # Check if identical
        identical = (torch.equal(param_orig, param1) and 
                    torch.equal(param_orig, param2) and 
                    torch.equal(param1, param2))
        
        if identical:
            identical_count += 1
            status = "✅ YES"
        else:
            status = "❌ NO"
        
        # Truncate long names
        display_name = name_orig[:47] + "..." if len(name_orig) > 50 else name_orig
        
        print(f"{display_name.ljust(50)}{orig_val:15.8f}{miner1_val:15.8f}{miner2_val:15.8f}{status.ljust(10)}")
    
    print("-" * 110)
    print(f"Total Parameters: {total_count}")
    print(f"Identical Parameters: {identical_count}")
    print(f"Identical Percentage: {identical_count/total_count*100:.1f}%")
    
    if identical_count == total_count:
        print("✅ ALL PARAMETERS ARE IDENTICAL!")
    else:
        print(f"❌ {total_count - identical_count} parameters are NOT identical")
    
    return identical_count == total_count

def main():
    """Run all visualization tests"""
    try:
        print("🎨 VISUAL PROOF OF IDENTICAL INITIALIZATION")
        print("=" * 60)
        
        # Test 1: Visual proof
        visual_proof = visualize_parameter_identity()
        
        # Test 2: Detailed table
        table_proof = create_parameter_comparison_table()
        
        # Final verdict
        print("\n🏆 FINAL VERDICT:")
        print("=" * 30)
        
        if visual_proof and table_proof:
            print("✅ CONFIRMED: Miners truly start with identical parameters")
            print("✅ Visual proof: All differences are effectively zero")
            print("✅ Table proof: All parameters are identical")
            print("✅ System is working correctly")
        else:
            print("❌ ISSUES DETECTED!")
            print("❌ Miners may not be starting with identical parameters")
        
    except Exception as e:
        print(f"\n❌ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
