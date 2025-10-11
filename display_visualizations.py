#!/usr/bin/env python3
"""
Display Generated Visualizations
This script displays the generated performance visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def display_visualizations():
    """Display all generated visualizations"""
    
    # Set up the display
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Find all visualization files
    plot_dir = "enhanced_binary_plots"
    if not os.path.exists(plot_dir):
        print(f"❌ Visualization directory '{plot_dir}' not found!")
        return
    
    # Get all PNG files
    plot_files = glob.glob(os.path.join(plot_dir, "*.png"))
    
    if not plot_files:
        print(f"❌ No visualization files found in '{plot_dir}'!")
        return
    
    print(f"📊 Found {len(plot_files)} visualization files:")
    for file in plot_files:
        print(f"   - {os.path.basename(file)}")
    
    # Display each plot
    for i, plot_file in enumerate(plot_files):
        try:
            # Load and display the image
            img = mpimg.imread(plot_file)
            
            plt.figure(i + 1, figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Visualization {i + 1}: {os.path.basename(plot_file)}", 
                     fontsize=14, fontweight='bold')
            
            print(f"✅ Displaying: {os.path.basename(plot_file)}")
            
        except Exception as e:
            print(f"❌ Error displaying {plot_file}: {e}")
    
    # Show all plots
    plt.tight_layout()
    plt.show()
    
    print("🎉 All visualizations displayed successfully!")

if __name__ == "__main__":
    print("🎨 Displaying Generated Performance Visualizations")
    print("=" * 60)
    display_visualizations()

