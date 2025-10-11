#!/usr/bin/env python3
"""
Test plot display functionality
"""

import os
import subprocess
import sys

def test_plot_display():
    # Check current directory and plot files
    print(f'Current directory: {os.getcwd()}')
    print(f'Files in current directory: {os.listdir(".")}')

    # Check if performance_plots directory exists
    if os.path.exists('performance_plots'):
        print('✅ performance_plots directory exists')
        plot_files = os.listdir('performance_plots')
        print(f'Plot files: {plot_files[:5]}...')  # Show first 5 files
        
        # Test opening a plot with full path
        plot_path = os.path.abspath('performance_plots/confusion_matrices_latest.png')
        print(f'Full plot path: {plot_path}')
        
        if os.path.exists(plot_path):
            try:
                os.startfile(plot_path)
                print('✅ Plot opened successfully')
            except Exception as e:
                print(f'❌ Failed to open plot: {e}')
        else:
            print('❌ Plot file not found')
    else:
        print('❌ performance_plots directory not found')

if __name__ == "__main__":
    test_plot_display()

