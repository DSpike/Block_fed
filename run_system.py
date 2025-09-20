#!/usr/bin/env python3
"""
Simple runner script for the Blockchain Federated Learning System
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main system
from main import main

if __name__ == "__main__":
    print("🚀 Starting Blockchain Federated Learning System...")
    print("📁 Project Structure:")
    print("   - Source code: src/")
    print("   - Smart contracts: contracts/")
    print("   - Deployment scripts: scripts/")
    print("   - Configuration: config/")
    print("   - Results: results/")
    print("   - Performance plots: performance_plots/")
    print("   - Tests: tests/")
    print("   - Documentation: docs/")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
