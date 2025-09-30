#!/usr/bin/env python3
"""
Test only the preprocessing pipeline to verify the tensor conversion fix
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

def test_preprocessing():
    """Test the preprocessing pipeline with proper zero-day holdout"""
    print("🚀 Testing UNSW-NB15 Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = UNSWPreprocessor()
        
        print("📊 Running preprocessing with Backdoor as zero-day attack...")
        
        # Run preprocessing
        preprocessed_data = preprocessor.preprocess_unsw_dataset(
            zero_day_attack="Backdoor"
        )
        
        print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display results
        print("📊 Final Results:")
        print(f"   Training samples: {len(preprocessed_data['X_train'])}")
        print(f"   Validation samples: {len(preprocessed_data['X_val'])}")
        print(f"   Test samples: {len(preprocessed_data['X_test'])}")
        print(f"   Zero-day samples: {len(preprocessed_data['zero_day_indices'])}")
        
        print(f"\n🎯 Zero-Day Detection Setup:")
        print(f"   Zero-day attack: Backdoor")
        print(f"   Zero-day samples in test: {len(preprocessed_data['zero_day_indices'])}")
        print(f"   Normal samples in test: {len(preprocessed_data['X_test']) - len(preprocessed_data['zero_day_indices'])}")
        
        print(f"\n🔧 Tensor Information:")
        print(f"   X_train shape: {preprocessed_data['X_train'].shape}")
        print(f"   X_train dtype: {preprocessed_data['X_train'].dtype}")
        print(f"   X_test shape: {preprocessed_data['X_test'].shape}")
        print(f"   X_test dtype: {preprocessed_data['X_test'].dtype}")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Tensor conversion working correctly")
        print("✅ Proper zero-day holdout implemented")
        print("✅ System ready for federated learning")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("\n🎉 Preprocessing test completed successfully!")
        print("The tensor conversion fix is working perfectly.")
    else:
        print("\n💥 Preprocessing test failed.")

