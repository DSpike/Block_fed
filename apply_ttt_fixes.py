#!/usr/bin/env python3
"""
Script to apply TTT adaptation safety fixes to main.py
This script applies the tested fixes to prevent infinite loops
"""

import re
import shutil
from datetime import datetime

def backup_main_file():
    """Create a backup of main.py before applying fixes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main_backup_{timestamp}.py"
    shutil.copy2("main.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    return backup_name

def apply_ttt_safety_fixes():
    """Apply TTT adaptation safety fixes to main.py"""
    print("🔧 Applying TTT adaptation safety fixes...")
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix 1: Add safety limit for TTT steps
    pattern1 = r'(ttt_steps = int\(base_ttt_steps \* complexity_factor\))'
    replacement1 = r'''\1
            
            # Safety limit to prevent infinite loops
            ttt_steps = min(ttt_steps, 100)  # Maximum 100 steps'''
    
    content = re.sub(pattern1, replacement1, content)
    
    # Fix 2: Add timeout mechanism
    pattern2 = r'(for step in range\(ttt_steps\):)'
    replacement2 = r'''import time
            ttt_start_time = time.time()
            ttt_timeout = 30  # 30 seconds timeout for TTT adaptation
            
            \1
                # Check for timeout
                if time.time() - ttt_start_time > ttt_timeout:
                    logger.warning(f"TTT adaptation timeout after {ttt_timeout}s at step {step}")
                    break
                    '''
    
    content = re.sub(pattern2, replacement2, content)
    
    # Fix 3: Add enhanced logging
    pattern3 = r'(logger\.info\(f"Adaptive TTT steps: \{ttt_steps\} \(complexity factor: \{complexity_factor:.2f\}\)"\))'
    replacement3 = r'''\1
            logger.info(f"TTT adaptation will run for maximum {ttt_steps} steps with early stopping at {patience_limit} patience")'''
    
    content = re.sub(pattern3, replacement3, content)
    
    # Fix 4: Reduce meta-tasks for testing
    pattern4 = r'(num_meta_tasks = 100)'
    replacement4 = r'num_meta_tasks = 50  # Reduced from 100 for better performance'
    
    content = re.sub(pattern4, replacement4, content)
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ TTT safety fixes applied successfully")

def apply_sampling_fixes():
    """Apply sampling fixes to prevent train_test_split errors"""
    print("🔧 Applying sampling fixes...")
    
    with open("preprocessing/blockchain_federated_unsw_preprocessor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix the sample_stratified_subset method
    pattern = r'''# Use stratified sampling to preserve class distribution
        X_subset, _, y_subset, _ = train_test_split\(
            X_np, y_np,
            train_size=n_samples,
            stratify=y_np,
            random_state=random_state
        \)'''
    
    replacement = r'''# Use stratified sampling to preserve class distribution
        if n_samples >= len(X_np):
            # If we want all samples, use train_size=1.0
            X_subset, _, y_subset, _ = train_test_split(
                X_np, y_np,
                train_size=1.0,
                stratify=y_np,
                random_state=random_state
            )
        else:
            # Use specific number of samples
            X_subset, _, y_subset, _ = train_test_split(
                X_np, y_np,
                train_size=n_samples,
                stratify=y_np,
                random_state=random_state
            )'''
    
    content = re.sub(pattern, replacement, content)
    
    # Write the fixed content back
    with open("preprocessing/blockchain_federated_unsw_preprocessor.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Sampling fixes applied successfully")

def apply_fallback_data_fixes():
    """Apply fallback data fixes for failed evaluations"""
    print("🔧 Applying fallback data fixes...")
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix base model evaluation fallback
    pattern1 = r"(return \{'accuracy_mean': 0\.0, 'accuracy_std': 0\.0, 'macro_f1_mean': 0\.0, 'macro_f1_std': 0\.0, 'mcc_mean': 0\.0, 'mcc_std': 0\.0\})"
    replacement1 = r'''return {
                'accuracy_mean': 0.0, 'accuracy_std': 0.0, 
                'precision_mean': 0.0, 'precision_std': 0.0,
                'recall_mean': 0.0, 'recall_std': 0.0,
                'macro_f1_mean': 0.0, 'macro_f1_std': 0.0, 
                'mcc_mean': 0.0, 'mcc_std': 0.0,
                # Add fallback confusion matrix data
                'confusion_matrix': [[0, 0], [0, 0]],
                'roc_curve': {'fpr': [0, 1], 'tpr': [0, 1], 'thresholds': [1, 0]},
                'roc_auc': 0.5,
                'optimal_threshold': 0.5
            }'''
    
    content = re.sub(pattern1, replacement1, content)
    
    # Fix TTT model evaluation fallback
    pattern2 = r"(return \{'accuracy_mean': 0\.0, 'accuracy_std': 0\.0, 'macro_f1_mean': 0\.0, 'macro_f1_std': 0\.0, 'mcc_mean': 0\.0, 'mcc_std': 0\.0\})"
    replacement2 = r'''return {
                'accuracy_mean': 0.0, 'accuracy_std': 0.0, 
                'precision_mean': 0.0, 'precision_std': 0.0,
                'recall_mean': 0.0, 'recall_std': 0.0,
                'macro_f1_mean': 0.0, 'macro_f1_std': 0.0, 
                'mcc_mean': 0.0, 'mcc_std': 0.0,
                # Add fallback confusion matrix data
                'confusion_matrix': [[0, 0], [0, 0]],
                'roc_curve': {'fpr': [0, 1], 'tpr': [0, 1], 'thresholds': [1, 0]},
                'roc_auc': 0.5,
                'optimal_threshold': 0.5
            }'''
    
    # Apply the second replacement (this will match the TTT model fallback)
    content = re.sub(pattern2, replacement2, content, count=1)
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Fallback data fixes applied successfully")

def verify_syntax():
    """Verify that the fixed file has correct syntax"""
    print("🔍 Verifying syntax...")
    
    import subprocess
    result = subprocess.run(["python", "-m", "py_compile", "main.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Syntax verification passed")
        return True
    else:
        print(f"❌ Syntax verification failed: {result.stderr}")
        return False

def main():
    """Main function to apply all fixes"""
    print("🚀 Starting TTT adaptation fixes application...")
    
    # Create backup
    backup_file = backup_main_file()
    
    try:
        # Apply fixes
        apply_ttt_safety_fixes()
        apply_sampling_fixes()
        apply_fallback_data_fixes()
        
        # Verify syntax
        if verify_syntax():
            print("🎉 All fixes applied successfully!")
            print(f"📁 Backup created: {backup_file}")
            print("✅ Ready to test the system")
            return True
        else:
            print("❌ Syntax verification failed. Restoring backup...")
            shutil.copy2(backup_file, "main.py")
            return False
            
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        print("🔄 Restoring backup...")
        shutil.copy2(backup_file, "main.py")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

