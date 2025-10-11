#!/usr/bin/env python3
"""
Comprehensive script to fix all issues in main.py
This script fixes indentation, TTT adaptation, and other issues
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

def fix_indentation_issues():
    """Fix common indentation issues in main.py"""
    print("🔧 Fixing indentation issues...")
    
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix common indentation patterns
        if line.strip().startswith('try:'):
            fixed_lines.append(line)
            i += 1
            # Ensure the next line is properly indented
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif line.strip().startswith('except') or line.strip().startswith('finally'):
            fixed_lines.append(line)
            i += 1
            # Ensure the next line is properly indented
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif line.strip().startswith('else:'):
            fixed_lines.append(line)
            i += 1
            # Ensure the next line is properly indented
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)
    
    print("✅ Indentation issues fixed")

def apply_critical_fixes():
    """Apply critical fixes to prevent infinite loops and errors"""
    print("🔧 Applying critical fixes...")
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix 1: Reduce meta-tasks to prevent long execution
    content = re.sub(r'num_meta_tasks = 100', 'num_meta_tasks = 20  # Reduced for testing', content)
    
    # Fix 2: Add safety limits to TTT steps
    ttt_safety_code = '''
            # Safety limit to prevent infinite loops
            ttt_steps = min(ttt_steps, 50)  # Maximum 50 steps for testing
            logger.info(f"TTT adaptation will run for maximum {ttt_steps} steps with early stopping at {patience_limit} patience")
            
            import time
            ttt_start_time = time.time()
            ttt_timeout = 15  # 15 seconds timeout for TTT adaptation'''
    
    # Insert safety code before the TTT loop
    content = re.sub(
        r'(for step in range\(ttt_steps\):)',
        f'{ttt_safety_code}\n            \1\n                # Check for timeout\n                if time.time() - ttt_start_time > ttt_timeout:\n                    logger.warning(f"TTT adaptation timeout after {{ttt_timeout}}s at step {{step}}")\n                    break\n                    ',
        content
    )
    
    # Fix 3: Add fallback data for failed evaluations
    fallback_data = '''{
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
    
    # Replace simple fallback with comprehensive fallback
    content = re.sub(
        r"return \{'accuracy_mean': 0\.0, 'accuracy_std': 0\.0, 'macro_f1_mean': 0\.0, 'macro_f1_std': 0\.0, 'mcc_mean': 0\.0, 'mcc_std': 0\.0\}",
        f'return {fallback_data}',
        content
    )
    
    # Fix 4: Add TTT adaptation data with steps
    ttt_data_fix = '''{
                    'task_accuracies': task_metrics['accuracy'],
                    'task_f1_scores': task_metrics['f1_score'],
                    'task_mcc_scores': task_metrics['mcc'],
                    'num_tasks': len(task_metrics['accuracy']),
                    'mean_accuracy': results['accuracy_mean'],
                    'std_accuracy': results['accuracy_std'],
                    'steps': list(range(len(task_metrics['accuracy'])))  # Add steps for plotting
                }'''
    
    # Replace TTT adaptation data
    content = re.sub(
        r"self\.ttt_adaptation_data = \{[^}]+\}",
        f"self.ttt_adaptation_data = {ttt_data_fix}",
        content
    )
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Critical fixes applied")

def fix_preprocessor():
    """Fix the preprocessor sampling issue"""
    print("🔧 Fixing preprocessor...")
    
    with open("preprocessing/blockchain_federated_unsw_preprocessor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix the sampling method
    sampling_fix = '''# Use stratified sampling to preserve class distribution
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
    
    # Replace the problematic sampling code
    content = re.sub(
        r'# Use stratified sampling to preserve class distribution\n        X_subset, _, y_subset, _ = train_test_split\(\n            X_np, y_np,\n            train_size=n_samples,\n            stratify=y_np,\n            random_state=random_state\n        \)',
        sampling_fix,
        content
    )
    
    # Write the fixed content back
    with open("preprocessing/blockchain_federated_unsw_preprocessor.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Preprocessor fixed")

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
    print("🚀 Starting comprehensive fixes application...")
    
    # Create backup
    backup_file = backup_main_file()
    
    try:
        # Apply fixes
        fix_indentation_issues()
        apply_critical_fixes()
        fix_preprocessor()
        
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