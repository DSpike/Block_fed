#!/usr/bin/env python3
"""
Apply only essential TTT safety fixes to main.py
This script applies minimal changes to prevent infinite loops
"""

import re
import shutil
from datetime import datetime

def apply_minimal_ttt_fixes():
    """Apply only essential TTT safety fixes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main_backup_{timestamp}.py"
    shutil.copy2("main.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix 1: Reduce meta-tasks to prevent long execution
    content = re.sub(r'num_meta_tasks = 100', 'num_meta_tasks = 20  # Reduced for testing', content)
    
    # Fix 2: Add safety limit to TTT steps
    content = re.sub(r'(ttt_steps = int\(base_ttt_steps \* complexity_factor\))', 
                    r'\1\n            \n            # Safety limit to prevent infinite loops\n            ttt_steps = min(ttt_steps, 50)  # Maximum 50 steps for testing', content)
    
    # Fix 3: Add timeout mechanism
    timeout_code = '''
            import time
            ttt_start_time = time.time()
            ttt_timeout = 15  # 15 seconds timeout for TTT adaptation'''
    
    content = re.sub(r'(for step in range\(ttt_steps\):)', 
                    f'{timeout_code}\n            \1\n                # Check for timeout\n                if time.time() - ttt_start_time > ttt_timeout:\n                    logger.warning(f"TTT adaptation timeout after {{ttt_timeout}}s at step {{step}}")\n                    break\n                    ', content)
    
    # Fix 4: Reduce patience for faster early stopping
    content = re.sub(r'patience_limit = 8', 'patience_limit = 5  # Reduced for testing', content)
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Minimal TTT safety fixes applied")
    
    # Verify syntax
    import subprocess
    result = subprocess.run(["python", "-m", "py_compile", "main.py"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Syntax verification passed")
        return True
    else:
        print(f"❌ Syntax verification failed: {result.stderr}")
        # Restore backup
        shutil.copy2(backup_name, "main.py")
        return False

if __name__ == "__main__":
    success = apply_minimal_ttt_fixes()
    exit(0 if success else 1)

