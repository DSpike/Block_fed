#!/usr/bin/env python3
"""
Fix all indentation issues in main.py systematically
"""

import shutil
from datetime import datetime

def fix_indentation_systematically():
    """Fix all indentation issues systematically"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main_backup_{timestamp}.py"
    shutil.copy2("main.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    
    # Read the file
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Fix specific problematic lines
    fixes_applied = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Fix import statements that should be indented
        if stripped.startswith('import ') and not line.startswith('    '):
            # Check if this import is inside a function/class
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line.endswith(':') or prev_line.startswith('def ') or prev_line.startswith('class '):
                    lines[i] = '    ' + line.lstrip()
                    fixes_applied += 1
                    print(f"✅ Fixed line {i+1}: {stripped}")
        
        # Fix return statements that should be indented
        elif stripped.startswith('return ') and not line.startswith('    '):
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line.endswith(':') or prev_line.startswith('def ') or prev_line.startswith('class '):
                    lines[i] = '    ' + line.lstrip()
                    fixes_applied += 1
                    print(f"✅ Fixed line {i+1}: {stripped}")
        
        # Fix other statements that should be indented
        elif stripped and not line.startswith('    ') and not line.startswith('#') and not stripped.startswith('class ') and not stripped.startswith('def '):
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line.endswith(':') or prev_line.startswith('def ') or prev_line.startswith('class '):
                    lines[i] = '    ' + line.lstrip()
                    fixes_applied += 1
                    print(f"✅ Fixed line {i+1}: {stripped}")
    
    print(f"🔧 Applied {fixes_applied} indentation fixes")
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(lines)
    
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
    success = fix_indentation_systematically()
    exit(0 if success else 1)

