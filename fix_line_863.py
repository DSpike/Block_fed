#!/usr/bin/env python3
"""
Targeted fix for the specific indentation issue at line 863
"""

import shutil
from datetime import datetime

def backup_and_fix():
    """Create backup and fix the specific indentation issue"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main_backup_{timestamp}.py"
    shutil.copy2("main.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    
    # Read the file
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find and fix line 863 (index 862)
    if len(lines) > 862:
        line_863 = lines[862].strip()
        print(f"🔍 Line 863: '{line_863}'")
        
        if line_863 == "else:":
            # Check if the next line is properly indented
            if len(lines) > 863:
                line_864 = lines[863]
                print(f"🔍 Line 864: '{repr(line_864)}'")
                
                if line_864.strip() and not line_864.startswith("    "):
                    # Fix the indentation
                    lines[863] = "    " + line_864.lstrip()
                    print("✅ Fixed indentation for line 864")
                else:
                    print("ℹ️ Line 864 already properly indented")
            else:
                print("⚠️ No line 864 found")
        else:
            print("ℹ️ Line 863 is not an 'else:' statement")
    
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
    success = backup_and_fix()
    exit(0 if success else 1)

