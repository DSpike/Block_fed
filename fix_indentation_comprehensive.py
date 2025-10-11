#!/usr/bin/env python3
"""
Comprehensive indentation fix for main.py
This script systematically fixes all indentation issues
"""

import shutil
from datetime import datetime

def fix_all_indentation():
    """Fix all indentation issues in main.py"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"main_backup_{timestamp}.py"
    shutil.copy2("main.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    
    # Read the file
    with open("main.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Fix common indentation patterns
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            i += 1
            continue
        
        # Fix try-except-else-finally blocks
        if stripped.startswith('try:'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('except') or stripped.startswith('finally'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('else:'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('if ') and stripped.endswith(':'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('for ') and stripped.endswith(':'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('while ') and stripped.endswith(':'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('def ') and stripped.endswith(':'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        elif stripped.startswith('class ') and stripped.endswith(':'):
            fixed_lines.append(line)
            i += 1
            # Ensure next non-empty line is indented
            while i < len(lines) and not lines[i].strip():
                fixed_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i].lstrip()
        
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write the fixed content back
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)
    
    print("✅ Indentation fixes applied")
    
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
    success = fix_all_indentation()
    exit(0 if success else 1)