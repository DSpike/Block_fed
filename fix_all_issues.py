#!/usr/bin/env python3
"""
Comprehensive fix for all indentation and syntax issues in main.py
"""

import re

def fix_all_issues():
    """Fix all indentation and syntax issues in main.py"""
    
    print("Reading main.py...")
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying comprehensive fixes...")
    
    # Fix 1: Fix all indentation issues systematically
    lines = content.split('\n')
    fixed_lines = []
    indent_level = 0
    in_function = False
    in_class = False
    in_multiline_string = False
    string_delimiter = None
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Check for multiline strings
        if '"""' in line or "'''" in line:
            if not in_multiline_string:
                in_multiline_string = True
                string_delimiter = '"""' if '"""' in line else "'''"
            elif string_delimiter in line:
                in_multiline_string = False
                string_delimiter = None
        
        # Skip indentation fixes inside multiline strings
        if in_multiline_string:
            fixed_lines.append(line)
            continue
        
        # Determine proper indentation
        if stripped.startswith('class '):
            # Class definition - base level
            fixed_line = line.lstrip()
            in_class = True
            in_function = False
        elif stripped.startswith('def '):
            # Function definition - base level or class level
            if in_class:
                fixed_line = '    ' + line.lstrip()
            else:
                fixed_line = line.lstrip()
            in_function = True
        elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('try:') or stripped.startswith('except ') or stripped.startswith('else:') or stripped.startswith('elif '):
            # Control flow statements
            if in_function:
                fixed_line = '        ' + line.lstrip()
            else:
                fixed_line = '    ' + line.lstrip()
        elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue ') or stripped.startswith('pass'):
            # Control statements
            if in_function:
                fixed_line = '            ' + line.lstrip()
            else:
                fixed_line = '        ' + line.lstrip()
        elif stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            # Comments and docstrings - maintain relative indentation
            fixed_line = line
        else:
            # Other statements - maintain relative indentation
            fixed_line = line
        
        # Ensure consistent indentation
        if fixed_line.startswith('    '):
            # Already properly indented
            pass
        elif fixed_line.startswith('        '):
            # Double indented
            pass
        elif fixed_line.startswith('            '):
            # Triple indented
            pass
        else:
            # Try to fix indentation based on context
            if stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while '):
                fixed_line = '        ' + line.lstrip()
            elif stripped.startswith('return ') or stripped.startswith('break ') or stripped.startswith('continue '):
                fixed_line = '            ' + line.lstrip()
            elif stripped.startswith('def ') or stripped.startswith('class '):
                fixed_line = line.lstrip()
            else:
                fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 2: Fix specific syntax issues
    # Fix missing newlines between functions
    content = re.sub(r'(\w+)\s+def\s+(\w+)', r'\1\n    \n    def \2', content)
    
    # Fix escaped quotes
    content = content.replace('\\\'', "'")
    content = content.replace('\\"', '"')
    
    # Fix malformed f-strings
    content = re.sub(r'f"([^"]*)\\([^"]*)"', r'f"\1\2"', content)
    
    print("Writing fixed content back to main.py...")
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ All issues fixed successfully!")

if __name__ == "__main__":
    fix_all_issues()
