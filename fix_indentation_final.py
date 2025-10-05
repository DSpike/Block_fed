#!/usr/bin/env python3
"""
Final indentation fix script for main.py
"""

import re

def fix_indentation():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix common indentation issues
        if line.strip().startswith('aggregated_losses = []') and i > 480:
            # Fix the aggregated_losses line
            fixed_lines.append('        aggregated_losses = []')
        elif line.strip().startswith('aggregated_accuracies = []') and i > 480:
            # Fix the aggregated_accuracies line
            fixed_lines.append('        aggregated_accuracies = []')
        elif line.strip().startswith('for epoch in range(num_epochs):') and i > 480:
            # Fix the for loop
            fixed_lines.append('        for epoch in range(num_epochs):')
        elif line.strip().startswith('epoch_losses = [history') and i > 480:
            # Fix the epoch_losses line
            fixed_lines.append('            epoch_losses = [history[\'epoch_losses\'][epoch] for history in client_meta_histories]')
        elif line.strip().startswith('avg_loss = sum(epoch_losses)') and i > 480:
            # Fix the avg_loss line
            fixed_lines.append('            avg_loss = sum(epoch_losses) / len(epoch_losses)')
        elif line.strip().startswith('aggregated_losses.append(avg_loss)') and i > 480:
            # Fix the append line
            fixed_lines.append('            aggregated_losses.append(avg_loss)')
        elif line.strip().startswith('epoch_accuracies = [history') and i > 480:
            # Fix the epoch_accuracies line
            fixed_lines.append('            epoch_accuracies = [history[\'epoch_accuracies\'][epoch] for history in client_meta_histories]')
        elif line.strip().startswith('avg_accuracy = sum(epoch_accuracies)') and i > 480:
            # Fix the avg_accuracy line
            fixed_lines.append('            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)')
        elif line.strip().startswith('aggregated_accuracies.append(avg_accuracy)') and i > 480:
            # Fix the append line
            fixed_lines.append('            aggregated_accuracies.append(avg_accuracy)')
        elif line.strip().startswith('return {') and i > 480:
            # Fix the return statement
            fixed_lines.append('        return {')
        elif line.strip().startswith("'epoch_losses': aggregated_losses,") and i > 480:
            # Fix the return dict
            fixed_lines.append("            'epoch_losses': aggregated_losses,")
        elif line.strip().startswith("'epoch_accuracies': aggregated_accuracies") and i > 480:
            # Fix the return dict
            fixed_lines.append("            'epoch_accuracies': aggregated_accuracies")
        elif line.strip().startswith('}') and i > 480 and i < 510:
            # Fix the closing brace
            fixed_lines.append('        }')
        else:
            fixed_lines.append(line)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Indentation fixed!")

if __name__ == "__main__":
    fix_indentation()