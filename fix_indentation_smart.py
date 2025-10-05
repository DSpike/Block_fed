#!/usr/bin/env python3
"""
Smart indentation fixer for main.py
"""

def fix_indentation():
    """Fix indentation issues in main.py"""
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic method
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if 'def _aggregate_meta_histories(self, client_meta_histories: List[Dict]) -> Dict:' in line:
            start_line = i
        elif start_line is not None and line.strip().startswith('def ') and i > start_line + 5:
            end_line = i
            break
    
    if start_line is None:
        print("Method not found!")
        return
    
    if end_line is None:
        end_line = len(lines)
    
    print(f"Found method from line {start_line + 1} to {end_line}")
    
    # Fix the method content
    fixed_method = '''    def _aggregate_meta_histories(self, client_meta_histories: List[Dict]) -> Dict:
        """
        Aggregate meta-learning histories from all clients.
        
        Args:
            client_meta_histories: List of meta-learning histories from clients
            
        Returns:
            aggregated_history: Aggregated meta-learning history
        """
        if not client_meta_histories:
            return {'epoch_losses': [], 'epoch_accuracies': []}
        
        # Average losses and accuracies across clients
        num_epochs = len(client_meta_histories[0]['epoch_losses'])
        aggregated_losses = []
        aggregated_accuracies = []
        
        for epoch in range(num_epochs):
            # Average loss across clients for this epoch
            epoch_losses = [history['epoch_losses'][epoch] for history in client_meta_histories]
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            aggregated_losses.append(avg_loss)
            
            # Average accuracy across clients for this epoch
            epoch_accuracies = [history['epoch_accuracies'][epoch] for history in client_meta_histories]
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            aggregated_accuracies.append(avg_accuracy)
        
        return {
            'epoch_losses': aggregated_losses,
            'epoch_accuracies': aggregated_accuracies
        }
'''
    
    # Replace the method
    new_lines = lines[:start_line] + fixed_method.split('\n') + lines[end_line:]
    
    # Write back to file
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' if not line.endswith('\n') else line for line in new_lines])
    
    print("Indentation fixed successfully!")

if __name__ == "__main__":
    fix_indentation()