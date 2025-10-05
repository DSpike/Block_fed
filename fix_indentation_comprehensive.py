#!/usr/bin/env python3
"""
Comprehensive indentation fixer for main.py
This script fixes all indentation issues in the main.py file
"""

import re

def fix_indentation_comprehensive():
    """Fix all indentation issues in main.py"""
    
    print("Reading main.py...")
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying indentation fixes...")
    
    # Fix 1: _aggregate_meta_histories method
    pattern1 = r'(\s+def _aggregate_meta_histories\(self, client_meta_histories: List\[Dict\]\) -> Dict:.*?)(\n\s+def|\n\n\s+def|\Z)'
    replacement1 = '''    def _aggregate_meta_histories(self, client_meta_histories: List[Dict]) -> Dict:
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
    content = re.sub(pattern1, replacement1 + r'\2', content, flags=re.DOTALL)
    
    # Fix 2: try-except indentation
    pattern2 = r'(\s+)try:\s*\n(\s+)outputs = self\.model\(batch_data\)'
    replacement2 = r'\1try:\n\1    outputs = self.model(batch_data)'
    content = re.sub(pattern2, replacement2, content)
    
    # Fix 3: if statement indentation for torch.cuda.empty_cache()
    pattern3 = r'(\s+)if self\.device\.type == [\'"]cuda[\'"]:\s*\n(\s+)torch\.cuda\.empty_cache\(\)'
    replacement3 = r'\1if self.device.type == \'cuda\':\n\1    torch.cuda.empty_cache()'
    content = re.sub(pattern3, replacement3, content)
    
    # Fix 4: else statement indentation
    pattern4 = r'(\s+)else:\s*\n(\s+)# Fallback to client_addresses dictionary\s*\n(\s+)client_address = self\.client_addresses\.get\(client_update\.client_id\)'
    replacement4 = r'\1else:\n\1    # Fallback to client_addresses dictionary\n\1    client_address = self.client_addresses.get(client_update.client_id)'
    content = re.sub(pattern4, replacement4, content)
    
    # Fix 5: if-else indentation in incentive summary
    pattern5 = r'(\s+)if self\.incentive_manager is not None:\s*\n(\s+)round_summary = self\.incentive_manager\.get_round_summary\(record\[[\'"]round_number[\'"]\]\)'
    replacement5 = r'\1if self.incentive_manager is not None:\n\1    round_summary = self.incentive_manager.get_round_summary(record[\'round_number\'])'
    content = re.sub(pattern5, replacement5, content)
    
    print("Writing fixed content back to main.py...")
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ All indentation issues fixed successfully!")

if __name__ == "__main__":
    fix_indentation_comprehensive()