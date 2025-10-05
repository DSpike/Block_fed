#!/usr/bin/env python3
"""
Comprehensive indentation fixer for main.py
"""

import re

def fix_all_indentation():
    """Fix all indentation issues in main.py"""
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common indentation patterns
    fixes = [
        # Fix method indentation
        (r'(\n    def _aggregate_meta_histories\(self, client_meta_histories: List\[Dict\]\) -> Dict:.*?)(\n    def|\n\n    def|\Z)', 
         r'\1\n    def _aggregate_meta_histories(self, client_meta_histories: List[Dict]) -> Dict:\n        """\n        Aggregate meta-learning histories from all clients.\n        \n        Args:\n            client_meta_histories: List of meta-learning histories from clients\n            \n        Returns:\n            aggregated_history: Aggregated meta-learning history\n        """\n        if not client_meta_histories:\n            return {\'epoch_losses\': [], \'epoch_accuracies\': []}\n        \n        # Average losses and accuracies across clients\n        num_epochs = len(client_meta_histories[0][\'epoch_losses\'])\n        aggregated_losses = []\n        aggregated_accuracies = []\n        \n        for epoch in range(num_epochs):\n            # Average loss across clients for this epoch\n            epoch_losses = [history[\'epoch_losses\'][epoch] for history in client_meta_histories]\n            avg_loss = sum(epoch_losses) / len(epoch_losses)\n            aggregated_losses.append(avg_loss)\n            \n            # Average accuracy across clients for this epoch\n            epoch_accuracies = [history[\'epoch_accuracies\'][epoch] for history in client_meta_histories]\n            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)\n            aggregated_accuracies.append(avg_accuracy)\n        \n        return {\n            \'epoch_losses\': aggregated_losses,\n            \'epoch_accuracies\': aggregated_accuracies\n        }\n\2'),
        
        # Fix try-except indentation
        (r'(\s+)try:\s*\n(\s+)outputs = self\.model\(batch_data\)', 
         r'\1try:\n\1    outputs = self.model(batch_data)'),
        
        # Fix if statement indentation
        (r'(\s+)if self\.device\.type == \'cuda\':\s*\n(\s+)torch\.cuda\.empty_cache\(\)', 
         r'\1if self.device.type == \'cuda\':\n\1    torch.cuda.empty_cache()'),
        
        # Fix else statement indentation
        (r'(\s+)else:\s*\n(\s+)client_address = self\.client_addresses\.get\(client_update\.client_id\)', 
         r'\1else:\n\1    client_address = self.client_addresses.get(client_update.client_id)'),
        
        # Fix if-else indentation in incentive summary
        (r'(\s+)if self\.incentive_manager is not None:\s*\n(\s+)round_summary = self\.incentive_manager\.get_round_summary\(record\[\'round_number\'\]\)', 
         r'\1if self.incentive_manager is not None:\n\1    round_summary = self.incentive_manager.get_round_summary(record[\'round_number\'])'),
    ]
    
    # Apply fixes
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back to file
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("All indentation issues fixed!")

if __name__ == "__main__":
    fix_all_indentation()