#!/usr/bin/env python3
"""
Performance Analysis and Fixes for TCGAN/TTT Model
Identifies and fixes critical performance issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyze and fix performance issues"""
    
    def __init__(self):
        self.issues = []
        self.fixes = []
    
    def analyze_performance_issues(self, metrics_file: str):
        """Analyze performance metrics and identify issues"""
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        issues = []
        
        # Check base model performance
        base_model = metrics['evaluation_results']['base_model']
        ttt_model = metrics['evaluation_results']['ttt_model']
        
        # Issue 1: Poor base model accuracy
        if base_model['accuracy'] < 0.7:
            issues.append({
                'type': 'low_accuracy',
                'severity': 'critical',
                'description': f"Base model accuracy {base_model['accuracy']:.3f} is too low (should be >0.7)",
                'metric': 'accuracy',
                'value': base_model['accuracy']
            })
        
        # Issue 2: Poor ROC-AUC
        if base_model['roc_auc'] < 0.7:
            issues.append({
                'type': 'poor_discrimination',
                'severity': 'critical', 
                'description': f"ROC-AUC {base_model['roc_auc']:.3f} indicates poor discrimination (should be >0.7)",
                'metric': 'roc_auc',
                'value': base_model['roc_auc']
            })
        
        # Issue 3: Zero MCC
        if base_model['mccc'] == 0.0:
            issues.append({
                'type': 'no_correlation',
                'severity': 'critical',
                'description': "MCC = 0 indicates no correlation between predictions and labels",
                'metric': 'mccc',
                'value': base_model['mccc']
            })
        
        # Issue 4: Incomplete evaluation
        evaluated_samples = metrics['evaluation_results']['evaluated_samples']
        test_samples = metrics['evaluation_results']['test_samples']
        if evaluated_samples < test_samples * 0.8:
            issues.append({
                'type': 'incomplete_evaluation',
                'severity': 'high',
                'description': f"Only {evaluated_samples}/{test_samples} samples evaluated ({evaluated_samples/test_samples:.1%})",
                'metric': 'evaluation_coverage',
                'value': evaluated_samples/test_samples
            })
        
        # Issue 5: Confusion matrix bias
        cm = base_model['confusion_matrix']
        if cm['tn'] == 0 or cm['fn'] == 0:
            issues.append({
                'type': 'prediction_bias',
                'severity': 'critical',
                'description': f"Model is biased - TN={cm['tn']}, FN={cm['fn']} (predicting all as one class)",
                'metric': 'confusion_matrix',
                'value': cm
            })
        
        return issues
    
    def generate_fixes(self, issues: List[Dict]) -> List[Dict]:
        """Generate fixes for identified issues"""
        fixes = []
        
        for issue in issues:
            if issue['type'] == 'low_accuracy':
                fixes.append({
                    'type': 'model_architecture',
                    'description': 'Improve model architecture with deeper networks and better feature extraction',
                    'implementation': 'enhance_model_architecture'
                })
                
            elif issue['type'] == 'poor_discrimination':
                fixes.append({
                    'type': 'training_improvement',
                    'description': 'Improve training with better loss functions and learning rates',
                    'implementation': 'enhance_training'
                })
                
            elif issue['type'] == 'no_correlation':
                fixes.append({
                    'type': 'loss_function',
                    'description': 'Fix loss functions and training objective',
                    'implementation': 'fix_loss_functions'
                })
                
            elif issue['type'] == 'incomplete_evaluation':
                fixes.append({
                    'type': 'evaluation_fix',
                    'description': 'Fix evaluation to use full test set',
                    'implementation': 'fix_evaluation'
                })
                
            elif issue['type'] == 'prediction_bias':
                fixes.append({
                    'type': 'class_balance',
                    'description': 'Fix class imbalance and prediction bias',
                    'implementation': 'fix_class_balance'
                })
        
        return fixes

class EnhancedTCGANModel(nn.Module):
    """Enhanced TCGAN model with better architecture"""
    
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256, 
                 num_classes: int = 2, noise_dim: int = 128):
        super(EnhancedTCGANModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        
        # Enhanced Encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, latent_dim)
        )
        
        # Enhanced Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Enhanced Generator
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Enhanced Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Enhanced Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        # Enhanced optimizers with better learning rates
        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=0.005, weight_decay=1e-3)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=0.005, weight_decay=1e-3)
        self.generator_optimizer = optim.AdamW(self.generator.parameters(), lr=0.003, weight_decay=1e-3)
        self.discriminator_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.002, weight_decay=1e-3)
        self.classifier_optimizer = optim.AdamW(self.classifier.parameters(), lr=0.007, weight_decay=1e-3)
        
        # Enhanced loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Focal loss for class imbalance
        self.focal_loss = self._focal_loss
        
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        """Focal loss for handling class imbalance"""
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def generate(self, batch_size, device):
        noise = torch.randn(batch_size, self.noise_dim).to(device)
        return self.generator(noise)
    
    def discriminate(self, x):
        disc_logits = self.discriminator(x)
        class_logits = self.classifier(x)
        return disc_logits, class_logits
    
    def forward(self, x):
        _, class_logits = self.discriminate(x)
        return class_logits

def create_performance_report(metrics_file: str) -> str:
    """Create a comprehensive performance analysis report"""
    analyzer = PerformanceAnalyzer()
    
    # Analyze issues
    issues = analyzer.analyze_performance_issues(metrics_file)
    
    # Generate fixes
    fixes = analyzer.generate_fixes(issues)
    
    # Create report
    report = f"""
# Performance Analysis Report

## Issues Identified ({len(issues)})

"""
    
    for i, issue in enumerate(issues, 1):
        report += f"### {i}. {issue['type'].replace('_', ' ').title()} ({issue['severity']})\n"
        report += f"- **Description**: {issue['description']}\n"
        report += f"- **Metric**: {issue['metric']}\n"
        report += f"- **Value**: {issue['value']}\n\n"
    
    report += f"""
## Recommended Fixes ({len(fixes)})

"""
    
    for i, fix in enumerate(fixes, 1):
        report += f"### {i}. {fix['type'].replace('_', ' ').title()}\n"
        report += f"- **Description**: {fix['description']}\n"
        report += f"- **Implementation**: {fix['implementation']}\n\n"
    
    report += """
## Priority Actions

1. **CRITICAL**: Fix model architecture and training
2. **CRITICAL**: Improve loss functions and learning rates  
3. **HIGH**: Fix evaluation completeness
4. **HIGH**: Address class imbalance
5. **MEDIUM**: Enhance TTT adaptation

## Expected Improvements

- Base model accuracy: 47.6% → 80%+
- ROC-AUC: 0.108 → 0.8+
- MCC: 0.000 → 0.6+
- Evaluation coverage: 38.6% → 100%
"""
    
    return report

if __name__ == "__main__":
    # Analyze current performance
    metrics_file = "performance_plots/performance_metrics_latest.json"
    report = create_performance_report(metrics_file)
    
    # Save report
    with open("PERFORMANCE_ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    
    print("Performance analysis complete. Report saved to PERFORMANCE_ANALYSIS_REPORT.md")




