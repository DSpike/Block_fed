#!/usr/bin/env python3
"""
Hyperparameter Optimization Module for TCGAN/TCAE + TTT Architecture
Optimizes sequence length and stride for better rare attack capture
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import ParameterGrid
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WindowSizeOptimizer:
    """
    🚀 ENHANCEMENT 4: Hyperparameter search module for sequence length and stride optimization
    """
    
    def __init__(self, model_class, device: torch.device, 
                 sequence_length_range: Tuple[int, int] = (8, 64),
                 stride_range: Tuple[int, int] = (1, 8),
                 max_evaluations: int = 20):
        self.model_class = model_class
        self.device = device
        self.sequence_length_range = sequence_length_range
        self.stride_range = stride_range
        self.max_evaluations = max_evaluations
        
        # Optimization results
        self.optimization_history = []
        self.best_config = None
        self.best_score = -float('inf')
        
    def generate_parameter_combinations(self) -> List[Dict]:
        """
        Generate parameter combinations for grid search
        """
        # Create parameter grid
        param_grid = {
            'sequence_length': list(range(self.sequence_length_range[0], 
                                        self.sequence_length_range[1] + 1, 4)),
            'stride': list(range(self.stride_range[0], 
                               self.stride_range[1] + 1, 1))
        }
        
        # Generate combinations
        combinations = list(ParameterGrid(param_grid))
        
        # Filter valid combinations (stride should be <= sequence_length/2)
        valid_combinations = []
        for combo in combinations:
            if combo['stride'] <= combo['sequence_length'] // 2:
                valid_combinations.append(combo)
        
        # Limit to max_evaluations
        if len(valid_combinations) > self.max_evaluations:
            # Sample evenly across the space
            indices = np.linspace(0, len(valid_combinations) - 1, 
                                self.max_evaluations, dtype=int)
            valid_combinations = [valid_combinations[i] for i in indices]
        
        logger.info(f"🔍 Generated {len(valid_combinations)} parameter combinations")
        return valid_combinations
    
    def evaluate_configuration(self, config: Dict, train_data: torch.Tensor, 
                             train_labels: torch.Tensor, test_data: torch.Tensor,
                             test_labels: torch.Tensor) -> Dict:
        """
        Evaluate a single configuration
        """
        try:
            # Create model with current configuration
            model = self.model_class(
                input_dim=config.get('input_dim', 50),
                sequence_length=config['sequence_length'],
                stride=config['stride'],
                latent_dim=config.get('latent_dim', 64),
                hidden_dim=config.get('hidden_dim', 128),
                num_classes=config.get('num_classes', 2),
                noise_dim=config.get('noise_dim', 64),
                use_shallow_adaptation=config.get('use_shallow_adaptation', True)
            ).to(self.device)
            
            # Quick training evaluation (limited epochs for speed)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            
            # Train for a few epochs to get a performance estimate
            train_losses = []
            for epoch in range(3):  # Limited epochs for speed
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(train_data)
                loss = nn.CrossEntropyLoss()(outputs, train_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Quick evaluation on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_data)
                test_loss = nn.CrossEntropyLoss()(test_outputs, test_labels)
                
                # Compute accuracy
                predictions = torch.argmax(test_outputs, dim=1)
                accuracy = (predictions == test_labels).float().mean().item()
            
            # Compute score (combination of accuracy and training stability)
            avg_train_loss = np.mean(train_losses[-2:])  # Last 2 epochs
            loss_stability = 1.0 / (1.0 + np.std(train_losses))  # Higher is better
            
            # Combined score: accuracy + stability - test loss
            score = accuracy + 0.1 * loss_stability - 0.1 * test_loss.item()
            
            result = {
                'config': config,
                'accuracy': accuracy,
                'test_loss': test_loss.item(),
                'train_loss': avg_train_loss,
                'loss_stability': loss_stability,
                'score': score,
                'train_losses': train_losses
            }
            
            logger.info(f"✅ Config {config}: Acc={accuracy:.3f}, Score={score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Configuration {config} failed: {e}")
            return {
                'config': config,
                'accuracy': 0.0,
                'test_loss': float('inf'),
                'train_loss': float('inf'),
                'loss_stability': 0.0,
                'score': -float('inf'),
                'error': str(e)
            }
    
    def optimize_parallel(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                         test_data: torch.Tensor, test_labels: torch.Tensor,
                         max_workers: int = 4) -> Dict:
        """
        Parallel optimization of hyperparameters
        """
        logger.info("🚀 Starting parallel hyperparameter optimization")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations()
        
        # Evaluate configurations in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(
                    self.evaluate_configuration, 
                    config, train_data, train_labels, test_data, test_labels
                ): config for config in combinations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update best configuration
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_config = result['config']
                        logger.info(f"🏆 New best config: {self.best_config} (score: {self.best_score:.3f})")
                        
                except Exception as e:
                    logger.error(f"❌ Configuration {config} failed: {e}")
        
        # Store optimization history
        self.optimization_history = results
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Optimization completed. Best config: {self.best_config}")
        logger.info(f"📊 Best accuracy: {results[0]['accuracy']:.3f}, Score: {results[0]['score']:.3f}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': results,
            'optimization_summary': self._create_summary(results)
        }
    
    def optimize_sequential(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                           test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict:
        """
        Sequential optimization of hyperparameters (for debugging or limited resources)
        """
        logger.info("🚀 Starting sequential hyperparameter optimization")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations()
        
        # Evaluate configurations sequentially
        results = []
        for i, config in enumerate(combinations):
            logger.info(f"🔍 Evaluating configuration {i+1}/{len(combinations)}: {config}")
            
            result = self.evaluate_configuration(
                config, train_data, train_labels, test_data, test_labels
            )
            results.append(result)
            
            # Update best configuration
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_config = result['config']
                logger.info(f"🏆 New best config: {self.best_config} (score: {self.best_score:.3f})")
        
        # Store optimization history
        self.optimization_history = results
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Optimization completed. Best config: {self.best_config}")
        logger.info(f"📊 Best accuracy: {results[0]['accuracy']:.3f}, Score: {results[0]['score']:.3f}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': results,
            'optimization_summary': self._create_summary(results)
        }
    
    def _create_summary(self, results: List[Dict]) -> Dict:
        """
        Create optimization summary
        """
        if not results:
            return {}
        
        # Sort by score
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Top 5 configurations
        top_5 = results_sorted[:5]
        
        # Statistics
        accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
        scores = [r['score'] for r in results if 'score' in r]
        
        summary = {
            'total_configurations': len(results),
            'best_configuration': results_sorted[0]['config'],
            'best_accuracy': results_sorted[0]['accuracy'],
            'best_score': results_sorted[0]['score'],
            'top_5_configurations': top_5,
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        }
        
        return summary
    
    def save_results(self, filepath: str):
        """
        Save optimization results to file
        """
        results_data = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'optimization_summary': self._create_summary(self.optimization_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"💾 Optimization results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load optimization results from file
        """
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ Results file {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.best_config = results_data['best_config']
        self.best_score = results_data['best_score']
        self.optimization_history = results_data['optimization_history']
        
        logger.info(f"📂 Optimization results loaded from {filepath}")
        logger.info(f"🏆 Best config: {self.best_config} (score: {self.best_score})")

class AdaptiveWindowSizeOptimizer(WindowSizeOptimizer):
    """
    🚀 ENHANCEMENT 4: Adaptive optimization that adjusts search space based on results
    """
    
    def __init__(self, model_class, device: torch.device, **kwargs):
        super().__init__(model_class, device, **kwargs)
        self.adaptive_search = True
        self.search_iterations = 3
        
    def adaptive_optimization(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                             test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict:
        """
        Adaptive optimization with iterative refinement
        """
        logger.info("🚀 Starting adaptive hyperparameter optimization")
        
        current_sequence_range = self.sequence_length_range
        current_stride_range = self.stride_range
        
        all_results = []
        
        for iteration in range(self.search_iterations):
            logger.info(f"🔄 Adaptive iteration {iteration + 1}/{self.search_iterations}")
            logger.info(f"📏 Sequence range: {current_sequence_range}, Stride range: {current_stride_range}")
            
            # Update ranges for this iteration
            self.sequence_length_range = current_sequence_range
            self.stride_range = current_stride_range
            
            # Run optimization for this iteration
            iteration_results = self.optimize_sequential(
                train_data, train_labels, test_data, test_labels
            )
            
            all_results.extend(iteration_results['all_results'])
            
            # Refine search space based on results
            if iteration < self.search_iterations - 1:
                best_config = iteration_results['best_config']
                
                # Narrow search space around best configuration
                seq_center = best_config['sequence_length']
                stride_center = best_config['stride']
                
                # Reduce range by 50% for next iteration
                seq_half_range = (current_sequence_range[1] - current_sequence_range[0]) // 4
                stride_half_range = (current_stride_range[1] - current_stride_range[0]) // 2
                
                current_sequence_range = (
                    max(seq_center - seq_half_range, self.sequence_length_range[0]),
                    min(seq_center + seq_half_range, self.sequence_length_range[1])
                )
                current_stride_range = (
                    max(stride_center - stride_half_range, self.stride_range[0]),
                    min(stride_center + stride_half_range, self.stride_range[1])
                )
        
        # Combine all results and find global best
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        self.best_config = all_results[0]['config']
        self.best_score = all_results[0]['score']
        self.optimization_history = all_results
        
        logger.info(f"✅ Adaptive optimization completed")
        logger.info(f"🏆 Global best config: {self.best_config} (score: {self.best_score:.3f})")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': all_results,
            'optimization_summary': self._create_summary(all_results)
        }

# Example usage and testing
if __name__ == "__main__":
    # This would be used in the main system
    logger.info("🚀 Window Size Optimizer ready for integration")
    logger.info("📋 Usage: optimizer.optimize_parallel(train_data, train_labels, test_data, test_labels)")


