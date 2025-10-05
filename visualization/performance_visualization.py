#!/usr/bin/env python3
"""
Performance Visualization Module for Blockchain Federated Learning System
Provides comprehensive plotting and visualization of performance metrics
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set IEEE standard plotting style
plt.style.use('default')  # Reset to default style
sns.set_palette("husl")

class PerformanceVisualizer:
    """
    Comprehensive performance visualization for blockchain federated learning system
    """
    
    def __init__(self, output_dir: str = "performance_plots", attack_name: str = ""):
        """
        Initialize the performance visualizer
        
        Args:
            output_dir: Directory to save plots
            attack_name: Name of the attack type for plot identification
        """
        self.output_dir = output_dir
        self.attack_name = attack_name
        # Use fixed filenames to avoid accumulation
        self.timestamp = "latest"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_filename(self, plot_type: str) -> str:
        """Generate filename including attack name"""
        if self.attack_name:
            return f"{plot_type}_{self.attack_name}_{self.timestamp}.png"
        else:
            return f"{plot_type}_{self.timestamp}.png"
    
    def _get_title_suffix(self) -> str:
        """Generate title suffix including attack name"""
        if self.attack_name:
            return f" ({self.attack_name} Attack)"
        else:
            return ""
        
        # Set IEEE top-tier paper standard parameters
        plt.rcParams.update({
            'font.family': 'Times New Roman',  # IEEE standard serif font
            'font.size': 10,                   # IEEE standard font size
            'axes.linewidth': 1.2,             # Professional axes thickness
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'grid.linewidth': 0.6,             # Subtle grid
            'grid.alpha': 0.3,                 # Light grid
            'lines.linewidth': 2.0,            # Professional line thickness
            'lines.markersize': 6,             # Appropriate marker size
            'xtick.labelsize': 9,              # Consistent tick label size
            'ytick.labelsize': 9,
            'legend.fontsize': 9,              # Legend font size
            'figure.titlesize': 12,            # Figure title size
            'axes.titlesize': 11,              # Subplot title size
            'figure.dpi': 300                  # High resolution for IEEE papers
        })
        
        logger.info(f"Performance visualizer initialized. Output directory: {output_dir}")
    
    def plot_training_history(self, training_history: Dict[str, List], save: bool = True) -> str:
        """
        Plot training history (loss and accuracy over epochs)
        
        Args:
            training_history: Dictionary with 'epoch_losses' and 'epoch_accuracies'
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(training_history['epoch_losses']) + 1)
        
        # Plot training loss
        ax1.plot(epochs, training_history['epoch_losses'], 'b-', linewidth=2, marker='o', markersize=6)
        ax1.set_title(f'Federated Training Loss Over Rounds{self._get_title_suffix()}', fontweight='bold', fontfamily='Times New Roman')
        ax1.set_xlabel('Round', fontfamily='Times New Roman')
        ax1.set_ylabel('Average Loss', fontfamily='Times New Roman')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add value labels for loss
        for i, loss in enumerate(training_history['epoch_losses']):
            ax1.annotate(f'{loss:.4f}', (epochs[i], loss), 
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=8, fontfamily='Times New Roman')
        
        # Plot training accuracy
        ax2.plot(epochs, training_history['epoch_accuracies'], 'g-', linewidth=2, marker='s', markersize=6)
        ax2.set_title('Federated Training Accuracy Over Rounds', fontweight='bold', fontfamily='Times New Roman')
        ax2.set_xlabel('Round', fontfamily='Times New Roman')
        ax2.set_ylabel('Average Accuracy', fontfamily='Times New Roman')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels for accuracy
        for i, acc in enumerate(training_history['epoch_accuracies']):
            ax2.annotate(f'{acc:.3f}', (epochs[i], acc), 
                        textcoords="offset points", xytext=(0,10), ha='center',
                        fontsize=8, fontfamily='Times New Roman')
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, self._get_filename("training_history"))
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    def plot_ttt_adaptation(self, ttt_adaptation_data: Dict, save: bool = True) -> str:
        """
        Plot TTT adaptation process showing loss evolution
        
        Args:
            ttt_adaptation_data: Dictionary with TTT adaptation data
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not ttt_adaptation_data:
            logger.warning("No TTT adaptation data provided for plotting")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        steps = ttt_adaptation_data['steps']
        total_losses = ttt_adaptation_data['total_losses']
        support_losses = ttt_adaptation_data['support_losses']
        consistency_losses = ttt_adaptation_data['consistency_losses']
        
        # Plot 1: Total Loss Evolution
        ax1.plot(steps, total_losses, 'b-', linewidth=2, marker='o', markersize=6, label='Total Loss')
        ax1.set_title('TTT Adaptation: Total Loss Evolution', fontweight='bold', fontfamily='Times New Roman')
        ax1.set_xlabel('TTT Step', fontfamily='Times New Roman')
        ax1.set_ylabel('Loss Value', fontfamily='Times New Roman')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend(prop={'family': 'Times New Roman'})
        
        # Add value labels for key points
        for i, loss in enumerate(total_losses):
            if i % 2 == 0:  # Show every other point to avoid clutter
                ax1.annotate(f'{loss:.2e}', (steps[i], loss), 
                            textcoords="offset points", xytext=(0,10), ha='center',
                            fontsize=8, fontfamily='Times New Roman')
        
        # Plot 2: Loss Components
        ax2.plot(steps, support_losses, 'g-', linewidth=2, marker='s', markersize=6, label='Support Loss')
        ax2.plot(steps, consistency_losses, 'r-', linewidth=2, marker='^', markersize=6, label='Consistency Loss')
        ax2.set_title('TTT Adaptation: Loss Components', fontweight='bold', fontfamily='Times New Roman')
        ax2.set_xlabel('TTT Step', fontfamily='Times New Roman')
        ax2.set_ylabel('Loss Value', fontfamily='Times New Roman')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.legend(prop={'family': 'Times New Roman'})
        
        # Add improvement annotations
        if len(total_losses) > 1:
            initial_loss = total_losses[0]
            final_loss = total_losses[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            ax1.annotate(f'Improvement: {improvement:.1f}%', 
                        xy=(steps[-1], final_loss), 
                        xytext=(steps[-1] - 2, final_loss * 1.5),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8, 
                                 edgecolor='black', linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', lw=1, color='black'))
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"ttt_adaptation_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"TTT adaptation plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    
    # plot_zero_day_detection_metrics method removed - not properly plotting
    
    def plot_client_performance(self, client_results: List[Dict], save: bool = True) -> str:
        """
        Plot individual client performance metrics
        
        Args:
            client_results: List of client evaluation results
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not client_results:
            logger.warning("No client results provided for plotting")
            return ""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Extract client data
        client_ids = [r.get('client_id', f'Client_{i}') for i, r in enumerate(client_results)]
        accuracies = [r.get('accuracy', 0) for r in client_results]
        f1_scores = [r.get('f1_score', 0) for r in client_results]
        precisions = [r.get('precision', 0) for r in client_results]
        recalls = [r.get('recall', 0) for r in client_results]
        
        # Plot client accuracy comparison
        bars1 = ax1.bar(client_ids, accuracies, color='skyblue', alpha=0.8)
        ax1.set_title('Client Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot F1-scores
        bars2 = ax2.bar(client_ids, f1_scores, color='lightgreen', alpha=0.8)
        ax2.set_title('Client F1-Score Comparison', fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot average performance across all clients
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        avg_values = [np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)]
        
        bars3 = ax3.bar(metrics, avg_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
        ax3.set_title('Average Performance Across All Clients', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars3, avg_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"client_performance_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Client performance plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    def plot_confusion_matrix(self, confusion_matrix_data: np.ndarray, 
                            class_names: List[str] = ['Normal', 'Attack'], 
                            save: bool = True) -> str:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix_data: 2D numpy array of confusion matrix
            class_names: List of class names
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix - Zero-Day Detection', fontweight='bold', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"confusion_matrix_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    def plot_blockchain_metrics(self, blockchain_data: Dict, save: bool = True) -> str:
        """
        Plot blockchain-related metrics
        
        Args:
            blockchain_data: Dictionary with blockchain metrics
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not blockchain_data:
            logger.warning("No blockchain data provided for plotting")
            return ""
        
        # Check if we have real data or empty data
        gas_used = blockchain_data.get('gas_used', [])
        logger.info(f"🔍 DEBUG: plot_blockchain_metrics - gas_used type: {type(gas_used)}, length: {len(gas_used) if hasattr(gas_used, '__len__') else 'no length'}, value: {gas_used}")
        if not gas_used:
            logger.warning("No gas usage data available - blockchain transactions may not have been recorded")
            # Create empty plot with message
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Blockchain Metrics - No Data Available', fontsize=16, fontweight='bold')
            
            ax1.text(0.5, 0.5, 'No Gas Data\nAvailable', ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax1.set_title('Gas Usage', fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            ax2.text(0.5, 0.5, 'No IPFS Data\nAvailable', ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax2.set_title('IPFS Storage', fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            ax3.text(0.5, 0.5, 'No Transaction\nData Available', ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax3.set_title('Transaction Count', fontweight='bold')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            
            plt.tight_layout()
            
            if save:
                plot_path = os.path.join(self.output_dir, f'blockchain_metrics_{self.timestamp}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Empty blockchain metrics plot saved: {plot_path}")
            
            plt.close()
            return plot_path if save else ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract blockchain metrics
        transactions = blockchain_data.get('transactions', [])
        ipfs_cids = blockchain_data.get('ipfs_cids', [])
        gas_used = blockchain_data.get('gas_used', [])
        block_numbers = blockchain_data.get('block_numbers', [])
        
        # Plot transactions over time
        if transactions:
            ax1.plot(range(len(transactions)), transactions, 'b-', linewidth=2, marker='o')
            ax1.set_title('Blockchain Transactions Over Time', fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Number of Transactions')
            ax1.grid(True, alpha=0.3)
        
        # Plot IPFS CIDs (as a count)
        if ipfs_cids:
            ax2.bar(range(len(ipfs_cids)), [1] * len(ipfs_cids), color='lightgreen', alpha=0.7)
            ax2.set_title('IPFS Model Storage Events', fontweight='bold')
            ax2.set_xlabel('Storage Event')
            ax2.set_ylabel('Models Stored')
            ax2.set_ylim(0, 1.2)
        
        # Plot gas usage
        if gas_used:
            ax3.bar(range(len(gas_used)), gas_used, color='orange', alpha=0.7)
            ax3.set_title('Gas Usage Per Transaction', fontweight='bold')
            ax3.set_xlabel('Transaction')
            ax3.set_ylabel('Gas Used')
            ax3.grid(True, alpha=0.3)
        
        # Plot block numbers
        if block_numbers:
            ax4.plot(range(len(block_numbers)), block_numbers, 'r-', linewidth=2, marker='s')
            ax4.set_title('Block Numbers Over Time', fontweight='bold')
            ax4.set_xlabel('Transaction')
            ax4.set_ylabel('Block Number')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"blockchain_metrics_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Blockchain metrics plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    def plot_gas_usage_analysis(self, blockchain_data: Dict, save: bool = True) -> str:
        """
        Plot detailed gas usage analysis with bar charts
        
        Args:
            blockchain_data: Dictionary with blockchain metrics
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not blockchain_data:
            logger.warning("No blockchain data provided for gas usage plotting")
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Blockchain Gas Usage Analysis', fontsize=16, fontweight='bold')
        
        # Extract gas usage data
        gas_used = blockchain_data.get('gas_used', [])
        transactions = blockchain_data.get('transactions', [])
        transaction_types = blockchain_data.get('transaction_types', [])
        rounds = blockchain_data.get('rounds', [])
        logger.info(f"🔍 DEBUG: plot_gas_usage_analysis - gas_used: {gas_used}, transactions: {len(transactions)}, types: {len(transaction_types)}, rounds: {len(rounds)}")
        
        # If no gas data, return empty plot with warning
        if not gas_used:
            logger.warning("No real gas data available for visualization - skipping gas usage analysis plot")
            # Create empty plot with message
            ax1.text(0.5, 0.5, 'No Gas Data Available\nReal blockchain transactions\nnot recorded during training', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax1.set_title('Gas Usage by Transaction Type - No Data', fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            ax2.text(0.5, 0.5, 'No Gas Data Available\nReal blockchain transactions\nnot recorded during training', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax2.set_title('Gas Usage by Round - No Data', fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            ax3.text(0.5, 0.5, 'No Gas Data Available\nReal blockchain transactions\nnot recorded during training', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax3.set_title('Individual Transaction Gas Usage - No Data', fontweight='bold')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            
            ax4.text(0.5, 0.5, 'No Gas Data Available\nReal blockchain transactions\nnot recorded during training', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Gas Usage Statistics - No Data', fontweight='bold')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            
            if save:
                plot_path = os.path.join(self.output_dir, f'gas_usage_analysis_{self.timestamp}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Empty gas usage analysis plot saved: {plot_path}")
            
            plt.close()
            return plot_path if save else ""
        
        # Plot 1: Gas Usage by Transaction Type
        if transaction_types:
            type_gas = {}
            for gas, tx_type in zip(gas_used, transaction_types):
                if tx_type not in type_gas:
                    type_gas[tx_type] = []
                type_gas[tx_type].append(gas)
            
            types = list(type_gas.keys())
            avg_gas = [np.mean(type_gas[t]) for t in types]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars1 = ax1.bar(types, avg_gas, color=colors[:len(types)], alpha=0.8, edgecolor='black', linewidth=1)
            ax1.set_title('Average Gas Usage by Transaction Type', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Gas Used (units)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars1, avg_gas):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Gas Usage by Round
        if rounds:
            round_gas = {}
            for gas, round_name in zip(gas_used, rounds):
                if round_name not in round_gas:
                    round_gas[round_name] = []
                round_gas[round_name].append(gas)
            
            round_names = list(round_gas.keys())
            total_gas_per_round = [sum(round_gas[r]) for r in round_names]
            colors2 = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            
            bars2 = ax2.bar(round_names, total_gas_per_round, color=colors2[:len(round_names)], 
                           alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_title('Total Gas Usage per Federated Round', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Total Gas Used (units)')
            ax2.set_xlabel('Federated Round')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars2, total_gas_per_round):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Gas Usage Trend Over Time
        if gas_used:
            x_pos = range(len(gas_used))
            colors3 = ['#FF6B6B' if 'Client' in str(tx_type) else 
                      '#4ECDC4' if 'Aggregation' in str(tx_type) else 
                      '#45B7D1' for tx_type in transaction_types] if transaction_types else ['#FF6B6B'] * len(gas_used)
            
            bars3 = ax3.bar(x_pos, gas_used, color=colors3, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax3.set_title('Gas Usage Trend Over All Transactions', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Transaction Number')
            ax3.set_ylabel('Gas Used (units)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add round separators
            if rounds:
                round_starts = []
                current_round = rounds[0]
                for i, round_name in enumerate(rounds):
                    if round_name != current_round:
                        round_starts.append(i)
                        current_round = round_name
                
                for start in round_starts:
                    ax3.axvline(x=start, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Plot 4: Gas Usage Statistics
        if gas_used:
            stats_data = {
                'Min': min(gas_used),
                'Max': max(gas_used),
                'Mean': np.mean(gas_used),
                'Median': np.median(gas_used),
                'Total': sum(gas_used)
            }
            
            stat_names = list(stats_data.keys())
            stat_values = list(stats_data.values())
            colors4 = ['#FF9999', '#FF6666', '#66B2FF', '#99FF99', '#FFCC99']
            
            bars4 = ax4.bar(stat_names, stat_values, color=colors4, alpha=0.8, 
                           edgecolor='black', linewidth=1)
            ax4.set_title('Gas Usage Statistics', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Gas Units')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars4, stat_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f'gas_usage_analysis_{self.timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gas usage analysis plot saved: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.close()  # Close figure instead of showing it
            return ""
    
    def create_comprehensive_report(self, system_data: Dict, save: bool = True) -> str:
        """
        Create a comprehensive performance report with all plots
        
        Args:
            system_data: Complete system data dictionary
            save: Whether to save the report
            
        Returns:
            report_path: Path to saved report
        """
        logger.info("Creating comprehensive performance report...")
        logger.info("Creating optimized figure (12x16 inches) with 9 subplots...")
        
        # Create an optimized figure with multiple subplots (reduced size for faster generation)
        fig = plt.figure(figsize=(12, 16))
        
        # Define grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        logger.info("Grid layout created")
        
        # Extract data
        training_history = system_data.get('training_history', {})
        round_results = system_data.get('round_results', [])
        evaluation_results = system_data.get('evaluation_results', {})
        client_results = system_data.get('client_results', [])
        blockchain_data = system_data.get('blockchain_data', {})
        
        # Plot 1: Training History
        if training_history:
            ax1 = fig.add_subplot(gs[0, 0])
            epochs = range(1, len(training_history.get('epoch_losses', [])) + 1)
            ax1.plot(epochs, training_history.get('epoch_losses', []), 'b-', linewidth=2, marker='o')
            ax1.set_title('Meta-Training Loss', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        
        # Plot 2: Training Accuracy
        if training_history:
            ax2 = fig.add_subplot(gs[0, 1])
            epochs = range(1, len(training_history.get('epoch_accuracies', [])) + 1)
            ax2.plot(epochs, training_history.get('epoch_accuracies', []), 'g-', linewidth=2, marker='s')
            ax2.set_title('Meta-Training Accuracy', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
        
        # Plot 3: Federated Rounds
        if round_results:
            ax3 = fig.add_subplot(gs[0, 2])
            rounds = [r.get('round_number', i+1) for i, r in enumerate(round_results)]
            
            # Extract accuracies from the correct structure
            accuracies = []
            for r in round_results:
                if isinstance(r, dict) and 'client_updates' in r:
                    # Calculate average accuracy from client updates
                    client_accuracies = []
                    for update in r['client_updates']:
                        if hasattr(update, 'validation_accuracy'):
                            client_accuracies.append(update.validation_accuracy)
                    
                    if client_accuracies:
                        accuracies.append(sum(client_accuracies) / len(client_accuracies))
                    else:
                        accuracies.append(0.0)
                else:
                    accuracies.append(0.0)
            
            ax3.plot(rounds, accuracies, 'purple', linewidth=2, marker='o')
            ax3.set_title('Federated Learning Accuracy', fontweight='bold')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Accuracy')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
        
        # Plot 4: Zero-Day Detection Metrics
        if evaluation_results:
            ax4 = fig.add_subplot(gs[1, 0])
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [
                evaluation_results.get('accuracy', 0),
                evaluation_results.get('precision', 0),
                evaluation_results.get('recall', 0),
                evaluation_results.get('f1_score', 0)
            ]
            bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax4.set_title('Zero-Day Detection Metrics', fontweight='bold')
            ax4.set_ylabel('Score')
            ax4.set_ylim(0, 1.1)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Client Performance
        if client_results:
            ax5 = fig.add_subplot(gs[1, 1])
            client_ids = [r.get('client_id', f'Client_{i}') for i, r in enumerate(client_results)]
            accuracies = [r.get('accuracy', 0) for r in client_results]
            bars = ax5.bar(client_ids, accuracies, color='lightblue', alpha=0.8)
            ax5.set_title('Client Accuracy Comparison', fontweight='bold')
            ax5.set_ylabel('Accuracy')
            ax5.set_ylim(0, 1.1)
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, accuracies):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Zero-Day Detection Rate
        if evaluation_results:
            ax6 = fig.add_subplot(gs[1, 2])
            zero_day_rate = evaluation_results.get('zero_day_detection_rate', 0)
            ax6.pie([zero_day_rate, 1-zero_day_rate], 
                   labels=['Zero-Day Detected', 'Normal/Attack'], 
                   autopct='%1.1f%%', 
                   colors=['red', 'lightblue'],
                   startangle=90)
            ax6.set_title('Zero-Day Detection Distribution', fontweight='bold')
        
        # Plot 7: Blockchain Transactions
        if blockchain_data:
            ax7 = fig.add_subplot(gs[2, 0])
            transactions = blockchain_data.get('transactions', [])
            if transactions:
                ax7.plot(range(len(transactions)), transactions, 'b-', linewidth=2, marker='o')
                ax7.set_title('Blockchain Transactions', fontweight='bold')
                ax7.set_xlabel('Time Step')
                ax7.set_ylabel('Transactions')
                ax7.grid(True, alpha=0.3)
        
        # Plot 8: IPFS Storage
        if blockchain_data:
            ax8 = fig.add_subplot(gs[2, 1])
            ipfs_cids = blockchain_data.get('ipfs_cids', [])
            if ipfs_cids:
                ax8.bar(range(len(ipfs_cids)), [1] * len(ipfs_cids), color='lightgreen', alpha=0.7)
                ax8.set_title('IPFS Model Storage', fontweight='bold')
                ax8.set_xlabel('Storage Event')
                ax8.set_ylabel('Models Stored')
                ax8.set_ylim(0, 1.2)
        
        # Plot 9: System Summary
        ax9 = fig.add_subplot(gs[2, 2])
        summary_data = {
            'Training Rounds': len(round_results),
            'Clients': len(client_results),
            'Zero-Day Rate': evaluation_results.get('zero_day_detection_rate', 0),
            'Final Accuracy': evaluation_results.get('accuracy', 0)
        }
        
        metrics = list(summary_data.keys())
        values = list(summary_data.values())
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        bars = ax9.bar(metrics, values, color=colors, alpha=0.8)
        ax9.set_title('System Summary', fontweight='bold')
        ax9.set_ylabel('Value')
        ax9.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add overall title
        fig.suptitle('Blockchain Federated Learning System - Comprehensive Performance Report', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            report_path = os.path.join(self.output_dir, f"comprehensive_report_{self.timestamp}.png")
            logger.info("Saving comprehensive report figure...")
            plt.savefig(report_path, dpi=150, bbox_inches='tight')  # Reduced DPI for faster generation
            logger.info(f"Comprehensive report saved: {report_path}")
        
        logger.info("Closing comprehensive report figure...")
        plt.close()  # Close figure instead of showing it
        logger.info("Comprehensive report generation completed")
        return report_path if save else ""
    
    def save_metrics_to_json(self, system_data: Dict, filename: str = None) -> str:
        """
        Save all metrics to a JSON file
        
        Args:
            system_data: Complete system data dictionary
            filename: Optional custom filename
            
        Returns:
            json_path: Path to saved JSON file
        """
        if filename is None:
            filename = f"performance_metrics_{self.timestamp}.json"
        
        json_path = os.path.join(self.output_dir, filename)
        
        # Add timestamp to data
        system_data['timestamp'] = self.timestamp
        system_data['generated_at'] = datetime.now().isoformat()
        
        with open(json_path, 'w') as f:
            json.dump(system_data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics saved to JSON: {json_path}")
        return json_path
    
    def plot_confusion_matrices(self, evaluation_results: Dict, save: bool = True, title_suffix: str = "") -> str:
        """
        Plot confusion matrices for zero-day detection evaluation
        
        Args:
            evaluation_results: Evaluation results containing confusion matrix data
            save: Whether to save the plot
            title_suffix: Suffix to add to the plot title
            
        Returns:
            plot_path: Path to saved plot
        """
        if not evaluation_results:
            logger.warning("No evaluation results provided for confusion matrix plotting")
            return ""
        
        # Extract confusion matrix data
        confusion_data = evaluation_results.get('confusion_matrix', {})
        if not confusion_data:
            logger.warning("No confusion matrix data found in evaluation results")
            return ""
        
        # Create figure with IEEE standard styling
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract confusion matrix values
        tn = confusion_data.get('tn', 0)
        fp = confusion_data.get('fp', 0)
        fn = confusion_data.get('fn', 0)
        tp = confusion_data.get('tp', 0)
        
        # Create confusion matrix array
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = ['Normal', 'Attack']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        # Add performance metrics as text
        accuracy = evaluation_results.get('accuracy', 0)
        precision = evaluation_results.get('precision', 0)
        recall = evaluation_results.get('recall', 0)
        f1_score = evaluation_results.get('f1_score', 0)
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set title and labels
        title = f'Confusion Matrix - Zero-Day Detection{title_suffix}'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Apply IEEE standard styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout()
        
        if save:
            # Generate filename based on title suffix
            if title_suffix:
                model_type = title_suffix.replace(" - ", "_").replace(" ", "_").lower()
                filename = f"confusion_matrices_{model_type}_{self.timestamp}.png"
            else:
                filename = f"confusion_matrices_{self.timestamp}.png"
            
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Confusion matrix plot saved: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.close()  # Close figure instead of showing it
            return ""
        
        # Set tick labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'], fontsize=11)
        ax.set_yticklabels(['Normal', 'Attack'], fontsize=11)
        
        # Add performance metrics as text
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                fontsize=10, fontweight='bold', fontfamily='Times New Roman',
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5))
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"confusion_matrices_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Confusion matrices plot saved: {plot_path}")
            plt.close()
            return plot_path
        else:
            plt.close()  # Close figure instead of showing it
            return ""
    
    def plot_performance_comparison_with_annotations(self, base_results: Dict, ttt_results: Dict, 
                                                   scenario_names: List[str] = None, save: bool = True) -> str:
        """
        Plot performance comparison with advanced annotations showing improvements/decreases
        
        Args:
            base_results: Base model results dictionary
            ttt_results: TTT model results dictionary  
            scenario_names: Names of test scenarios
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not base_results or not ttt_results:
            logger.warning("No comparison data provided for plotting")
            return ""
        
        # Set up scenario names
        if scenario_names is None:
            scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5']
        
        # Extract metrics for comparison
        base_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mccc']
        ttt_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'mccc']
        
        base_values = [base_results.get(metric, 0) for metric in base_metrics]
        ttt_values = [ttt_results.get(metric, 0) for metric in ttt_metrics]
        
        # Create figure with IEEE standard styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Performance Comparison
        x = np.arange(len(base_metrics))
        width = 0.35
        
        # Add attack name to labels if available
        base_label = f'Base Model [{self.attack_name}]' if self.attack_name else 'Base Model'
        ttt_label = f'TTT Model [{self.attack_name}]' if self.attack_name else 'TTT Model'
        
        bars1 = ax1.bar(x - width/2, base_values, width, label=base_label, 
                       color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, ttt_values, width, label=ttt_label, 
                       color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels with professional formatting
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                        f'{value:.3f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, 
                                 edgecolor='black', linewidth=0.5))
        
        add_value_labels(bars1, base_values)
        add_value_labels(bars2, ttt_values)
        
        # Add performance improvement annotations
        for i, (base_val, ttt_val) in enumerate(zip(base_values, ttt_values)):
            if base_val > 0:
                improvement = ((ttt_val - base_val) / base_val) * 100
                if abs(improvement) > 1:  # Only show significant improvements
                    # Position annotation at the middle of the TTT bar
                    ttt_bar_center = x[i] + width/2
                    color = 'green' if improvement > 0 else 'red'
                    ax1.annotate(f'{improvement:+.1f}%', 
                               xy=(ttt_bar_center, ttt_val + 0.05), 
                               xytext=(ttt_bar_center, ttt_val + 0.1),
                               ha='center', va='bottom',
                               fontsize=8, fontweight='bold', fontfamily='Times New Roman',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8, 
                                        edgecolor='black', linewidth=0.5),
                               arrowprops=dict(arrowstyle='->', lw=1, color='black'))
        
        ax1.set_xlabel('Performance Metrics', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
        ax1.set_ylabel('Score', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
        # Add attack name to main title
        title_suffix = f" ({self.attack_name} Attack)" if self.attack_name else ""
        ax1.set_title(f'Performance Comparison: Base vs TTT Model{title_suffix}', fontsize=13, fontweight='bold', 
                     pad=15, fontfamily='Times New Roman')
        ax1.set_xticks(x)
        # Fix MCCC label to MCC
        metric_labels = []
        for m in base_metrics:
            if m == 'mccc':
                metric_labels.append('MCC')
            else:
                metric_labels.append(m.replace('_', ' ').title())
        
        ax1.set_xticklabels(metric_labels, fontfamily='Times New Roman')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.2)
        
        # Plot 2: Performance Improvement Analysis
        improvements = []
        for i in range(len(base_values)):
            if base_values[i] > 0:
                improvement = ((ttt_values[i] - base_values[i]) / base_values[i]) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        # Color-code improvements
        improvement_colors = ['#28a745' if imp >= 0 else '#dc3545' for imp in improvements]
        
        bars3 = ax2.bar(range(len(improvements)), improvements, color=improvement_colors, 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add improvement value labels
        for bar, improvement in zip(bars3, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (1 if height >= 0 else -3),
                    f'{improvement:+.1f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold', fontfamily='Times New Roman',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, 
                             edgecolor='black', linewidth=0.5))
        
        ax2.set_xlabel('Performance Metrics', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
        ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
        ax2.set_title('Performance Improvement Analysis', fontsize=13, fontweight='bold', 
                     pad=15, fontfamily='Times New Roman')
        ax2.set_xticks(range(len(improvements)))
        # Fix MCCC label to MCC for improvement analysis
        ax2.set_xticklabels(metric_labels, fontfamily='Times New Roman')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics
        avg_improvement = np.mean(improvements)
        max_improvement = np.max(improvements)
        min_improvement = np.min(improvements)
        
        summary_text = f'Avg: {avg_improvement:+.1f}%\nMax: {max_improvement:+.1f}%\nMin: {min_improvement:+.1f}%'
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
                fontsize=9, fontweight='bold', fontfamily='Times New Roman',
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5))
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, self._get_filename("performance_comparison_annotated"))
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Performance comparison plot with annotations saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""
    
    def plot_training_trends_with_annotations(self, training_history: Dict, save: bool = True) -> str:
        """
        Plot training trends with trend annotations
        
        Args:
            training_history: Training history dictionary
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        if not training_history:
            logger.warning("No training history provided for plotting")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(training_history['epoch_losses']) + 1)
        losses = training_history['epoch_losses']
        accuracies = training_history['epoch_accuracies']
        
        # Plot 1: Loss with trend annotations
        ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=6)
        
        # Add trend annotations
        if len(losses) > 1:
            # Calculate trend
            trend_slope = (losses[-1] - losses[0]) / (len(losses) - 1)
            trend_direction = "Decreasing" if trend_slope < 0 else "Increasing"
            trend_color = "green" if trend_slope < 0 else "red"
            
            # Add trend arrow
            ax1.annotate(f'{trend_direction} Trend', 
                        xy=(epochs[-1], losses[-1]), 
                        xytext=(epochs[-1] - 2, losses[-1] + max(losses) * 0.1),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=trend_color, alpha=0.8, 
                                 edgecolor='black', linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color=trend_color))
        
        ax1.set_title('Training Loss with Trend Analysis', fontweight='bold', fontfamily='Times New Roman')
        ax1.set_xlabel('Epoch', fontfamily='Times New Roman')
        ax1.set_ylabel('Loss', fontfamily='Times New Roman')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Accuracy with trend annotations
        ax2.plot(epochs, accuracies, 'g-', linewidth=2, marker='s', markersize=6)
        
        # Add trend annotations
        if len(accuracies) > 1:
            # Calculate trend
            trend_slope = (accuracies[-1] - accuracies[0]) / (len(accuracies) - 1)
            trend_direction = "Improving" if trend_slope > 0 else "Declining"
            trend_color = "green" if trend_slope > 0 else "red"
            
            # Add trend arrow
            ax2.annotate(f'{trend_direction} Trend', 
                        xy=(epochs[-1], accuracies[-1]), 
                        xytext=(epochs[-1] - 2, accuracies[-1] - 0.1),
                        ha='center', va='top',
                        fontsize=10, fontweight='bold', fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=trend_color, alpha=0.8, 
                                 edgecolor='black', linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', lw=2, color=trend_color))
        
        ax2.set_title('Training Accuracy with Trend Analysis', fontweight='bold', fontfamily='Times New Roman')
        ax2.set_xlabel('Epoch', fontfamily='Times New Roman')
        ax2.set_ylabel('Accuracy', fontfamily='Times New Roman')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"training_trends_annotated_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Training trends plot with annotations saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""

    def plot_roc_curves(self, base_results: Dict, ttt_results: Dict, save: bool = True) -> str:
        """
        Plot ROC curves for both base and TTT models
        
        Args:
            base_results: Base model evaluation results
            ttt_results: TTT model evaluation results
            save: Whether to save the plot
            
        Returns:
            plot_path: Path to saved plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot base model ROC curve
        if 'roc_curve' in base_results:
            base_fpr = base_results['roc_curve']['fpr']
            base_tpr = base_results['roc_curve']['tpr']
            base_auc = base_results.get('roc_auc', 0)
            base_threshold = base_results.get('optimal_threshold', 0.5)
            
            ax.plot(base_fpr, base_tpr, 'b-', linewidth=2, 
                   label=f'Base Model (AUC = {base_auc:.3f}, Threshold = {base_threshold:.3f})')
        
        # Plot TTT model ROC curve
        if 'roc_curve' in ttt_results:
            ttt_fpr = ttt_results['roc_curve']['fpr']
            ttt_tpr = ttt_results['roc_curve']['tpr']
            ttt_auc = ttt_results.get('roc_auc', 0)
            ttt_threshold = ttt_results.get('optimal_threshold', 0.5)
            
            ax.plot(ttt_fpr, ttt_tpr, 'r-', linewidth=2,
                   label=f'TTT Model (AUC = {ttt_auc:.3f}, Threshold = {ttt_threshold:.3f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        # Customize plot
        ax.set_xlabel('False Positive Rate', fontfamily='Times New Roman', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontfamily='Times New Roman', fontsize=12)
        ax.set_title('ROC Curves Comparison: Base vs TTT Models', fontweight='bold', fontfamily='Times New Roman', fontsize=14)
        ax.legend(prop={'family': 'Times New Roman'}, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add performance annotations
        if 'roc_curve' in base_results and 'roc_curve' in ttt_results:
            base_auc = base_results.get('roc_auc', 0)
            ttt_auc = ttt_results.get('roc_auc', 0)
            improvement = ttt_auc - base_auc
            
            ax.text(0.6, 0.2, f'AUC Improvement: {improvement:+.3f}', 
                   fontsize=10, fontfamily='Times New Roman',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(self.output_dir, f"roc_curves_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"ROC curves plot saved: {plot_path}")
        
        plt.close()  # Close figure instead of showing it
        return plot_path if save else ""

    def plot_token_distribution(self, incentive_data: Dict[str, Any], save: bool = True) -> str:
        """
        Create a simple single bar chart showing total tokens distributed to each client
        
        Args:
            incentive_data: Dictionary containing token distribution data
            save: Whether to save the plot
            
        Returns:
            str: Path to saved plot or empty string
        """
        try:
            # Extract data from incentive_data
            participant_rewards = incentive_data.get('participant_rewards', {})
            total_rewards = incentive_data.get('total_rewards_distributed', 0)
            
            if not participant_rewards:
                logger.warning("No token distribution data available for visualization")
                return ""
            
            # Create a single figure with one bar chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle('Total ERC-20 Token Distribution to Clients', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
            
            # Sort participants by token amount (descending)
            sorted_participants = sorted(participant_rewards.items(), key=lambda x: x[1], reverse=True)
            participant_names = [f"Client {i+1}" for i in range(len(sorted_participants))]
            token_amounts = [amount for _, amount in sorted_participants]
            
            # Create bar chart with IEEE-standard styling
            bars = ax.bar(participant_names, token_amounts, 
                         color=['#FCF4A4',  '#D1D0C8', '#6A642A', '#d62728', '#9467bd', '#8c564b'][:len(participant_names)],
                         alpha=0.8, edgecolor='none', linewidth=1.2)
                         
            
            ax.set_xlabel('Federated Learning Clients', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
            ax.set_ylabel('Total Tokens Received', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
            ax.set_title('Cumulative Token Distribution Across All Rounds', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Add value labels on top of bars
            for bar, value in zip(bars, token_amounts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(token_amounts)*0.01,
                       f'{value:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            
            # Add statistics text box
            if total_rewards > 0:
                stats_text = f"""
Token Distribution Summary:
• Total Tokens Distributed: {total_rewards:,.0f}
• Number of Participants: {len(participant_rewards)}
• Average per Client: {total_rewards/len(participant_rewards):,.0f}
• Incentive Mechanism: Performance-based ERC-20
                """
                fig.text(0.02, 0.02, stats_text, fontsize=11, fontfamily='Times New Roman',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                        verticalalignment='bottom')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for statistics text
            
            if save:
                plot_path = os.path.join(self.output_dir, f"token_distribution_{self.attack_name}_{self.timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                logger.info(f"Token distribution plot saved: {plot_path}")
            
            plt.close()
            return plot_path if save else ""
            
        except Exception as e:
            logger.error(f"Error creating token distribution visualization: {str(e)}")
            return ""

def main():
    """Example usage of the performance visualizer"""
    # Initialize visualizer
    visualizer = PerformanceVisualizer()
    
    # Example data (replace with actual system data)
    example_data = {
        'training_history': {
            'epoch_losses': [0.5, 0.3, 0.2, 0.1, 0.05],
            'epoch_accuracies': [0.6, 0.7, 0.8, 0.9, 0.95]
        },
        'round_results': [
            {'round': 1, 'accuracy': 0.8, 'avg_loss': 0.2, 'num_clients': 3},
            {'round': 2, 'accuracy': 0.85, 'avg_loss': 0.15, 'num_clients': 3},
            {'round': 3, 'accuracy': 0.9, 'avg_loss': 0.1, 'num_clients': 3}
        ],
        'evaluation_results': {
            'accuracy': 0.5,
            'precision': 0.4,
            'recall': 0.3,
            'f1_score': 0.35,
            'zero_day_detection_rate': 0.0,
            'avg_confidence': 0.0
        },
        'client_results': [
            {'client_id': 'client_1', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3},
            {'client_id': 'client_2', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3},
            {'client_id': 'client_3', 'accuracy': 0.5, 'f1_score': 0.3, 'precision': 0.4, 'recall': 0.3}
        ],
        'blockchain_data': {
            'transactions': [1, 2, 3, 4, 5],
            'ipfs_cids': ['Qm1', 'Qm2', 'Qm3', 'Qm4', 'Qm5'],
            'gas_used': [21000, 25000, 23000, 24000, 22000]
        }
    }
    
    # Generate all plots
    visualizer.plot_training_history(example_data['training_history'])
    visualizer.plot_federated_rounds(example_data['round_results'])
    # visualizer.plot_zero_day_detection_metrics(example_data['evaluation_results'])  # Removed - not properly plotting
    visualizer.plot_client_performance(example_data['client_results'])
    visualizer.plot_blockchain_metrics(example_data['blockchain_data'])
    visualizer.create_comprehensive_report(example_data)
    visualizer.save_metrics_to_json(example_data)
    
    logger.info("All performance visualizations completed!")

if __name__ == "__main__":
    main()
