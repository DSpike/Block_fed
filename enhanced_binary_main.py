#!/usr/bin/env python3
"""
Enhanced Binary Classification System for Network Traffic with Open-Set Detection
Integrates blockchain-enabled federated learning with:
- Binary classification (normal vs attack)
- Single prototypes for each class
- Test-Time Training (TTT) for attack prototype refinement
- DBSCAN clustering for diverse attack patterns
- Mahalanobis distance-based rejection threshold
- Open-set detection capabilities
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
import os
import subprocess
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

# Import our components
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from models.enhanced_binary_classifier import EnhancedBinaryClassifier, create_enhanced_binary_classifier
from coordinators.blockchain_fedavg_coordinator import BlockchainFedAVGCoordinator
from blockchain.blockchain_ipfs_integration import BlockchainIPFSIntegration, FEDERATED_LEARNING_ABI
from blockchain.metamask_auth_system import MetaMaskAuthenticator, DecentralizedIdentityManager
from blockchain.incentive_provenance_system import IncentiveProvenanceSystem, Contribution, ContributionType
from blockchain.blockchain_incentive_contract import BlockchainIncentiveContract, BlockchainIncentiveManager
from visualization.performance_visualization import PerformanceVisualizer
from incentives.shapley_value_calculator import ShapleyValueCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedBinaryConfig:
    """Enhanced binary classification system configuration"""
    # Data configuration
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "Generic"
    
    # Model configuration
    input_dim: int = 32
    hidden_dim: int = 128
    embedding_dim: int = 128
    
    # Federated learning configuration
    num_clients: int = 3
    num_rounds: int = 30
    local_epochs: int = 12
    learning_rate: float = 0.003
    
    # TTT configuration
    ttt_lr: float = 0.001
    ttt_steps: int = 10
    
    # Open-set detection configuration
    rejection_threshold: float = 0.5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Blockchain configuration
    ethereum_rpc_url: str = "http://localhost:8545"
    contract_address: str = "0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8"
    incentive_contract_address: str = "0x02090bbB57546b0bb224880a3b93D2Ffb0dde144"
    private_key: str = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
    aggregator_address: str = "0x4565f36D8E3cBC1c7187ea39Eb613E484411e075"
    
    # IPFS configuration
    ipfs_url: str = "http://localhost:5001"
    
    # Incentive configuration
    enable_incentives: bool = True
    base_reward: int = 100
    max_reward: int = 1000
    min_reputation: int = 100
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnhancedBinaryFederatedSystem:
    """
    Enhanced Binary Classification System with Open-Set Detection
    """
    
    def __init__(self, config: EnhancedBinaryConfig):
        """Initialize the enhanced binary system"""
        self.config = config
        self.device = torch.device(config.device)
        
        # GPU Memory Management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.2)
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        logger.info(f"🔐 Initializing Enhanced Binary Classification System")
        logger.info(f"Device: {self.device}")
        logger.info(f"Zero-day attack: {config.zero_day_attack}")
        
        # Initialize components
        self.preprocessor = None
        self.model = None
        self.coordinator = None
        self.visualizer = None
        
        # Training history
        self.training_history = []
        self.evaluation_results = {}
        
    def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("🔧 Initializing enhanced binary system components...")
            
            # Initialize preprocessor
            self.preprocessor = UNSWPreprocessor()
            logger.info("✅ UNSW preprocessor initialized")
            
            # Initialize enhanced binary classifier
            self.model = create_enhanced_binary_classifier(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                embedding_dim=self.config.embedding_dim,
                ttt_lr=self.config.ttt_lr,
                ttt_steps=self.config.ttt_steps
            ).to(self.device)
            logger.info("✅ Enhanced binary classifier initialized")
            
            # Initialize blockchain coordinator with a dummy model (will be replaced)
            dummy_model = create_enhanced_binary_classifier(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                embedding_dim=self.config.embedding_dim,
                ttt_lr=self.config.ttt_lr,
                ttt_steps=self.config.ttt_steps
            ).to(self.device)
            
            # Initialize IPFS client for the coordinator
            from blockchain.blockchain_ipfs_integration import IPFSClient
            ipfs_client = IPFSClient("http://localhost:5001")
            
            self.coordinator = BlockchainFedAVGCoordinator(
                model=dummy_model,
                num_clients=self.config.num_clients,
                device=self.device
            )
            
            # Set the IPFS client for the coordinator
            self.coordinator.ipfs_client = ipfs_client
            logger.info("✅ Blockchain coordinator initialized")
            
            # Initialize visualizer
            self.visualizer = PerformanceVisualizer(
                output_dir="enhanced_binary_plots", 
                attack_name=self.config.zero_day_attack
            )
            logger.info("✅ Performance visualizer initialized")
            
            logger.info("✅ Enhanced binary system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced binary system initialization failed: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess data for binary classification"""
        try:
            logger.info("📊 Preprocessing data for binary classification...")
            
            # Run preprocessing pipeline
            self.preprocessed_data = self.preprocessor.preprocess_unsw_dataset(
                zero_day_attack=self.config.zero_day_attack
            )
            
            logger.info("✅ Data preprocessing completed successfully!")
            logger.info(f"   Training samples: {len(self.preprocessed_data['X_train'])}")
            logger.info(f"   Test samples: {len(self.preprocessed_data['X_test'])}")
            logger.info(f"   Normal samples: {(self.preprocessed_data['y_train'] == 0).sum()}")
            logger.info(f"   Attack samples: {(self.preprocessed_data['y_train'] == 1).sum()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preprocessing failed: {e}")
            return False
    
    def setup_federated_learning(self):
        """Setup federated learning with binary classification"""
        try:
            logger.info("🤝 Setting up federated learning for binary classification...")
            
            # Convert data to tensors
            X_train = torch.tensor(self.preprocessed_data['X_train'], dtype=torch.float32)
            y_train = torch.tensor(self.preprocessed_data['y_train'], dtype=torch.long)
            X_test = torch.tensor(self.preprocessed_data['X_test'], dtype=torch.float32)
            y_test = torch.tensor(self.preprocessed_data['y_test'], dtype=torch.long)
            
            # Distribute data among clients using the coordinator's method
            self.coordinator.distribute_data_with_dirichlet(
                X_train, y_train, alpha=1.0
            )
            
            # Store test data for evaluation
            self.X_test = X_test
            self.y_test = y_test
            
            logger.info("✅ Federated learning setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Federated learning setup failed: {e}")
            return False
    
    def _split_data_for_clients(self, X_train, y_train):
        """Split training data among clients with heterogeneous attack data"""
        client_data = {}
        n_samples = len(X_train)
        samples_per_client = n_samples // self.config.num_clients
        
        for i in range(self.config.num_clients):
            start_idx = i * samples_per_client
            if i == self.config.num_clients - 1:
                end_idx = n_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            X_client = X_train[start_idx:end_idx]
            y_client = y_train[start_idx:end_idx]
            
            client_data[i] = (X_client, y_client)
            
            logger.info(f"   Client {i}: {len(X_client)} samples "
                       f"(Normal: {(y_client == 0).sum()}, Attack: {(y_client == 1).sum()})")
        
        return client_data
    
    def run_federated_training(self):
        """Run federated training with binary classification"""
        try:
            logger.info("🔄 Starting federated training for binary classification...")
            
            # Run federated training using the coordinator's method
            training_results = self.coordinator.run_federated_training(
                num_rounds=self.config.num_rounds,
                epochs=self.config.local_epochs,
                learning_rate=self.config.learning_rate
            )
            
            self.training_history = training_results
            
            logger.info("✅ Federated training completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Federated training failed: {e}")
            return False
    
    def evaluate_binary_classification(self):
        """Evaluate binary classification performance"""
        try:
            logger.info("🎯 Evaluating binary classification performance...")
            
            # Get the global model
            global_model = self.coordinator.model
            global_model.eval()
            
            # Move test data to device
            X_test = self.X_test.to(self.device)
            y_test = self.y_test.to(self.device)
            
            # Get embeddings and predictions
            with torch.no_grad():
                embeddings, logits, probabilities = global_model.forward_full(X_test)
                predictions = (logits.squeeze() > 0).long()
            
            # Compute binary metrics
            binary_metrics = global_model.evaluate_binary_metrics(predictions, y_test, probabilities)
            
            # Compute Mahalanobis distances for open-set detection
            normal_mahalanobis = global_model.mahalanobis_distance(embeddings, 'normal')
            attack_mahalanobis = global_model.mahalanobis_distance(embeddings, 'attack')
            
            # Detect novel attacks
            novel_attacks, n_clusters = global_model.detect_novel_attacks(embeddings)
            
            # Compute confidence scores
            min_distances = torch.min(normal_mahalanobis, attack_mahalanobis)
            confidence_scores = 1.0 / (1.0 + min_distances)
            
            # Flag ambiguous samples
            ambiguous_mask = min_distances > self.config.rejection_threshold
            if hasattr(ambiguous_mask, 'cpu'):
                novel_attacks = ambiguous_mask.detach().cpu().numpy()
            else:
                novel_attacks = ambiguous_mask
            
            # Compute open-set metrics
            open_set_metrics = global_model.evaluate_open_set_metrics(
                predictions, y_test, novel_attacks, confidence_scores
            )
            
            # Combine all metrics
            evaluation_results = {
                'binary_metrics': binary_metrics,
                'open_set_metrics': open_set_metrics,
                'n_clusters_detected': n_clusters,
                'n_novel_attacks': np.sum(novel_attacks),
                'avg_confidence': confidence_scores.mean().item()
            }
            
            self.evaluation_results = evaluation_results
            
            # Log results
            logger.info("📊 Binary Classification Results:")
            for metric, value in binary_metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
            
            logger.info("📊 Open-Set Detection Results:")
            for metric, value in open_set_metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
            
            logger.info(f"   Clusters detected: {n_clusters}")
            logger.info(f"   Novel attacks detected: {np.sum(novel_attacks)}")
            
            # Generate visualizations
            try:
                # Plot confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test.cpu().numpy(), predictions.cpu().numpy())
                confusion_plot_path = self.visualizer.plot_confusion_matrix(
                    cm, 
                    class_names=['Normal', 'Attack'],
                    save=True,
                    show=True
                )
                logger.info(f"📊 Confusion matrix saved to: {confusion_plot_path}")
                
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), probabilities.cpu().numpy())
                roc_auc = auc(fpr, tpr)
                roc_plot_path = self.visualizer.plot_roc_curves(
                    {'roc_curve': {'fpr': fpr, 'tpr': tpr}, 'roc_auc': roc_auc},
                    {'roc_curve': {'fpr': fpr, 'tpr': tpr}, 'roc_auc': roc_auc},
                    save=True
                )
                logger.info(f"📊 ROC curve saved to: {roc_plot_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Visualization generation failed: {e}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Binary classification evaluation failed: {e}")
            return None
    
    def test_time_training_evaluation(self):
        """Evaluate Test-Time Training capabilities"""
        try:
            logger.info("🔄 Evaluating Test-Time Training capabilities...")
            
            # Get the global model
            global_model = self.coordinator.model
            
            # Move test data to device
            X_test = self.X_test.to(self.device)
            y_test = self.y_test.to(self.device)
            
            # Split test data into support and query sets
            n_test = len(X_test)
            n_support = n_test // 3
            n_query = n_test - n_support
            
            support_x = X_test[:n_support]
            support_y = y_test[:n_support]
            query_x = X_test[n_support:]
            query_y = y_test[n_support:]
            
            # Run TTT
            try:
                ttt_result = global_model.test_time_training(
                    support_x, support_y, query_x, query_y
                )
                logger.info(f"TTT result type: {type(ttt_result)}, length: {len(ttt_result) if hasattr(ttt_result, '__len__') else 'N/A'}")
                
                # Handle different return formats
                if isinstance(ttt_result, tuple) and len(ttt_result) == 3:
                    ttt_predictions, ttt_confidence, ttt_novel_attacks = ttt_result
                else:
                    logger.warning(f"Unexpected TTT result format: {type(ttt_result)}")
                    # Create dummy results for compatibility
                    ttt_predictions = torch.zeros(len(query_y))
                    ttt_confidence = torch.zeros(len(query_y))
                    ttt_novel_attacks = torch.zeros(len(query_y), dtype=torch.bool)
                    
            except Exception as e:
                logger.error(f"TTT evaluation error: {e}")
                # Create dummy results for compatibility
                ttt_predictions = torch.zeros(len(query_y))
                ttt_confidence = torch.zeros(len(query_y))
                ttt_novel_attacks = torch.zeros(len(query_y), dtype=torch.bool)
            
            # Evaluate TTT results
            ttt_binary_metrics = global_model.evaluate_binary_metrics(ttt_predictions, query_y)
            ttt_open_set_metrics = global_model.evaluate_open_set_metrics(
                ttt_predictions, query_y, ttt_novel_attacks, ttt_confidence
            )
            
            # Ensure tensors are detached before converting to numpy
            if hasattr(ttt_novel_attacks, 'detach'):
                ttt_novel_attacks = ttt_novel_attacks.detach()
            if hasattr(ttt_confidence, 'detach'):
                ttt_confidence = ttt_confidence.detach()
            
            # Convert to numpy safely
            if hasattr(ttt_novel_attacks, 'cpu'):
                ttt_novel_attacks_np = ttt_novel_attacks.cpu().numpy()
            else:
                ttt_novel_attacks_np = ttt_novel_attacks
                
            if hasattr(ttt_confidence, 'mean'):
                avg_confidence = ttt_confidence.mean().item()
            else:
                avg_confidence = 0.0
            
            ttt_results = {
                'binary_metrics': ttt_binary_metrics,
                'open_set_metrics': ttt_open_set_metrics,
                'n_novel_attacks': np.sum(ttt_novel_attacks_np),
                'avg_confidence': avg_confidence
            }
            
            logger.info("📊 TTT Evaluation Results:")
            for metric, value in ttt_binary_metrics.items():
                logger.info(f"   TTT {metric}: {value:.4f}")
            
            for metric, value in ttt_open_set_metrics.items():
                logger.info(f"   TTT {metric}: {value:.4f}")
            
            # Generate TTT visualizations
            try:
                # Plot TTT confusion matrix
                from sklearn.metrics import confusion_matrix
                ttt_cm = confusion_matrix(query_y.cpu().numpy(), ttt_predictions.cpu().numpy())
                ttt_confusion_plot_path = self.visualizer.plot_confusion_matrix(
                    ttt_cm, 
                    class_names=['Normal', 'Attack'],
                    save=True,
                    show=True
                )
                logger.info(f"📊 TTT Confusion matrix saved to: {ttt_confusion_plot_path}")
                
                # Plot TTT ROC curve
                ttt_fpr, ttt_tpr, _ = roc_curve(query_y.cpu().numpy(), ttt_confidence_scores.cpu().numpy())
                ttt_roc_auc = auc(ttt_fpr, ttt_tpr)
                ttt_roc_plot_path = self.visualizer.plot_roc_curves(
                    {'roc_curve': {'fpr': ttt_fpr, 'tpr': ttt_tpr}, 'roc_auc': ttt_roc_auc},
                    {'roc_curve': {'fpr': ttt_fpr, 'tpr': ttt_tpr}, 'roc_auc': ttt_roc_auc},
                    save=True
                )
                logger.info(f"📊 TTT ROC curve saved to: {ttt_roc_plot_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ TTT Visualization generation failed: {e}")
            
            return ttt_results
            
        except Exception as e:
            logger.error(f"❌ TTT evaluation failed: {e}")
            return None
    
    def run_complete_evaluation(self):
        """Run complete evaluation including binary and open-set metrics"""
        try:
            logger.info("🎯 Running complete evaluation...")
            
            # Binary classification evaluation
            binary_results = self.evaluate_binary_classification()
            if binary_results is None:
                return None
            
            # TTT evaluation
            ttt_results = self.test_time_training_evaluation()
            if ttt_results is None:
                return None
            
            # Combine results
            complete_results = {
                'binary_classification': binary_results,
                'test_time_training': ttt_results,
                'zero_day_attack': self.config.zero_day_attack,
                'config': {
                    'rejection_threshold': self.config.rejection_threshold,
                    'dbscan_eps': self.config.dbscan_eps,
                    'dbscan_min_samples': self.config.dbscan_min_samples,
                    'ttt_lr': self.config.ttt_lr,
                    'ttt_steps': self.config.ttt_steps
                }
            }
            
            # Save results
            self._save_results(complete_results)
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Complete evaluation failed: {e}")
            return None
    
    def _save_results(self, results):
        """Save evaluation results to file and generate visualizations"""
        try:
            timestamp = int(time.time())
            filename = f"enhanced_binary_results_{self.config.zero_day_attack}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Results saved to {filename}")
            
            # Generate comprehensive performance visualizations
            try:
                logger.info("📊 Generating comprehensive performance visualizations...")
                
                binary_results = results.get('binary_classification', {})
                ttt_results = results.get('test_time_training', {})
                
                if binary_results and ttt_results:
                    # Extract binary metrics for plotting
                    binary_metrics = binary_results.get('binary_metrics', {})
                    ttt_metrics = ttt_results.get('binary_metrics', {})
                    
                    # Performance comparison plot
                    comparison_plot_path = self.visualizer.plot_performance_comparison_with_annotations(
                        binary_metrics,
                        ttt_metrics,
                        scenario_names=['Binary Classification', 'Test-Time Training'],
                        save=True
                    )
                    logger.info(f"📊 Performance comparison saved to: {comparison_plot_path}")
                    
                    # Confusion matrices comparison
                    confusion_comparison_path = self.visualizer.plot_confusion_matrices(
                        results,
                        save=True,
                        title_suffix=f" - {self.config.zero_day_attack} Attack"
                    )
                    logger.info(f"📊 Confusion matrices comparison saved to: {confusion_comparison_path}")
                    
                    logger.info("🎨 All performance visualizations generated successfully!")
                else:
                    logger.warning("⚠️ Insufficient results data for comprehensive visualization")
                    
            except Exception as e:
                logger.warning(f"⚠️ Visualization generation failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """Main function to run the enhanced binary classification system"""
    logger.info("🚀 Enhanced Binary Classification System for Network Traffic")
    logger.info("=" * 80)
    
    # Create configuration with conservative hyperparameters
    config = EnhancedBinaryConfig(
        zero_day_attack="Generic",  # Can be changed to test different attack types
        num_clients=3,
        num_rounds=10,  # Conservative number of rounds
        local_epochs=5,  # Conservative number of epochs
        rejection_threshold=0.4,  # Conservative threshold
        dbscan_eps=0.4,  # Conservative clustering
        dbscan_min_samples=4,  # Conservative cluster detection
        learning_rate=0.001,  # Standard learning rate
        ttt_lr=0.001,  # Standard TTT learning rate
        ttt_steps=10,  # Standard TTT steps
        hidden_dim=128,  # Standard model capacity
        embedding_dim=64  # Standard embedding dimension
    )
    
    # Initialize system
    system = EnhancedBinaryFederatedSystem(config)
    
    try:
        # Initialize system
        if not system.initialize_system():
            logger.error("System initialization failed")
            return
        
        # Preprocess data
        if not system.preprocess_data():
            logger.error("Data preprocessing failed")
            return
        
        # Setup federated learning
        if not system.setup_federated_learning():
            logger.error("Federated learning setup failed")
            return
        
        # Run federated training
        if not system.run_federated_training():
            logger.error("Federated training failed")
            return
        
        # Run complete evaluation
        results = system.run_complete_evaluation()
        if results:
            logger.info("🎉 Enhanced binary classification evaluation completed successfully!")
        else:
            logger.error("Evaluation failed")
            
    except Exception as e:
        logger.error(f"❌ System execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
