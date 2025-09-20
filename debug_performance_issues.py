#!/usr/bin/env python3
"""
Debug script to investigate performance issues in the blockchain federated learning system
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import torch
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from models.transductive_fewshot_model import TransductiveFewShotModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_distribution():
    """Debug the data distribution and zero-day split"""
    logger.info("🔍 DEBUGGING DATA DISTRIBUTION AND ZERO-DAY SPLIT")
    logger.info("=" * 60)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data
    logger.info("Loading UNSW-NB15 dataset...")
    train_df = pd.read_csv("UNSW_NB15_training-set.csv")
    test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
    
    logger.info(f"Original training data shape: {train_df.shape}")
    logger.info(f"Original test data shape: {test_df.shape}")
    
    # Check original attack categories
    logger.info("\n📊 ORIGINAL ATTACK CATEGORIES:")
    logger.info("Training data attack_cat distribution:")
    train_attack_dist = train_df['attack_cat'].value_counts()
    for attack, count in train_attack_dist.items():
        logger.info(f"  {attack}: {count} samples")
    
    logger.info("\nTest data attack_cat distribution:")
    test_attack_dist = test_df['attack_cat'].value_counts()
    for attack, count in test_attack_dist.items():
        logger.info(f"  {attack}: {count} samples")
    
    # Check if 'Exploits' exists in the data
    logger.info(f"\n🔍 CHECKING FOR 'EXPLOITS' ATTACK TYPE:")
    logger.info(f"Exploits in training data: {'Exploits' in train_df['attack_cat'].values}")
    logger.info(f"Exploits in test data: {'Exploits' in test_df['attack_cat'].values}")
    
    if 'Exploits' in train_df['attack_cat'].values:
        train_exploits = train_df[train_df['attack_cat'] == 'Exploits']
        logger.info(f"Training Exploits samples: {len(train_exploits)}")
    
    if 'Exploits' in test_df['attack_cat'].values:
        test_exploits = test_df[test_df['attack_cat'] == 'Exploits']
        logger.info(f"Test Exploits samples: {len(test_exploits)}")
    
    # Run preprocessing
    logger.info("\n🔄 RUNNING PREPROCESSING...")
    try:
        data = preprocessor.preprocess_unsw_dataset(zero_day_attack='Exploits')
        
        logger.info("\n📈 PREPROCESSING RESULTS:")
        logger.info(f"Training data shape: {data['X_train'].shape}")
        logger.info(f"Validation data shape: {data['X_val'].shape}")
        logger.info(f"Test data shape: {data['X_test'].shape}")
        logger.info(f"Feature count: {len(data['feature_names'])}")
        logger.info(f"Zero-day attack: {data['zero_day_attack']}")
        
        # Check binary label distribution
        logger.info("\n🎯 BINARY LABEL DISTRIBUTION:")
        train_binary_dist = pd.Series(data['y_train']).value_counts().sort_index()
        val_binary_dist = pd.Series(data['y_val']).value_counts().sort_index()
        test_binary_dist = pd.Series(data['y_test']).value_counts().sort_index()
        
        logger.info("Training binary labels:")
        for label, count in train_binary_dist.items():
            logger.info(f"  Label {label}: {count} samples ({count/len(data['y_train'])*100:.1f}%)")
        
        logger.info("Validation binary labels:")
        for label, count in val_binary_dist.items():
            logger.info(f"  Label {label}: {count} samples ({count/len(data['y_val'])*100:.1f}%)")
        
        logger.info("Test binary labels:")
        for label, count in test_binary_dist.items():
            logger.info(f"  Label {label}: {count} samples ({count/len(data['y_test'])*100:.1f}%)")
        
        # Check if the zero-day split is working correctly
        logger.info("\n🚨 ZERO-DAY SPLIT ANALYSIS:")
        if 'zero_day_attack' in data:
            logger.info(f"Zero-day attack specified: {data['zero_day_attack']}")
            
            # Check if test data has the expected 50/50 split
            test_normal = torch.sum(data['y_test'] == 0).item()
            test_attack = torch.sum(data['y_test'] == 1).item()
            total_test = len(data['y_test'])
            
            logger.info(f"Test data - Normal: {test_normal} ({test_normal/total_test*100:.1f}%)")
            logger.info(f"Test data - Attack: {test_attack} ({test_attack/total_test*100:.1f}%)")
            
            if abs(test_normal - test_attack) > total_test * 0.1:  # More than 10% difference
                logger.warning("⚠️  Test data is NOT balanced! This could cause performance issues.")
            else:
                logger.info("✅ Test data is properly balanced.")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_model_performance(data):
    """Debug model performance issues"""
    if data is None:
        logger.error("No data available for model debugging")
        return
    
    logger.info("\n🤖 DEBUGGING MODEL PERFORMANCE")
    logger.info("=" * 60)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = TransductiveFewShotModel(
        feature_dim=len(data['feature_names']),
        hidden_dim=256,
        num_classes=2,  # Binary classification
        device=device
    )
    
    # Convert data to tensors
    X_train = torch.FloatTensor(data['X_train']).to(device)
    y_train = torch.LongTensor(data['y_train']).to(device)
    X_test = torch.FloatTensor(data['X_test']).to(device)
    y_test = torch.LongTensor(data['y_test']).to(device)
    
    logger.info(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    logger.info(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
    
    # Check data quality
    logger.info("\n📊 DATA QUALITY CHECK:")
    logger.info(f"Training data - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}, Mean: {X_train.mean():.4f}")
    logger.info(f"Test data - Min: {X_test.min():.4f}, Max: {X_test.max():.4f}, Mean: {X_test.mean():.4f}")
    
    # Check for NaN or infinite values
    train_nan = torch.isnan(X_train).sum().item()
    train_inf = torch.isinf(X_train).sum().item()
    test_nan = torch.isnan(X_test).sum().item()
    test_inf = torch.isinf(X_test).sum().item()
    
    logger.info(f"Training data - NaN: {train_nan}, Inf: {train_inf}")
    logger.info(f"Test data - NaN: {test_nan}, Inf: {test_inf}")
    
    if train_nan > 0 or train_inf > 0 or test_nan > 0 or test_inf > 0:
        logger.warning("⚠️  Data contains NaN or infinite values!")
    
    # Test basic model forward pass
    logger.info("\n🧪 TESTING MODEL FORWARD PASS:")
    try:
        model.eval()
        with torch.no_grad():
            # Test on a small batch
            test_batch = X_test[:100]
            output = model(test_batch)
            logger.info(f"Model output shape: {output.shape}")
            logger.info(f"Model output range: {output.min():.4f} to {output.max():.4f}")
            
            # Check predictions
            predictions = torch.argmax(output, dim=1)
            pred_dist = torch.bincount(predictions)
            logger.info(f"Prediction distribution: {pred_dist}")
            
            # Check if model is predicting only one class
            if len(pred_dist) == 1:
                logger.warning("⚠️  Model is predicting only one class!")
            elif pred_dist[0] == 0 or pred_dist[1] == 0:
                logger.warning("⚠️  Model is heavily biased towards one class!")
            
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()

def debug_visualization_data():
    """Debug visualization data issues"""
    logger.info("\n📊 DEBUGGING VISUALIZATION DATA")
    logger.info("=" * 60)
    
    # Check if performance metrics file exists
    metrics_file = "performance_plots/performance_metrics_20250918_190705.json"
    if os.path.exists(metrics_file):
        logger.info(f"Found metrics file: {metrics_file}")
        
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        logger.info("📈 METRICS SUMMARY:")
        logger.info(f"Overall accuracy: {metrics.get('accuracy', 'N/A')}")
        logger.info(f"F1-score: {metrics.get('f1_score', 'N/A')}")
        logger.info(f"Zero-day detection rate: {metrics.get('zero_day_detection_rate', 'N/A')}")
        
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            logger.info(f"Confusion Matrix - TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}")
            
            # Calculate additional metrics
            total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
            precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
            recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
            specificity = cm['tn'] / (cm['tn'] + cm['fp']) if (cm['tn'] + cm['fp']) > 0 else 0
            
            logger.info(f"Calculated Precision: {precision:.4f}")
            logger.info(f"Calculated Recall: {recall:.4f}")
            logger.info(f"Calculated Specificity: {specificity:.4f}")
            
            # Analyze the confusion matrix
            logger.info("\n🔍 CONFUSION MATRIX ANALYSIS:")
            if cm['tp'] < cm['fn'] * 0.1:  # TP is less than 10% of FN
                logger.warning("⚠️  SEVERE CLASS IMBALANCE: Model is missing most attacks!")
                logger.warning(f"   Only {cm['tp']} attacks detected out of {cm['tp'] + cm['fn']} total attacks")
            
            if cm['fp'] > cm['tn'] * 0.3:  # FP is more than 30% of TN
                logger.warning("⚠️  HIGH FALSE POSITIVE RATE: Model is misclassifying many normal samples as attacks")
            
            if cm['tp'] + cm['fn'] == 0:
                logger.error("❌ NO ATTACK SAMPLES IN TEST DATA!")
            elif cm['tp'] == 0:
                logger.error("❌ MODEL FAILED TO DETECT ANY ATTACKS!")
        
        # Check client results
        if 'client_results' in metrics:
            client_results = metrics['client_results']
            logger.info(f"\n👥 CLIENT RESULTS ({len(client_results)} clients):")
            for i, client in enumerate(client_results):
                logger.info(f"  Client {i+1}: Accuracy={client.get('accuracy', 'N/A'):.4f}, "
                          f"F1={client.get('f1_score', 'N/A'):.4f}")
    else:
        logger.warning(f"Metrics file not found: {metrics_file}")

def main():
    """Main debugging function"""
    logger.info("🚀 STARTING PERFORMANCE ISSUE DEBUGGING")
    logger.info("=" * 80)
    
    # Debug data distribution
    data = debug_data_distribution()
    
    # Debug model performance
    debug_model_performance(data)
    
    # Debug visualization data
    debug_visualization_data()
    
    logger.info("\n🎯 DEBUGGING SUMMARY:")
    logger.info("=" * 60)
    logger.info("1. Check if 'Exploits' attack type exists in the dataset")
    logger.info("2. Verify zero-day split is working correctly")
    logger.info("3. Check if test data is properly balanced")
    logger.info("4. Analyze confusion matrix for class imbalance")
    logger.info("5. Check if model is predicting only one class")
    logger.info("6. Verify data quality (no NaN/inf values)")
    
    logger.info("\n✅ Debugging completed!")

if __name__ == "__main__":
    main()
