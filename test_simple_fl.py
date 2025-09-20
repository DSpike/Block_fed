"""
Test script for simplified federated learning without blockchain complexity
"""
import torch
import logging
import sys
import os

# Add src to path
sys.path.append('src')

from src.coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator
from src.models.transductive_fewshot_model import TransductiveFewShotModel
from src.preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test simplified federated learning"""
    logger.info("🚀 Testing Simplified Federated Learning (No Blockchain)")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # GPU Memory Management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.3)
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransductiveFewShotModel(
        input_dim=25,
        hidden_dim=64,
        num_classes=2
    )
    
    # Initialize coordinator
    logger.info("Initializing coordinator...")
    coordinator = SimpleFedAVGCoordinator(model, num_clients=3, device=device)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    preprocessor = UNSWPreprocessor()
    
    # Load UNSW-NB15 dataset
    processed_data = preprocessor.preprocess_unsw_dataset()
    logger.info(f"Available keys: {list(processed_data.keys())}")
    
    train_data = processed_data['X_train']
    train_labels = processed_data['y_train']
    test_data = processed_data['X_test']
    test_labels = processed_data['y_test']
    
    # Convert to tensors (data is already numpy arrays)
    train_data = torch.FloatTensor(train_data)
    train_labels = torch.LongTensor(train_labels)
    test_data = torch.FloatTensor(test_data)
    test_labels = torch.LongTensor(test_labels)
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Training labels shape: {train_labels.shape}")
    
    # Distribute data
    coordinator.distribute_data(train_data, train_labels)
    
    # Run federated learning
    logger.info("Starting federated learning...")
    num_rounds = 3
    
    for round_num in range(num_rounds):
        logger.info(f"\n🔄 ROUND {round_num + 1}/{num_rounds}")
        logger.info("-" * 40)
        
        try:
            result = coordinator.run_federated_round(epochs=2)
            logger.info(f"✅ Round {round_num + 1} completed successfully")
            
            # Evaluate on test data
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_data.to(device))
                predictions = torch.argmax(test_outputs, dim=1)
                accuracy = (predictions == test_labels.to(device)).float().mean().item()
                logger.info(f"Test accuracy after round {round_num + 1}: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Round {round_num + 1} failed: {str(e)}")
            break
    
    logger.info("\n🎉 Simplified federated learning test completed!")
    logger.info("If this worked without memory errors, the issue is in the blockchain complexity.")

if __name__ == "__main__":
    main()
