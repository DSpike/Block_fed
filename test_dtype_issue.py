#!/usr/bin/env python3
"""
Test script to isolate the dtype issue in TCGAN model
"""

import torch
import torch.nn as nn
from models.tcgan_ttt_model import TCGANTTTModel

def test_dtype_issue():
    """Test the dtype issue with TCGAN model"""
    print("Testing TCGAN model dtype compatibility...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a small test model
    model = TCGANTTTModel(
        input_dim=22,
        sequence_length=10,
        latent_dim=64,  # Reduced for testing
        hidden_dim=128,  # Reduced for testing
        num_classes=2,
        noise_dim=64
    ).to(device)
    
    # Create test data
    batch_size = 4
    test_data = torch.randn(batch_size, 22).to(device)
    test_labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    print(f"Input dtype: {test_data.dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    
    try:
        # Test forward pass
        print("Testing forward pass...")
        outputs = model(test_data)
        print(f"Output dtype: {outputs.dtype}")
        print(f"Output shape: {outputs.shape}")
        print("✅ Forward pass successful!")
        
        # Test training
        print("Testing training...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        outputs = model(test_data)
        loss = criterion(outputs, test_labels)
        loss.backward()
        optimizer.step()
        
        print(f"Training loss: {loss.item()}")
        print("✅ Training successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        
        # Check if it's a dtype issue
        if "dtype" in str(e) or "Half" in str(e) or "Float" in str(e):
            print("🔍 This is a dtype compatibility issue")
            
            # Try to identify the problematic layer
            print("Checking model components...")
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    print(f"  {name}: {module.weight.dtype}")
        
        return False
    
    return True

if __name__ == "__main__":
    test_dtype_issue()
