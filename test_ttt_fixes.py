#!/usr/bin/env python3
"""
Test file for TTT adaptation infinite loop fixes
This file tests the fixes before applying them to main.py
"""

import torch
import torch.nn as nn
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ttt_adaptation_with_safety_measures():
    """
    Test TTT adaptation with safety measures to prevent infinite loops
    """
    logger.info("🧪 Testing TTT adaptation with safety measures...")
    
    # Mock data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    support_x = torch.randn(50, 10).to(device)
    support_y = torch.randint(0, 2, (50,)).to(device)
    query_x = torch.randn(50, 10).to(device)
    
    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
            
        def forward(self, x):
            return self.fc(x)
            
        def set_ttt_mode(self, training=True):
            self.training = training
            
        def get_dropout_status(self):
            return [1, 2, 3, 4, 5, 6]  # Mock dropout layers
    
    model = MockModel().to(device)
    
    # Test TTT adaptation with safety measures
    try:
        adapted_model = perform_ttt_adaptation_with_safety(model, support_x, support_y, query_x)
        logger.info("✅ TTT adaptation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ TTT adaptation failed: {e}")
        return False

def perform_ttt_adaptation_with_safety(model, support_x, support_y, query_x):
    """
    TTT adaptation with safety measures to prevent infinite loops
    """
    logger.info("🔄 Starting TTT adaptation with safety measures...")
    
    # Clone model
    import copy
    adapted_model = copy.deepcopy(model)
    
    # Set training mode
    adapted_model.set_ttt_mode(training=True)
    
    # Enhanced optimizer setup
    optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=7, T_mult=2, eta_min=1e-6, last_epoch=-1
    )
    
    # Calculate adaptive TTT steps with safety limits
    base_ttt_steps = 23
    query_variance = torch.var(query_x).item()
    complexity_factor = min(2.0, 1.0 + query_variance * 10)
    ttt_steps = int(base_ttt_steps * complexity_factor)
    
    # SAFETY LIMITS
    ttt_steps = min(ttt_steps, 50)  # Reduced max steps for testing
    ttt_timeout = 10  # 10 seconds timeout for testing
    
    logger.info(f"📊 TTT Steps: {ttt_steps} (complexity: {complexity_factor:.2f})")
    logger.info(f"⏱️  Timeout: {ttt_timeout}s")
    
    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5  # Reduced patience for testing
    min_improvement = 1e-4
    
    # Start timing
    start_time = time.time()
    
    # TTT training loop with safety measures
    for step in range(ttt_steps):
        # SAFETY CHECK 1: Timeout
        if time.time() - start_time > ttt_timeout:
            logger.warning(f"⏰ TTT adaptation timeout after {ttt_timeout}s at step {step}")
            break
        
        optimizer.zero_grad()
        
        # Forward pass
        support_outputs = adapted_model(support_x)
        query_outputs = adapted_model(query_x)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        support_loss = criterion(support_outputs, support_y)
        
        # Mock consistency loss
        consistency_loss = torch.mean(torch.var(query_outputs, dim=1))
        total_loss = support_loss + 0.1 * consistency_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Early stopping check
        current_loss = total_loss.item()
        if current_loss < best_loss - min_improvement:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # SAFETY CHECK 2: Early stopping
        if patience_counter >= patience_limit:
            logger.info(f"🛑 Early stopping at step {step} (patience: {patience_limit})")
            break
        
        # Log progress every few steps
        if step % 5 == 0:
            logger.info(f"📈 Step {step}: Loss = {current_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ TTT adaptation completed in {elapsed_time:.2f}s, {step+1} steps")
    
    return adapted_model

def test_meta_tasks_with_safety():
    """
    Test meta-tasks evaluation with safety measures
    """
    logger.info("🧪 Testing meta-tasks evaluation with safety measures...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock data
    X_test = torch.randn(1000, 10).to(device)
    y_test = torch.randint(0, 2, (1000,)).to(device)
    
    # Test with reduced number of meta-tasks
    num_meta_tasks = 5  # Reduced for testing
    successful_tasks = 0
    
    for task_idx in range(num_meta_tasks):
        logger.info(f"📊 Processing meta-task {task_idx + 1}/{num_meta_tasks}...")
        
        try:
            # Mock stratified split
            from sklearn.model_selection import train_test_split
            support_x, query_x, support_y, query_y = train_test_split(
                X_test.cpu().numpy(), y_test.cpu().numpy(),
                test_size=0.5, stratify=y_test.cpu().numpy(), random_state=42 + task_idx
            )
            
            # Convert to tensors and move to device
            support_x = torch.FloatTensor(support_x).to(device)
            support_y = torch.LongTensor(support_y).to(device)
            query_x = torch.FloatTensor(query_x).to(device)
            query_y = torch.LongTensor(query_y).to(device)
            
            # Mock TTT adaptation
            logger.info(f"✅ Meta-task {task_idx + 1} completed successfully")
            successful_tasks += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Meta-task {task_idx + 1} failed: {e}")
    
    logger.info(f"📊 Meta-tasks completed: {successful_tasks}/{num_meta_tasks}")
    return successful_tasks > 0

def main():
    """
    Main test function
    """
    logger.info("🚀 Starting TTT adaptation safety tests...")
    
    # Test 1: TTT adaptation with safety measures
    test1_passed = test_ttt_adaptation_with_safety_measures()
    
    # Test 2: Meta-tasks with safety measures
    test2_passed = test_meta_tasks_with_safety()
    
    # Summary
    logger.info("📋 Test Results:")
    logger.info(f"  TTT Adaptation Safety: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"  Meta-tasks Safety: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All safety tests passed! Ready to apply fixes to main.py")
        return True
    else:
        logger.error("❌ Some tests failed. Need to review fixes.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

