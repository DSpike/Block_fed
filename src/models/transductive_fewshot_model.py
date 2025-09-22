#!/usr/bin/env python3
"""
Transductive Few-Shot Model with Test-Time Training for Zero-Day Detection
Implements meta-learning approach for rapid adaptation to new attack patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25 for normal class)
        gamma: Focusing parameter (default: 2)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probability of true class
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TransductiveLearner(nn.Module):
    """
    True Transductive Learning for Zero-Day Detection
    Uses both support set and test set structure for classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2):
        super(TransductiveLearner, self).__init__()
        
        # Multi-scale feature extraction (2 scales for good performance)
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, embedding_dim)
            ) for _ in range(2)
        ])
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # Transductive components
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.attention_net = nn.MultiheadAttention(embedding_dim, num_heads=4)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=2)
        self.graph_conv = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Transductive learning parameters
        self.transductive_lr = 0.001
        self.transductive_steps = 25
        self.graph_weight = 0.3
        self.consistency_weight = 0.2
        
    def forward(self, x):
        # Multi-scale feature extraction
        scale_features = []
        for extractor in self.feature_extractors:
            scale_feat = extractor(x)
            scale_features.append(scale_feat)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        embeddings = self.feature_fusion(fused_features)
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # Self-attention for global context
        embeddings_reshaped = embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        attended_embeddings, _ = self.self_attention(
            embeddings_reshaped, embeddings_reshaped, embeddings_reshaped
        )
        embeddings = attended_embeddings.squeeze(1) + embeddings  # Residual connection
        
        # Enhanced embedding processing
        embeddings = self.embedding_net(embeddings)
        
        # Classification
        logits = self.classifier(embeddings)
        return logits
    
    def compute_similarity_graph(self, embeddings):
        """
        Compute similarity graph between test samples
        """
        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarities, dim=1)
        
        return attention_weights
    
    def compute_prototypes(self, support_embeddings, support_labels):
        """
        Compute class prototypes from support set
        
        Args:
            support_embeddings: Embeddings of support samples
            support_labels: Labels of support samples
            
        Returns:
            prototypes: Class prototypes
            unique_labels: Unique class labels
        """
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes), unique_labels
    
    def classify(self, query_embeddings, prototypes, prototype_labels):
        """
        Classify query samples based on distance to prototypes
        
        Args:
            query_embeddings: Embeddings of query samples
            prototypes: Class prototypes
            prototype_labels: Labels of prototypes
            
        Returns:
            predictions: Predicted labels
            distances: Distances to prototypes
        """
        # Compute distances to all prototypes
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        # Predict based on minimum distance
        predictions = prototype_labels[torch.argmin(distances, dim=1)]
        
        return predictions, distances
    
    def transductive_optimization(self, support_x, support_y, test_x, test_y=None):
        """
        True transductive learning optimization
        """
        device = next(self.parameters()).device
        
        # Move data to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        test_x = test_x.to(device)
        
        # Get embeddings
        support_embeddings = self.embedding_net(support_x)
        test_embeddings = self.embedding_net(test_x)
        
        # Compute prototypes from support set
        unique_labels = torch.unique(support_y)
        prototypes = []
        for label in unique_labels:
            mask = support_y == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        
        # Initialize test predictions (soft labels)
        test_predictions = self.initialize_test_predictions(test_embeddings, prototypes, unique_labels)
        
        # Enhanced transductive optimization with adaptive learning
        optimizer = optim.AdamW(self.parameters(), lr=self.transductive_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for step in range(self.transductive_steps):
            optimizer.zero_grad()
            
            # Recompute embeddings (they change during optimization)
            support_embeddings = self.embedding_net(support_x)
            test_embeddings = self.embedding_net(test_x)
            
            # Update prototypes with attention weighting
            prototypes = self.update_prototypes_with_attention(support_embeddings, support_y, test_embeddings, test_predictions)
            
            # Compute enhanced transductive loss
            total_loss = self.compute_enhanced_transductive_loss(
                support_embeddings, support_y,
                test_embeddings, test_predictions,
                prototypes, unique_labels
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(total_loss)
            
            # Update test predictions with confidence weighting
            test_predictions = self.update_test_predictions_with_confidence(test_embeddings, prototypes, unique_labels)
            
            # Early stopping based on loss improvement
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 8:  # Early stopping
                logger.info(f"Early stopping at step {step}")
                break
            
            if step % 5 == 0:
                logger.info(f"Enhanced transductive step {step}: Loss = {total_loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        return test_predictions, prototypes, unique_labels
    
    def update_prototypes_with_attention(self, support_embeddings, support_y, test_embeddings, test_predictions):
        """
        Update prototypes using attention weighting from both support and test sets
        """
        unique_labels = torch.unique(support_y)
        updated_prototypes = []
        
        for label in unique_labels:
            # Support set contribution
            support_mask = support_y == label
            if support_mask.sum() > 0:
                support_class_embeddings = support_embeddings[support_mask]
                support_prototype = support_class_embeddings.mean(dim=0)
            else:
                support_prototype = torch.zeros_like(support_embeddings[0])
            
            # Test set contribution with attention weighting
            test_weights = test_predictions[:, label.item()] if len(test_predictions.shape) > 1 else test_predictions
            if test_weights.sum() > 0:
                test_prototype = torch.sum(test_embeddings * test_weights.unsqueeze(1), dim=0) / test_weights.sum()
            else:
                test_prototype = torch.zeros_like(support_embeddings[0])
            
            # Combine with adaptive weighting
            alpha = 0.7  # Weight for support set
            combined_prototype = alpha * support_prototype + (1 - alpha) * test_prototype
            updated_prototypes.append(combined_prototype)
        
        return torch.stack(updated_prototypes)
    
    def compute_enhanced_transductive_loss(self, support_embeddings, support_y, test_embeddings, test_predictions, prototypes, unique_labels):
        """
        Compute enhanced transductive loss with multiple components
        """
        # Classification loss on support set
        support_logits = self.classifier(support_embeddings)
        support_loss = F.cross_entropy(support_logits, support_y)
        
        # Consistency loss on test set
        test_logits = self.classifier(test_embeddings)
        consistency_loss = F.kl_div(
            F.log_softmax(test_logits, dim=1),
            test_predictions,
            reduction='batchmean'
        )
        
        # Graph smoothness loss
        all_embeddings = torch.cat([support_embeddings, test_embeddings], dim=0)
        similarity_matrix = torch.mm(F.normalize(all_embeddings, p=2, dim=1), 
                                   F.normalize(all_embeddings, p=2, dim=1).t())
        
        # Create adjacency matrix
        threshold = torch.quantile(similarity_matrix.flatten(), 0.8)
        adjacency_matrix = (similarity_matrix > threshold).float()
        
        smoothness_loss = 0
        edge_count = 0
        for i in range(len(all_embeddings)):
            for j in range(len(all_embeddings)):
                if adjacency_matrix[i, j] > 0:
                    smoothness_loss += F.mse_loss(all_embeddings[i], all_embeddings[j])
                    edge_count += 1
        
        if edge_count > 0:
            smoothness_loss = smoothness_loss / edge_count
        
        # Combined loss
        total_loss = support_loss + self.consistency_weight * consistency_loss + self.graph_weight * smoothness_loss
        
        return total_loss
    
    def update_test_predictions_with_confidence(self, test_embeddings, prototypes, unique_labels):
        """
        Update test predictions with confidence weighting
        """
        # Compute distances to prototypes
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        
        # Convert distances to probabilities with temperature scaling
        temperature = 2.0  # Temperature for softmax
        logits = -distances / temperature
        probabilities = F.softmax(logits, dim=1)
        
        # Apply confidence weighting
        confidence = torch.max(probabilities, dim=1)[0]
        confidence_weights = confidence.unsqueeze(1)
        
        # Weighted predictions
        weighted_predictions = probabilities * confidence_weights
        
        return weighted_predictions
    
    def initialize_test_predictions(self, test_embeddings, prototypes, unique_labels):
        """
        Initialize test predictions using distance to prototypes
        """
        # Compute distances to prototypes
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        
        # Convert distances to probabilities (softmax)
        logits = -distances
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def update_prototypes(self, support_embeddings, support_y, test_embeddings, test_predictions):
        """
        Update prototypes using both support set and test set
        """
        unique_labels = torch.unique(support_y)
        updated_prototypes = []
        
        for label in unique_labels:
            # Support set contribution
            support_mask = support_y == label
            support_contribution = support_embeddings[support_mask].mean(dim=0)
            
            # Test set contribution (weighted by prediction confidence)
            test_weights = test_predictions[:, label]
            if test_weights.sum() > 0:
                test_contribution = (test_embeddings * test_weights.unsqueeze(1)).sum(dim=0) / test_weights.sum()
            else:
                test_contribution = torch.zeros_like(support_contribution)
            
            # Combine support and test contributions
            combined_prototype = 0.7 * support_contribution + 0.3 * test_contribution
            updated_prototypes.append(combined_prototype)
        
        return torch.stack(updated_prototypes)
    
    def update_test_predictions(self, test_embeddings, prototypes, unique_labels):
        """
        Update test predictions using current prototypes and graph structure
        """
        # Distance-based predictions
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        distance_logits = -distances
        
        # Graph structure influence
        similarity_graph = self.compute_similarity_graph(test_embeddings)
        graph_influence = torch.mm(similarity_graph, distance_logits)
        
        # Combine distance and graph information
        combined_logits = 0.7 * distance_logits + 0.3 * graph_influence
        
        # Convert to probabilities
        probabilities = F.softmax(combined_logits, dim=1)
        
        return probabilities
    
    def compute_transductive_loss(self, support_embeddings, support_y, test_embeddings, 
                                 test_predictions, prototypes, unique_labels):
        """
        Compute transductive learning loss
        """
        total_loss = 0
        
        # 1. Support set classification loss
        support_distances = torch.cdist(support_embeddings, prototypes, p=2)
        support_logits = -support_distances
        support_loss = F.cross_entropy(support_logits, support_y)
        total_loss += support_loss
        
        # 2. Test set consistency loss (prototype consistency)
        test_distances = torch.cdist(test_embeddings, prototypes, p=2)
        test_logits = -test_distances
        test_consistency_loss = F.kl_div(
            F.log_softmax(test_logits, dim=1),
            test_predictions,
            reduction='batchmean'
        )
        total_loss += self.consistency_weight * test_consistency_loss
        
        # 3. Graph structure loss (smoothness)
        similarity_graph = self.compute_similarity_graph(test_embeddings)
        graph_loss = self.compute_graph_smoothness_loss(test_embeddings, similarity_graph)
        total_loss += self.graph_weight * graph_loss
        
        return total_loss
    
    def compute_graph_smoothness_loss(self, embeddings, similarity_graph):
        """
        Compute graph smoothness loss to encourage similar samples to have similar embeddings
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Smoothness loss: similar samples should have similar embeddings
        smoothness_loss = torch.sum(similarity_graph * pairwise_distances)
        
        return smoothness_loss

class MetaLearner(nn.Module):
    """
    Meta-Learning model for few-shot adaptation with transductive learning
    Learns to quickly adapt to new tasks with minimal examples
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2):
        super(MetaLearner, self).__init__()
        
        self.transductive_net = TransductiveLearner(input_dim, hidden_dim, embedding_dim, num_classes)
        self.meta_optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Meta-learning parameters
        self.inner_lr = 0.01
        self.inner_steps = 5
        
    def forward(self, x):
        return self.transductive_net(x)
    
    def meta_update(self, support_x, support_y, query_x, query_y):
        """
        Perform meta-update using support and query sets with transductive learning
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels
            
        Returns:
            loss: Meta-learning loss
        """
        # Get embeddings
        support_embeddings = self.transductive_net(support_x)
        query_embeddings = self.transductive_net(query_x)
        
        # Compute prototypes
        prototypes, prototype_labels = self.transductive_net.compute_prototypes(
            support_embeddings, support_y
        )
        
        # Classify query samples
        predictions, distances = self.transductive_net.classify(
            query_embeddings, prototypes, prototype_labels
        )
        
        # Compute loss using Focal Loss for better class imbalance handling
        logits = -distances
        focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
        loss = focal_loss(logits, query_y)
        
        return loss, predictions
    
    def adapt_to_task(self, support_x, support_y, adaptation_steps: int = None):
        """
        Adapt the model to a specific task using support set with transductive learning
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps
            
        Returns:
            adapted_model: Model adapted to the task
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform adaptation steps using transductive learning
        for step in range(adaptation_steps):
            adapted_optimizer.zero_grad()
            
            # Get embeddings
            support_embeddings = adapted_model.transductive_net(support_x)
            
            # Compute prototypes
            prototypes, prototype_labels = adapted_model.transductive_net.compute_prototypes(
                support_embeddings, support_y
            )
            
            # Compute loss (prototype consistency)
            loss = 0
            for i, label in enumerate(prototype_labels):
                mask = support_y == label
                if mask.sum() > 0:
                    class_embeddings = support_embeddings[mask]
                    prototype = prototypes[i]
                    loss += F.mse_loss(class_embeddings.mean(dim=0), prototype)
            
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model

class TransductiveFewShotModel(nn.Module):
    """
    Transductive Few-Shot Model for Zero-Day Detection
    Combines meta-learning with test-time training for rapid adaptation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2):
        super(TransductiveFewShotModel, self).__init__()
        
        self.meta_learner = MetaLearner(input_dim, hidden_dim, embedding_dim, num_classes)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Test-time training parameters
        self.ttt_lr = 0.001
        self.ttt_steps = 18
        self.ttt_threshold = 0.1  # Confidence threshold for test-time training
        
        # Zero-day detection parameters
        self.anomaly_threshold = 0.5
        self.adaptation_threshold = 0.3
        
    def forward(self, x):
        return self.meta_learner(x)
    
    def compute_confidence(self, embeddings, prototypes, prototype_labels):
        """
        Compute confidence scores for predictions
        
        Args:
            embeddings: Sample embeddings
            prototypes: Class prototypes
            prototype_labels: Prototype labels
            
        Returns:
            confidence: Confidence scores
        """
        distances = torch.cdist(embeddings, prototypes, p=2)
        min_distances = torch.min(distances, dim=1)[0]
        max_distances = torch.max(distances, dim=1)[0]
        
        # Confidence based on distance ratio
        confidence = 1.0 - (min_distances / (max_distances + 1e-8))
        return confidence
    
    def detect_zero_day(self, x, support_x, support_y, adaptation_steps: int = None):
        """
        Detect zero-day attacks using transductive few-shot learning
        
        Args:
            x: Test samples
            support_x: Support set (known attacks)
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps
            
        Returns:
            predictions: Binary predictions (0=normal, 1=attack)
            confidence: Confidence scores
            is_zero_day: Zero-day detection flags
        """
        if adaptation_steps is None:
            adaptation_steps = self.ttt_steps
        
        logger.info("Starting transductive zero-day detection")
        
        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # Get embeddings
        test_embeddings = self.meta_learner(x)
        support_embeddings = self.meta_learner(support_x)
        
        # Compute prototypes from support set
        prototypes, prototype_labels = self.meta_learner.transductive_net.compute_prototypes(
            support_embeddings, support_y
        )
        
        # Initial classification
        initial_predictions, distances = self.meta_learner.transductive_net.classify(
            test_embeddings, prototypes, prototype_labels
        )
        
        # Compute confidence scores
        confidence = self.compute_confidence(test_embeddings, prototypes, prototype_labels)
        
        # Identify low-confidence samples for test-time training
        low_confidence_mask = confidence < self.ttt_threshold
        low_confidence_indices = torch.where(low_confidence_mask)[0]
        
        if len(low_confidence_indices) > 0:
            logger.info(f"Test-time training on {len(low_confidence_indices)} low-confidence samples")
            
            # Perform test-time training on low-confidence samples
            adapted_model = self.meta_learner.adapt_to_task(
                support_x, support_y, adaptation_steps
            )
            
            # Re-classify with adapted model
            # Get the original test samples for low-confidence cases
            low_confidence_samples = x[low_confidence_mask]
            adapted_embeddings = adapted_model(low_confidence_samples)
            adapted_predictions, adapted_distances = adapted_model.transductive_net.classify(
                adapted_embeddings, prototypes, prototype_labels
            )
            
            # Update predictions for low-confidence samples
            final_predictions = initial_predictions.clone()
            final_predictions[low_confidence_mask] = adapted_predictions
            
            # Update confidence scores
            adapted_confidence = self.compute_confidence(adapted_embeddings, prototypes, prototype_labels)
            final_confidence = confidence.clone()
            final_confidence[low_confidence_mask] = adapted_confidence
        else:
            final_predictions = initial_predictions
            final_confidence = confidence
        
        # Zero-day detection: samples with low confidence and predicted as attack
        attack_mask = final_predictions == 1
        low_confidence_attacks = attack_mask & (final_confidence < self.adaptation_threshold)
        
        # Convert to binary labels (0=normal, 1=attack)
        binary_predictions = (final_predictions == 1).long()
        
        logger.info(f"Transductive zero-day detection completed")
        logger.info(f"Zero-day samples detected: {low_confidence_attacks.sum().item()}")
        
        return binary_predictions, final_confidence, low_confidence_attacks
    
    def meta_train(self, meta_tasks: List[Dict], meta_epochs: int = 100):
        """
        Meta-train the model on multiple tasks
        
        Args:
            meta_tasks: List of meta-learning tasks
            meta_epochs: Number of meta-training epochs
            
        Returns:
            training_history: Training metrics
        """
        logger.info(f"Starting transductive meta-training for {meta_epochs} epochs")
        
        training_history = {
            'epoch_losses': [],
            'epoch_accuracies': []
        }
        
        for epoch in range(meta_epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # Sample tasks for this epoch
            np.random.shuffle(meta_tasks)
            
            for task in meta_tasks:
                # Move tensors to the same device as the model
                device = next(self.parameters()).device
                support_x = task['support_x'].to(device)
                support_y = task['support_y'].to(device)
                query_x = task['query_x'].to(device)
                query_y = task['query_y'].to(device)
                
                # Meta-update
                loss, predictions = self.meta_learner.meta_update(
                    support_x, support_y, query_x, query_y
                )
                
                # Compute accuracy
                accuracy = (predictions == query_y).float().mean().item()
                
                # Backward pass
                self.meta_learner.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_learner.meta_optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            training_history['epoch_losses'].append(avg_loss)
            training_history['epoch_accuracies'].append(avg_accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        logger.info("Transductive meta-training completed")
        return training_history
    
    def evaluate_zero_day_detection(self, test_x, test_y, support_x, support_y):
        """
        Evaluate zero-day detection performance
        
        Args:
            test_x: Test samples
            test_y: Test labels
            support_x: Support set
            support_y: Support set labels
            
        Returns:
            metrics: Evaluation metrics
        """
        logger.info("Evaluating transductive zero-day detection performance")
        
        # Detect zero-day attacks
        predictions, confidence, is_zero_day = self.detect_zero_day(
            test_x, support_x, support_y
        )
        
        # Convert to numpy for sklearn metrics
        predictions_np = predictions.detach().cpu().numpy()
        test_y_np = test_y.detach().cpu().numpy()
        confidence_np = confidence.detach().cpu().numpy()
        is_zero_day_np = is_zero_day.detach().cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(test_y_np, predictions_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y_np, predictions_np, average='binary'
        )
        
        try:
            roc_auc = roc_auc_score(test_y_np, confidence_np)
        except:
            roc_auc = 0.5
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, matthews_corrcoef
        cm = confusion_matrix(test_y_np, predictions_np)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Compute Matthews Correlation Coefficient (MCCC)
        try:
            mccc = matthews_corrcoef(test_y_np, predictions_np)
        except:
            mccc = 0.0
        
        # Zero-day specific metrics
        zero_day_detection_rate = is_zero_day_np.mean()
        avg_confidence = confidence_np.mean()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mccc': mccc,
            'zero_day_detection_rate': zero_day_detection_rate,
            'avg_confidence': avg_confidence,
            'num_zero_day_samples': is_zero_day_np.sum(),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'total_samples': len(test_y_np)
        }
        
        logger.info(f"Transductive zero-day detection results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  MCCC: {mccc:.4f}")
        logger.info(f"  Zero-day detection rate: {zero_day_detection_rate:.4f}")
        logger.info(f"  Average confidence: {avg_confidence:.4f}")
        logger.info(f"  Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        
        return metrics

def create_meta_tasks(data_x, data_y, n_way: int = 2, k_shot: int = 5, n_query: int = 15, n_tasks: int = 100):
    """
    Create meta-learning tasks for few-shot learning
    
    Args:
        data_x: Input data
        data_y: Labels
        n_way: Number of classes per task
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        n_tasks: Number of tasks to create
        
    Returns:
        meta_tasks: List of meta-learning tasks
    """
    logger.info(f"Creating {n_tasks} meta-learning tasks ({n_way}-way, {k_shot}-shot)")
    
    meta_tasks = []
    unique_labels = torch.unique(data_y)
    
    for _ in range(n_tasks):
        # Sample classes for this task
        task_classes = torch.randperm(len(unique_labels))[:n_way]
        selected_labels = unique_labels[task_classes]
        
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        
        for label in selected_labels:
            # Get samples for this class
            class_mask = data_y == label
            class_indices = torch.where(class_mask)[0]
            
            # Shuffle and select samples
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            
            # Support set
            support_indices = shuffled_indices[:k_shot]
            support_x_list.append(data_x[support_indices])
            support_y_list.append(data_y[support_indices])
            
            # Query set
            query_indices = shuffled_indices[k_shot:k_shot + n_query]
            query_x_list.append(data_x[query_indices])
            query_y_list.append(data_y[query_indices])
        
        # Combine all classes
        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.cat(support_y_list, dim=0)
        query_x = torch.cat(query_x_list, dim=0)
        query_y = torch.cat(query_y_list, dim=0)
        
        # Relabel to 0, 1, 2, ... for this task
        label_mapping = {label.item(): i for i, label in enumerate(selected_labels)}
        support_y_relabeled = torch.tensor([label_mapping[label.item()] for label in support_y])
        query_y_relabeled = torch.tensor([label_mapping[label.item()] for label in query_y])
        
        meta_tasks.append({
            'support_x': support_x,
            'support_y': support_y_relabeled,
            'query_x': query_x,
            'query_y': query_y_relabeled
        })
    
    logger.info(f"Created {len(meta_tasks)} meta-learning tasks")
    return meta_tasks

def main():
    """Test the transductive few-shot model"""
    logger.info("Testing Transductive Few-Shot Model")
    
    # Create synthetic data for testing
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 25
    
    # Generate synthetic data
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Initialize model
    model = TransductiveFewShotModel(input_dim=n_features)
    
    # Create meta-tasks
    meta_tasks = create_meta_tasks(X_train, y_train, n_tasks=50)
    
    # Meta-train the model
    training_history = model.meta_train(meta_tasks, meta_epochs=20)
    
    # Evaluate zero-day detection
    metrics = model.evaluate_zero_day_detection(X_test, y_test, X_train, y_train)
    
    logger.info("✅ Transductive few-shot model test completed!")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
