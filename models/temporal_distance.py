#!/usr/bin/env python3
"""
Temporal Distance Metrics for TCN-based Few-Shot Learning
Implements DTW (Dynamic Time Warping) and other temporal distance functions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DTWDistance(nn.Module):
    """
    Dynamic Time Warping Distance for temporal sequences
    """
    
    def __init__(self, max_warping_window: Optional[int] = None):
        super(DTWDistance, self).__init__()
        self.max_warping_window = max_warping_window
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute DTW distance between two temporal sequences
        
        Args:
            x: First sequence (batch_size, features, time_steps)
            y: Second sequence (batch_size, features, time_steps)
            
        Returns:
            DTW distance (batch_size,)
        """
        batch_size, features, time_x = x.shape
        _, _, time_y = y.shape
        
        # Compute pairwise distances
        x_expanded = x.unsqueeze(3)  # (batch_size, features, time_x, 1)
        y_expanded = y.unsqueeze(2)  # (batch_size, features, 1, time_y)
        
        # Euclidean distance for each timestep pair
        distances = torch.sqrt(torch.sum((x_expanded - y_expanded) ** 2, dim=1))  # (batch_size, time_x, time_y)
        
        # Compute DTW using dynamic programming
        dtw_distances = []
        for b in range(batch_size):
            dtw_dist = self._compute_dtw(distances[b], self.max_warping_window)
            dtw_distances.append(dtw_dist)
        
        return torch.stack(dtw_distances)
    
    def _compute_dtw(self, distance_matrix: torch.Tensor, max_warping_window: Optional[int] = None) -> torch.Tensor:
        """
        Compute DTW distance using dynamic programming
        
        Args:
            distance_matrix: (time_x, time_y) distance matrix
            max_warping_window: Maximum warping window size
            
        Returns:
            DTW distance (scalar)
        """
        time_x, time_y = distance_matrix.shape
        
        if max_warping_window is None:
            max_warping_window = max(time_x, time_y)
        
        # Initialize DP table
        dtw_matrix = torch.full((time_x + 1, time_y + 1), float('inf'), device=distance_matrix.device)
        dtw_matrix[0, 0] = 0
        
        # Fill DP table
        for i in range(1, time_x + 1):
            for j in range(1, time_y + 1):
                if abs(i - j) <= max_warping_window:
                    cost = distance_matrix[i - 1, j - 1]
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],      # insertion
                        dtw_matrix[i, j - 1],      # deletion
                        dtw_matrix[i - 1, j - 1]   # match
                    )
        
        return dtw_matrix[time_x, time_y]

class TemporalPrototypicalNetworks(nn.Module):
    """
    Temporal Prototypical Networks using DTW distance
    """
    
    def __init__(self, feature_dim: int, num_classes: int = 2, dtw_window: Optional[int] = None):
        super(TemporalPrototypicalNetworks, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dtw_distance = DTWDistance(max_warping_window=dtw_window)
        
    def forward(self, support_sequences: torch.Tensor, support_labels: torch.Tensor, 
                query_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal prototypical networks
        
        Args:
            support_sequences: (n_support, feature_dim, time_steps)
            support_labels: (n_support,)
            query_sequences: (n_query, feature_dim, time_steps)
            
        Returns:
            Logits (n_query, num_classes)
        """
        # Compute prototypes for each class
        prototypes = self._compute_temporal_prototypes(support_sequences, support_labels)
        
        # Compute distances from queries to prototypes
        query_distances = []
        for i in range(query_sequences.shape[0]):
            query_seq = query_sequences[i:i+1]  # (1, feature_dim, time_steps)
            distances = []
            
            for class_id in range(self.num_classes):
                if class_id in prototypes:
                    prototype_seq = prototypes[class_id]  # (1, feature_dim, time_steps)
                    dist = self.dtw_distance(query_seq, prototype_seq)
                    distances.append(dist)
                else:
                    # If no support samples for this class, use large distance
                    distances.append(torch.tensor(float('inf'), device=query_sequences.device))
            
            query_distances.append(torch.stack(distances))
        
        distances = torch.stack(query_distances)  # (n_query, num_classes)
        
        # Convert distances to logits (negative distances = higher probability)
        logits = -distances
        
        return logits
    
    def _compute_temporal_prototypes(self, support_sequences: torch.Tensor, 
                                   support_labels: torch.Tensor) -> dict:
        """
        Compute temporal prototypes for each class
        
        Args:
            support_sequences: (n_support, feature_dim, time_steps)
            support_labels: (n_support,)
            
        Returns:
            Dictionary mapping class_id to prototype sequence
        """
        prototypes = {}
        
        for class_id in range(self.num_classes):
            class_mask = (support_labels == class_id)
            if class_mask.sum() > 0:
                class_sequences = support_sequences[class_mask]  # (n_class_samples, feature_dim, time_steps)
                
                # For temporal sequences, we can use different prototype strategies:
                # 1. Mean prototype (average across time)
                # 2. Representative prototype (closest to mean)
                # 3. Multiple prototypes per class
                
                # Using mean prototype for simplicity
                prototype = class_sequences.mean(dim=0, keepdim=True)  # (1, feature_dim, time_steps)
                prototypes[class_id] = prototype
        
        return prototypes

class TemporalCosineSimilarity(nn.Module):
    """
    Temporal Cosine Similarity for sequences
    """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal cosine similarity
        
        Args:
            x: (batch_size, features, time_steps)
            y: (batch_size, features, time_steps)
            
        Returns:
            Similarity scores (batch_size,)
        """
        # Flatten temporal dimension
        x_flat = x.view(x.shape[0], -1)  # (batch_size, features * time_steps)
        y_flat = y.view(y.shape[0], -1)  # (batch_size, features * time_steps)
        
        # Compute cosine similarity
        x_norm = torch.nn.functional.normalize(x_flat, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(y_flat, p=2, dim=1)
        
        similarity = torch.sum(x_norm * y_norm, dim=1)
        return similarity

def test_temporal_distances():
    """Test the temporal distance functions"""
    logger.info("Testing temporal distance functions...")
    
    # Create test data
    batch_size = 2
    features = 64
    time_steps = 10
    
    x = torch.randn(batch_size, features, time_steps)
    y = torch.randn(batch_size, features, time_steps)
    
    # Test DTW distance
    dtw = DTWDistance()
    dtw_dist = dtw(x, y)
    logger.info(f"DTW distance shape: {dtw_dist.shape}")
    logger.info(f"DTW distances: {dtw_dist}")
    
    # Test Temporal Cosine Similarity
    tcs = TemporalCosineSimilarity()
    cos_sim = tcs(x, y)
    logger.info(f"Cosine similarity shape: {cos_sim.shape}")
    logger.info(f"Cosine similarities: {cos_sim}")
    
    # Test Temporal Prototypical Networks
    n_support = 5
    n_query = 3
    support_sequences = torch.randn(n_support, features, time_steps)
    support_labels = torch.randint(0, 2, (n_support,))
    query_sequences = torch.randn(n_query, features, time_steps)
    
    tpn = TemporalPrototypicalNetworks(features, num_classes=2)
    logits = tpn(support_sequences, support_labels, query_sequences)
    logger.info(f"Temporal Prototypical Networks logits shape: {logits.shape}")
    logger.info(f"Logits: {logits}")
    
    logger.info("✅ Temporal distance functions test completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_temporal_distances()

