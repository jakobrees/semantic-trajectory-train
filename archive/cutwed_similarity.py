import torch
import torch.nn as nn
import logging
from typing import List
from fast_dtw import FastDTWEmbeddingSimilarity, batched_token_trajectory_similarity

class DTWCalculator(nn.Module):
    """Optimized DTW calculator for embedding trajectories"""
    def __init__(self):
        super(DTWCalculator, self).__init__()
        self.logger = logging.getLogger("dtw_calculator")
        self.similarity_fn = FastDTWEmbeddingSimilarity(distance_metric="cosine")
        self.logger.info("Using GPU-accelerated DTW implementation")
    
    def forward(self, trajectory1, trajectory2):
        """
        Calculate DTW distance between two trajectory tensors
        
        Args:
            trajectory1: Tensor of shape [n_layers, embedding_dim]
            trajectory2: Tensor of shape [n_layers, embedding_dim]
            
        Returns:
            Tensor containing DTW distance
        """
        distance = self.similarity_fn(trajectory1, trajectory2)
        # Convert distance to similarity score (1 / (1 + distance))
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def batch_compare(self, query_trajectories: List[torch.Tensor], 
                      doc_trajectories: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute similarities between multiple query and document trajectories in parallel
        
        Args:
            query_trajectories: List of query token trajectories
            doc_trajectories: List of document token trajectories
            
        Returns:
            Similarity matrix [len(query_trajectories), len(doc_trajectories)]
        """
        return batched_token_trajectory_similarity(
            query_trajectories, 
            doc_trajectories,
            distance_metric="cosine",
            device=self.similarity_fn.weight.device if hasattr(self.similarity_fn, 'weight') else None
        )