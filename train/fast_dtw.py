import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Optional

class FastDTWEmbeddingSimilarity(nn.Module):
	"""
	GPU-accelerated, batched DTW similarity module for comparing embedding trajectories.
	Optimized for the specific use case of comparing layer-wise embeddings in Transformer models.
	"""
	def __init__(self, distance_metric: str = "cosine"):
		super(FastDTWEmbeddingSimilarity, self).__init__()
		self.distance_metric = distance_metric

	def compute_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""
		Compute pairwise distances between all points in two embedding sequences.
		Works with batched inputs.

		Args:
			x: Tensor of shape [batch_size, n_points, embedding_dim] or [n_points, embedding_dim]
			y: Tensor of shape [batch_size, m_points, embedding_dim] or [m_points, embedding_dim]

		Returns:
			Tensor of shape [batch_size, n_points, m_points] or [n_points, m_points]
		"""
		# Handle non-batched inputs
		orig_x_dim = x.dim()
		orig_y_dim = y.dim()
		if x.dim() == 2:
			x = x.unsqueeze(0)
		if y.dim() == 2:
			y = y.unsqueeze(0)

		batch_size, n, dim_x = x.shape
		_, m, dim_y = y.shape

		if self.distance_metric == "cosine":
			# Normalize embeddings for cosine distance
			x_norm = F.normalize(x, p=2, dim=2)
			y_norm = F.normalize(y, p=2, dim=2)

			# Cosine similarity matrix (batched dot product)
			# Shape: [batch_size, n, m]
			similarity = torch.bmm(x_norm, y_norm.transpose(1, 2))

			# Convert to distance (1 - similarity)
			# Clamp to avoid numerical issues
			distance = 1.0 - similarity.clamp(-1.0, 1.0)

		elif self.distance_metric == "euclidean":
			# Reshape for broadcasting
			x_expanded = x.unsqueeze(2)  # [batch_size, n, 1, dim]
			y_expanded = y.unsqueeze(1)  # [batch_size, 1, m, dim]

			# Compute squared Euclidean distance
			# Shape: [batch_size, n, m]
			distance = torch.sum((x_expanded - y_expanded)**2, dim=3) / dim_x

			# Add sqrt if needed - often skipped in practice as it preserves order
			distance = torch.sqrt(distance)

		elif self.distance_metric == "manhattan":
			# Reshape for broadcasting
			x_expanded = x.unsqueeze(2)  # [batch_size, n, 1, dim]
			y_expanded = y.unsqueeze(1)  # [batch_size, 1, m, dim]

			# Compute Manhattan distance
			# Shape: [batch_size, n, m]
			distance = torch.sum(torch.abs(x_expanded - y_expanded), dim=3)

		else:
			raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

		# Return with original dimensions
		if orig_x_dim == 2 and orig_y_dim == 2:
			return distance.squeeze(0)

		return distance

	@torch.jit.export
	def dtw_forward_pass(self, distance_matrix: torch.Tensor, band_radius: Optional[int] = None) -> torch.Tensor:
		"""
		Compute the DTW distance using forward dynamic programming.
		JIT-optimized implementation.

		Args:
			distance_matrix: Pairwise distance matrix [batch_size, n, m] or [n, m]
			band_radius: Sakoe-Chiba band radius (None for no constraint)

		Returns:
			DTW distance for each batch element
		"""
		# Handle non-batched inputs
		if distance_matrix.dim() == 2:
			distance_matrix = distance_matrix.unsqueeze(0)

		batch_size, n, m = distance_matrix.shape
		device = distance_matrix.device

		# Default band radius if not specified
		if band_radius is None:
			band_radius = min(n, m) // 2

		# Initialize accumulated cost matrix
		# Adding inf outside the band constraints
		acc_cost = torch.full((batch_size, n, m), float('inf'), device=device)

		# First element is just the distance
		acc_cost[:, 0, 0] = distance_matrix[:, 0, 0]

		# Fill first row and column (with band constraint)
		for i in range(1, min(n, band_radius + 1)):
			acc_cost[:, i, 0] = acc_cost[:, i-1, 0] + distance_matrix[:, i, 0]

		for j in range(1, min(m, band_radius + 1)):
			acc_cost[:, 0, j] = acc_cost[:, 0, j-1] + distance_matrix[:, 0, j]

		# Fill the rest of the matrix within the band constraint
		for i in range(1, n):
			# Determine band boundaries for column j
			j_start = max(1, i - band_radius)
			j_end = min(m, i + band_radius + 1)

			for j in range(j_start, j_end):
				min_prev_cost = torch.min(
					torch.min(acc_cost[:, i-1, j], acc_cost[:, i, j-1]),
					acc_cost[:, i-1, j-1]
				)
				acc_cost[:, i, j] = min_prev_cost + distance_matrix[:, i, j]

		# Return final DTW distances
		return acc_cost[:, n-1, m-1]

	def forward(self, 
				seq1: torch.Tensor,
				seq2: torch.Tensor,
				band_radius: Optional[int] = None) -> torch.Tensor:
		"""
		Compute DTW similarity between two sequences of embeddings.

		Args:
			seq1: First sequence of embeddings [batch_size, n_layers, embedding_dim] or [n_layers, embedding_dim]
			seq2: Second sequence of embeddings [batch_size, n_layers, embedding_dim] or [n_layers, embedding_dim]
			band_radius: Sakoe-Chiba band radius (defaults to half of shorter sequence)

		Returns:
			DTW distances [batch_size] or scalar
		"""
		# Compute distance matrix
		distance_matrix = self.compute_distance_matrix(seq1, seq2)

		# Compute DTW distance
		dtw_distance = self.dtw_forward_pass(distance_matrix, band_radius)

		# Return distances (squeeze if input was non-batched)
		if seq1.dim() == 2 and seq2.dim() == 2:
			return dtw_distance.squeeze(0)
		return dtw_distance

def batched_token_trajectory_similarity(
	query_trajectories: List[torch.Tensor],
	doc_trajectories: List[torch.Tensor],
	distance_metric: str = "cosine",
	device: torch.device = None
) -> torch.Tensor:
	"""
	Compute DTW similarities between multiple query and document token trajectories in parallel.

	Args:
		query_trajectories: List of query token trajectories [n_layers, embedding_dim]
		doc_trajectories: List of document token trajectories [n_layers, embedding_dim]
		distance_metric: Distance metric ("cosine", "euclidean", "manhattan")
		device: Device to run computation on

	Returns:
		Similarity matrix of shape [len(query_trajectories), len(doc_trajectories)]
	"""
	if not query_trajectories or not doc_trajectories:
		return torch.tensor(0.0)

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Stack trajectories into batches
	query_batch = torch.stack(query_trajectories).to(device)  # [num_queries, n_layers, dim]
	doc_batch = torch.stack(doc_trajectories).to(device)      # [num_docs, n_layers, dim]

	# Create similarity calculator
	similarity_fn = FastDTWEmbeddingSimilarity(distance_metric=distance_metric).to(device)

	# Initialize similarity matrix
	num_queries = len(query_trajectories)
	num_docs = len(doc_trajectories)
	similarity_matrix = torch.zeros((num_queries, num_docs), device=device)

	# Process in chunks if needed for very large numbers of trajectories
	chunk_size = 1000  # Can be tuned based on available memory

	for i in range(0, num_queries, chunk_size):
		q_chunk = query_batch[i:i+chunk_size]

		for j in range(0, num_docs, chunk_size):
			d_chunk = doc_batch[j:j+chunk_size]

			# For each query in the chunk, compute DTW against all docs in chunk
			for q_idx, query in enumerate(q_chunk):
				# Expand query to match batch dimension of docs
				query_expanded = query.unsqueeze(0).expand(len(d_chunk), -1, -1)

				# Compute DTW distances for this query against all docs in chunk
				distances = similarity_fn(query_expanded, d_chunk)

				# Convert distances to similarities (1 / (1 + distance))
				chunk_similarities = 1.0 / (1.0 + distances)

				# Store in similarity matrix
				similarity_matrix[i+q_idx, j:j+len(d_chunk)] = chunk_similarities

	return similarity_matrix