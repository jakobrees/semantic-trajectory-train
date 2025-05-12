import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import os
import gc
import numpy as np
from reranker_model import TokenWeighter, VocabLookupWeighter, get_device
from fast_dtw import FastDTWEmbeddingSimilarity

class MultiLayerReranker(nn.Module):
	"""Neural reranker that extracts embeddings from multiple specified layers with optimized parallel processing."""
	def __init__(
		self,
		model_name: str = "meta-llama/Llama-2-7b-hf",
		layer_indices: List[int] = [0, 3, 6, 9, 12, 15, 18, 21],
		dtw_layer_indices: List[int] = [6, 9, 12, 15],
		device: Optional[str] = None,
		dtype: Optional[torch.dtype] = None,
		max_length: int = 512,
		normalize_embeddings: bool = True,
		similarity_fn: str = "cosine",
		weight_normalization: str = "linear",  # Changed default to linear
		weighting_mode="full"
	):
		super(MultiLayerReranker, self).__init__()
		# Model configuration
		self.model_name = model_name
		self.layer_indices = layer_indices
		self.dtw_layer_indices = dtw_layer_indices
		self.max_length = max_length
		self.normalize_embeddings = normalize_embeddings
		self.similarity_fn = similarity_fn
		self.weight_normalization = weight_normalization
		self.weighting_mode = weighting_mode

		# Logger setup
		self.logger = logging.getLogger("multi_layer_reranker")

		# Ensure DTW layers are included in layer indices
		for idx in dtw_layer_indices:
			if idx not in layer_indices:
				self.layer_indices.append(idx)
				self.logger.warning(f"Added DTW layer {idx} to layer_indices")

		# Sort layer indices for efficient extraction
		self.layer_indices = sorted(self.layer_indices)

		# Device setup
		if device is None:
			self.device = get_device()
		else:
			self.device = torch.device(device)

		# Auto-select dtype if not specified
		if dtype is None:
			self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32
		else:
			self.dtype = dtype

		self.logger.info(f"Using device: {self.device} with dtype: {self.dtype}")

		# Initialize model components
		self.tokenizer = None
		self.model = None
		self._is_loaded = False

		# Initialize token weighters for each layer
		self.layer_weighters = nn.ModuleDict()
		self.dtw_weighter = None

		# Initialize FastDTWEmbeddingSimilarity
		self.dtw_calculator = FastDTWEmbeddingSimilarity(distance_metric=self.similarity_fn)
		self.logger.info("Using GPU-accelerated FastDTWEmbeddingSimilarity")

	def load_model(self):
		"""Load the model and tokenizer into memory"""
		if self._is_loaded:
			return

		self.logger.info(f"Loading tokenizer: {self.model_name}")
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

		# Fix for tokenizers without pad token
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.logger.info("Set pad_token to eos_token since it was not defined")

		self.logger.info(f"Loading model: {self.model_name}")
		self.model = AutoModel.from_pretrained(
			self.model_name,
			torch_dtype=self.dtype,
			device_map="auto",
			output_hidden_states=True
		)

		# Freeze model parameters
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()

		# Initialize weighters if not done already
		if not self.layer_weighters:
			vocab_size = len(self.tokenizer)
			self.logger.info(f"Initializing weighters with vocab size: {vocab_size}")
			# Initialize layer weighters
			for layer_idx in self.layer_indices:
				self.layer_weighters[f"layer_{layer_idx}"] = VocabLookupWeighter(
					vocab_size, 
					weighting_mode=self.weighting_mode
				)
			# Initialize DTW weighter
			self.dtw_weighter = VocabLookupWeighter(
				vocab_size, 
				weighting_mode=self.weighting_mode
			)
			# Move to device
			self.to(self.device)

		# Move DTW calculator to device
		self.dtw_calculator.to(self.device)

		self._is_loaded = True
		self.logger.info(f"Model loaded successfully with weight_normalization={self.weight_normalization}, weighting_mode={self.weighting_mode}")

	def unload_model(self):
		"""Unload model and tokenizer from memory"""
		if not self._is_loaded:
			return

		self.logger.info("Unloading model...")
		del self.model
		del self.tokenizer
		self.model = None
		self.tokenizer = None
		self._is_loaded = False

		# Clear device memory
		if self.device.type == "cuda":
			torch.cuda.empty_cache()

		# Force garbage collection
		gc.collect()
		self.logger.info("Model unloaded successfully.")

	def encode_multi_layer(self, texts, remove_special_tokens=True, is_query=False):
		"""
		Encode texts and extract embeddings from all specified layers in one forward pass.
		Optimized for batch processing.

		Args:
			texts: Single text or list of texts to encode
			remove_special_tokens: Whether to remove special tokens from outputs
			is_query: Whether the texts are queries or documents

		Returns:
			Dict with layer_data containing embeddings and token info for each layer
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")

		# Handle single text input
		is_single_text = isinstance(texts, str)
		if is_single_text:
			texts = [texts]

		# Tokenize inputs
		batch_encoding = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		)

		# Move to appropriate device
		batch_encoding = {k: v.to(self.device) for k, v in batch_encoding.items()}

		# Forward pass to get embeddings from all layers
		with torch.no_grad():
			outputs = self.model(**batch_encoding, return_dict=True)

		# Extract hidden states for all layers of interest
		all_hidden_states = outputs.hidden_states

		# Create results container for each text
		results = []

		# Process each text in the batch
		for batch_idx, text in enumerate(texts):
			# Get attention mask for actual tokens (excluding padding)
			attention_mask = batch_encoding["attention_mask"][batch_idx]
			seq_length = attention_mask.sum().item()

			# Get token IDs
			token_ids = batch_encoding["input_ids"][batch_idx][:seq_length].tolist()

			# Get token strings
			tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

			# Initialize container for layer-specific data
			layer_data = {}

			# Extract embeddings for each layer of interest
			for layer_idx in self.layer_indices:
				# Ensure layer index is valid
				if layer_idx >= len(all_hidden_states):
					raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(all_hidden_states)} layers.")

				# Get embeddings for this sequence
				embeddings = all_hidden_states[layer_idx][batch_idx, :seq_length].clone()

				# Normalize embeddings if requested
				if self.normalize_embeddings:
					embeddings = F.normalize(embeddings, p=2, dim=1)

				# Filter out special tokens if requested
				if remove_special_tokens:
					special_tokens = set(self.tokenizer.all_special_tokens)
					keep_mask = [token not in special_tokens for token in tokens]

					# Filter embeddings, tokens, and token IDs
					if not any(keep_mask):
						# Handle edge case of all special tokens
						filtered_embeddings = embeddings.new_zeros((0, embeddings.size(1)))
						filtered_tokens = []
						filtered_token_ids = []
						filtered_token_indices = []
					else:
						# Get indices of tokens to keep
						keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

						# Use index_select for GPU efficiency
						select_indices = torch.tensor(keep_indices, device=embeddings.device)
						filtered_embeddings = torch.index_select(embeddings, 0, select_indices)
						filtered_tokens = [tokens[i] for i in keep_indices]
						filtered_token_ids = [token_ids[i] for i in keep_indices]
						filtered_token_indices = keep_indices
				else:
					filtered_embeddings = embeddings
					filtered_tokens = tokens
					filtered_token_ids = token_ids
					filtered_token_indices = list(range(seq_length))

				# Store layer data
				layer_data[layer_idx] = {
					"embeddings": filtered_embeddings,
					"tokens": filtered_tokens,
					"token_ids": filtered_token_ids,
					"token_indices": filtered_token_indices,
					"is_query": is_query  # Store query/document flag
				}

			# Extract token trajectories for DTW layers only if we have DTW layers specified
			trajectories = {}
			if self.dtw_layer_indices:  # Only process if there are DTW layers
				first_layer = self.dtw_layer_indices[0]
				if first_layer in layer_data:  # Make sure the first DTW layer is in layer_data
					for token_idx in range(len(layer_data[first_layer]["tokens"])):
						token = layer_data[first_layer]["tokens"][token_idx]
						token_id = layer_data[first_layer]["token_ids"][token_idx]

						# Extract embedding trajectory across layers
						trajectory = []
						for layer_idx in self.dtw_layer_indices:
							# Find matching token position in each layer
							token_pos = None
							for i, tid in enumerate(layer_data[layer_idx]["token_ids"]):
								if tid == token_id:
									token_pos = i
									break

							if token_pos is not None:
								trajectory.append(layer_data[layer_idx]["embeddings"][token_pos])

						if len(trajectory) == len(self.dtw_layer_indices):
							# Only include complete trajectories
							trajectories[token_idx] = {
								"token": token,
								"token_id": token_id,
								"embeddings": torch.stack(trajectory)
							}

			# Create final result structure
			result = {
				"layer_data": layer_data,
				"trajectories": trajectories,
				"is_query": is_query  # Store query/document flag at the top level too
			}
			results.append(result)

		# Return single result for single input
		if is_single_text:
			return results[0]
		return results

	def calculate_layer_similarity(self, query_data, doc_data, layer_idx):
		"""
		Calculate similarity between query and document for a specific layer.

		Args:
			query_data: Dictionary with query embeddings and tokens from encode_multi_layer
			doc_data: Dictionary with document embeddings and tokens
			layer_idx: Which layer to use for similarity calculation

		Returns:
			Tensor containing similarity score
		"""
		# Get layer-specific data
		query_layer = query_data["layer_data"][layer_idx]
		doc_layer = doc_data["layer_data"][layer_idx]
		query_embeddings = query_layer["embeddings"]
		query_tokens = query_layer["tokens"]
		query_token_ids = query_layer["token_ids"]
		query_token_indices = query_layer["token_indices"]
		doc_embeddings = doc_layer["embeddings"]
		doc_tokens = doc_layer["tokens"]
		doc_token_ids = doc_layer["token_ids"]
		doc_token_indices = doc_layer["token_indices"]

		# Use is_query flags (default to True for query_layer, False for doc_layer)
		is_query_query = query_layer.get("is_query", True)
		is_query_doc = doc_layer.get("is_query", False)

		# Handle empty embeddings case
		if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
			return torch.tensor(0.0, device=self.device)

		# Calculate token-wise similarities - matrix of shape [num_query_tokens, num_doc_tokens]
		if self.similarity_fn == "cosine":
			# Embeddings should already be normalized
			similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())
		elif self.similarity_fn == "dot":
			similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())
		elif self.similarity_fn == "euclidean":
			# Use cdist for efficient distance calculation
			distances = torch.cdist(query_embeddings, doc_embeddings, p=2)
			# Convert distance to similarity
			similarity_matrix = 1.0 / (1.0 + distances)
		else:
			raise ValueError(f"Unsupported similarity function: {self.similarity_fn}")

		# For each query token, find the most similar document token
		max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

		# Get weighter for this layer
		layer_weighter = self.layer_weighters[f"layer_{layer_idx}"]

		# Calculate weights for query tokens
		query_weights = layer_weighter(
			query_tokens, 
			query_token_indices,
			token_ids=query_token_ids,
			is_query=is_query_query
		)

		# Get weights for the corresponding document tokens
		best_doc_indices = [doc_token_indices[idx.item()] if idx.item() < len(doc_token_indices) else 0 
						  for idx in max_indices]
		best_doc_tokens = [doc_tokens[idx.item()] if idx.item() < len(doc_tokens) else "" 
						 for idx in max_indices]
		best_doc_token_ids = [doc_token_ids[idx.item()] if idx.item() < len(doc_token_ids) else 0 
							for idx in max_indices]

		doc_weights = layer_weighter(
			best_doc_tokens,
			best_doc_indices,
			token_ids=best_doc_token_ids,
			is_query=is_query_doc
		)

		# Apply weighting scheme based on mode
		if self.weighting_mode == "query_only":
			combined_weights = query_weights
		elif self.weighting_mode == "none":
			combined_weights = torch.ones_like(query_weights) / len(query_weights)
		else:  # "full"
			combined_weights = query_weights * doc_weights

		# Weight normalization
		if self.weight_normalization == "linear":
			weight_sum = combined_weights.sum()
			if weight_sum > 0:
				normalized_weights = combined_weights / weight_sum
			else:
				normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)
		elif self.weight_normalization == "softmax":
			# Clamp inputs to softmax to avoid NaNs
			combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
			normalized_weights = F.softmax(combined_weights, dim=0)
		else:
			raise ValueError(f"Unsupported weight normalization: {self.weight_normalization}")

		# Calculate weighted sum of similarities
		score = (max_similarities * normalized_weights).sum()
		return score

	def calculate_dtw_similarity(self, query_data, doc_data, top_k=5):
		"""
		Calculate similarity using DTW on token trajectories.
		Optimized with FastDTWEmbeddingSimilarity for parallel GPU computation.

		Args:
			query_data: Dictionary with query trajectories from encode_multi_layer
			doc_data: Dictionary with document trajectories
			top_k: Number of top candidates to consider for each query token

		Returns:
			Tensor containing similarity score
		"""
		query_trajectories = query_data["trajectories"]
		doc_trajectories = doc_data["trajectories"]

		# Use is_query flags
		is_query_query = query_data.get("is_query", True)
		is_query_doc = doc_data.get("is_query", False)

		# Handle empty trajectories case
		if not query_trajectories or not doc_trajectories:
			return torch.tensor(0.0, device=self.device)

		# Use first DTW layer for pre-filtering
		first_layer_idx = self.dtw_layer_indices[0]

		# Get first layer data for similarity calculation
		query_first_layer = query_data["layer_data"][first_layer_idx]
		doc_first_layer = doc_data["layer_data"][first_layer_idx]

		# Calculate first layer similarity matrix for pre-filtering
		query_embeddings = query_first_layer["embeddings"]
		doc_embeddings = doc_first_layer["embeddings"]

		if self.similarity_fn == "cosine" or self.similarity_fn == "dot":
			# Normalize for cosine similarity if needed
			if self.similarity_fn == "cosine" and not self.normalize_embeddings:
				query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
				doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

			# Compute similarity matrix efficiently
			first_layer_sim = torch.mm(query_embeddings, doc_embeddings.t())
		else:
			# Euclidean distance
			distances = torch.cdist(query_embeddings, doc_embeddings, p=2)
			first_layer_sim = 1.0 / (1.0 + distances)

		# Collect trajectories for batch processing
		query_traj_list = []
		query_traj_tokens = []
		query_traj_ids = []
		query_traj_positions = []
		doc_candidates = []  # List of (doc_traj_idx, doc_token, doc_token_id, doc_pos) for each query token

		# Process each query token trajectory
		for q_idx, q_traj_data in query_trajectories.items():
			query_token = q_traj_data["token"]
			query_token_id = q_traj_data["token_id"]
			query_trajectory = q_traj_data["embeddings"]

			# Find corresponding position in first layer data
			q_pos = None
			for i, tid in enumerate(query_first_layer["token_ids"]):
				if tid == query_token_id:
					q_pos = i
					break

			if q_pos is None:
				continue

			# Get top-k candidate document tokens from first layer similarity
			sim_scores = first_layer_sim[q_pos]
			top_k_values, top_k_indices = torch.topk(sim_scores, min(top_k, len(sim_scores)))

			# Find matching document trajectories
			candidates = []
			for k, d_idx_tensor in enumerate(top_k_indices):
				d_idx = d_idx_tensor.item()

				# Get document token ID
				doc_token_id = doc_first_layer["token_ids"][d_idx]

				# Find corresponding trajectory
				d_traj_idx = None
				for idx, traj_data in doc_trajectories.items():
					if traj_data["token_id"] == doc_token_id:
						d_traj_idx = idx
						break

				if d_traj_idx is not None:
					# Find position in first layer
					d_pos = None
					for i, tid in enumerate(doc_first_layer["token_ids"]):
						if tid == doc_token_id:
							d_pos = i
							break

					if d_pos is not None:
						candidates.append((d_traj_idx, doc_trajectories[d_traj_idx]["token"], 
										  doc_token_id, d_pos))

			# If we found candidates, add this query token to the batch
			if candidates:
				query_traj_list.append(query_trajectory)
				query_traj_tokens.append(query_token)
				query_traj_ids.append(query_token_id)
				query_traj_positions.append(q_pos)
				doc_candidates.append(candidates)

		# If no valid pairs were found
		if not query_traj_list:
			return torch.tensor(0.0, device=self.device)

		# Process each query trajectory against its candidates
		max_similarities = []
		query_weights = []
		doc_weights = []

		# Process candidates for each query token
		for q_idx in range(len(query_traj_list)):
			query_trajectory = query_traj_list[q_idx]
			query_token = query_traj_tokens[q_idx]
			query_token_id = query_traj_ids[q_idx]
			query_pos = query_traj_positions[q_idx]
			candidates = doc_candidates[q_idx]

			# For each candidate, calculate DTW similarity
			best_similarity = 0.0
			best_candidate = None

			for candidate in candidates:
				d_traj_idx, doc_token, doc_token_id, doc_pos = candidate
				doc_trajectory = doc_trajectories[d_traj_idx]["embeddings"]

				# Calculate DTW distance using optimized implementation
				similarity = self.dtw_calculator(query_trajectory.unsqueeze(0), 
												doc_trajectory.unsqueeze(0)).item()

				# Update best match
				if similarity > best_similarity:
					best_similarity = similarity
					best_candidate = candidate

			if best_similarity > 0:
				# Add to results
				max_similarities.append(best_similarity)

				# Calculate token weights
				q_weight = self.dtw_weighter(
					[query_token],
					[query_first_layer["token_indices"][query_pos]],
					token_ids=[query_token_id],
					is_query=is_query_query
				)

				d_traj_idx, doc_token, doc_token_id, doc_pos = best_candidate
				d_weight = self.dtw_weighter(
					[doc_token],
					[doc_first_layer["token_indices"][doc_pos]],
					token_ids=[doc_token_id],
					is_query=is_query_doc
				)

				query_weights.append(q_weight)
				doc_weights.append(d_weight)

		# Handle case where no matches were found
		if not max_similarities:
			return torch.tensor(0.0, device=self.device)

		# Convert lists to tensors
		similarities = torch.tensor(max_similarities, device=self.device)
		query_weights = torch.cat(query_weights)
		doc_weights = torch.cat(doc_weights)

		# Apply weighting mode
		if self.weighting_mode == "query_only":
			combined_weights = query_weights
		elif self.weighting_mode == "none":
			combined_weights = torch.ones_like(query_weights) / len(query_weights)
		else:  # "full"
			combined_weights = query_weights * doc_weights

		# Apply weight normalization
		if self.weight_normalization == "linear":
			weight_sum = combined_weights.sum()
			if weight_sum > 0:
				normalized_weights = combined_weights / weight_sum
			else:
				normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)
		elif self.weight_normalization == "softmax":
			combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
			normalized_weights = F.softmax(combined_weights, dim=0)
		else:
			raise ValueError(f"Unsupported weight normalization: {self.weight_normalization}")

		# Calculate final score
		score = (similarities * normalized_weights).sum()
		return score

	def batch_calculate_layer_similarity(self, batch_query_data, batch_doc_data, layer_idx):
		"""
		Calculate similarity between batches of queries and documents for a specific layer.
		Optimized for parallel processing.

		Args:
			batch_query_data: List of dictionaries with query embeddings
			batch_doc_ionaries with document embeddings
			layer_idx: Which layer to use for similarity calculation

		Returns:
			Tensor of similarity scores for each query-document pair
		"""
		batch_size = len(batch_query_data)

		# Prepare containers for batched embeddings and metadata
		all_query_embeddings = []
		all_doc_embeddings = []
		all_query_lengths = []  # Store lengths for proper indexing
		all_doc_lengths = []

		# Collect metadata for token weighting
		query_metadata = []
		doc_metadata = []

		# Collect all embeddings and metadata
		for i in range(batch_size):
			query_data = batch_query_data[i]
			doc_data = batch_doc_data[i]

			query_layer = query_data["layer_data"][layer_idx]
			doc_layer = doc_data["layer_data"][layer_idx]

			# Skip empty embeddings
			if len(query_layer["embeddings"]) == 0 or len(doc_layer["embeddings"]) == 0:
				all_query_embeddings.append(None)
				all_doc_embeddings.append(None)
				all_query_lengths.append(0)
				all_doc_lengths.append(0)
				query_metadata.append(None)
				doc_metadata.append(None)
				continue

			# Collect embeddings
			all_query_embeddings.append(query_layer["embeddings"])
			all_doc_embeddings.append(doc_layer["embeddings"])
			all_query_lengths.append(len(query_layer["embeddings"]))
			all_doc_lengths.append(len(doc_layer["embeddings"]))

			# Get is_query flag
			is_query_query = query_data.get("is_query", True)  # Default True for query
			is_query_doc = doc_data.get("is_query", False)     # Default False for doc

			# Collect metadata for token weighting
			query_metadata.append({
				"tokens": query_layer["tokens"],
				"token_ids": query_layer["token_ids"],
				"token_indices": query_layer["token_indices"],
				"is_query": is_query_query
			})

			doc_metadata.append({
				"tokens": doc_layer["tokens"],
				"token_ids": doc_layer["token_ids"],
				"token_indices": doc_layer["token_indices"],
				"is_query": is_query_doc
			})

		# Initialize result tensor
		scores = torch.zeros(batch_size, device=self.device)

		# Get weighter for this layer
		layer_weighter = self.layer_weighters[f"layer_{layer_idx}"]

		# Process each pair with valid embeddings
		for i in range(batch_size):
			if all_query_embeddings[i] is None or all_doc_embeddings[i] is None:
				scores[i] = 0.0
				continue

			query_emb = all_query_embeddings[i]
			doc_emb = all_doc_embeddings[i]

			# Calculate similarity matrix
			if self.similarity_fn == "cosine":
				similarity_matrix = torch.mm(query_emb, doc_emb.t())
			elif self.similarity_fn == "dot":
				similarity_matrix = torch.mm(query_emb, doc_emb.t())
			elif self.similarity_fn == "euclidean":
				distances = torch.cdist(query_emb, doc_emb, p=2)
				similarity_matrix = 1.0 / (1.0 + distances)

			# Get max similarities for each query token
			max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

			# Apply token weighting if weighter exists
			if layer_weighter is not None:
				query_meta = query_metadata[i]
				doc_meta = doc_metadata[i]

				# Calculate query token weights
				query_weights = layer_weighter(
					query_meta["tokens"],
					query_meta["token_indices"],
					token_ids=query_meta["token_ids"],
					is_query=query_meta["is_query"]
				)

				# Find the best matching document tokens
				best_doc_indices = [doc_meta["token_indices"][idx.item()] if idx.item() < len(doc_meta["token_indices"]) else 0 
								for idx in max_indices]
				best_doc_tokens = [doc_meta["tokens"][idx.item()] if idx.item() < len(doc_meta["tokens"]) else "" 
								for idx in max_indices]
				best_doc_token_ids = [doc_meta["token_ids"][idx.item()] if idx.item() < len(doc_meta["token_ids"]) else 0 
									for idx in max_indices]

				# Calculate doc token weights
				doc_weights = layer_weighter(
					best_doc_tokens,
					best_doc_indices,
					token_ids=best_doc_token_ids,
					is_query=doc_meta["is_query"]
				)

				# Apply weighting mode
				if self.weighting_mode == "query_only":
					combined_weights = query_weights
				elif self.weighting_mode == "none":
					combined_weights = torch.ones_like(query_weights) / len(query_weights)
				else:  # "full"
					combined_weights = query_weights * doc_weights

				# Normalize weights
				if self.weight_normalization == "linear":
					weight_sum = combined_weights.sum()
					if weight_sum > 0:
						normalized_weights = combined_weights / weight_sum
					else:
						normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)
				elif self.weight_normalization == "softmax":
					combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
					normalized_weights = F.softmax(combined_weights, dim=0)

				# Calculate weighted score
				scores[i] = (max_similarities * normalized_weights).sum()
			else:
				# No weighting - use mean
				scores[i] = max_similarities.mean()

		return scores

	def batch_calculate_dtw_similarity(self, batch_query_data, batch_doc_data, top_k=5):
		"""
		Calculate DTW similarity for batches of query-document pairs.
		Optimized to process multiple pairs in parallel.

		Args:
			batch_query_data: List of dictionaries with query trajectories
			batch_doc_data: List of dictionaries with document trajectories
			top_k: Number of top candidates to consider for each query token

		Returns:
			Tensor of DTW similarity scores for each query-document pair
		"""
		batch_size = len(batch_query_data)
		scores = torch.zeros(batch_size, device=self.device)

		# Process batch efficiently but in chunks to manage memory
		for i in range(batch_size):
			query_data = batch_query_data[i]
			doc_data = batch_doc_data[i]

			query_trajectories = query_data["trajectories"]
			doc_trajectories = doc_data["trajectories"]

			# Get is_query flags
			is_query_query = query_data.get("is_query", True)  # Default True for query
			is_query_doc = doc_data.get("is_query", False)     # Default False for doc

			# Handle empty trajectories case
			if not query_trajectories or not doc_trajectories:
				scores[i] = 0.0
				continue

			# Use first DTW layer for pre-filtering
			first_layer_idx = self.dtw_layer_indices[0]

			# Get first layer data for similarity calculation
			query_first_layer = query_data["layer_data"][first_layer_idx]
			doc_first_layer = doc_data["layer_data"][first_layer_idx]

			# Fast first-layer similarity calculation for filtering
			query_embeddings = query_first_layer["embeddings"]
			doc_embeddings = doc_first_layer["embeddings"]

			if self.similarity_fn == "cosine":
				if not self.normalize_embeddings:
					query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
					doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
				first_layer_sim = torch.mm(query_embeddings, doc_embeddings.t())
			else:
				distances = torch.cdist(query_embeddings, doc_embeddings, p=2)
				first_layer_sim = 1.0 / (1.0 + distances)

			# Collect all query trajectories and their metadata
			all_query_trajectories = []
			all_query_tokens = []
			all_query_token_ids = []
			all_query_positions = []
			all_candidate_trajectories = []  # List of lists of candidate doc trajectories for each query
			all_candidate_metadata = []      # List of lists of candidate metadata for each query

			# First, identify all candidates for all query tokens
			for q_idx, q_traj_data in query_trajectories.items():
				query_token = q_traj_data["token"]
				query_token_id = q_traj_data["token_id"]
				query_trajectory = q_traj_data["embeddings"]

				# Find position in first layer
				q_pos = None
				for j, tid in enumerate(query_first_layer["token_ids"]):
					if tid == query_token_id:
						q_pos = j
						break

				if q_pos is None:
					continue

				# Find top-k candidates using first layer similarity
				sim_scores = first_layer_sim[q_pos]
				top_k_values, top_k_indices = torch.topk(sim_scores, min(top_k, len(sim_scores)))

				# Collect candidates for this query token
				candidates_trajectories = []
				candidates_metadata = []

				for k, d_idx_tensor in enumerate(top_k_indices):
					d_idx = d_idx_tensor.item()
					doc_token_id = doc_first_layer["token_ids"][d_idx]

					# Find matching trajectory
					d_traj_idx = None
					for idx, traj_data in doc_trajectories.items():
						if traj_data["token_id"] == doc_token_id:
							d_traj_idx = idx
							break

					if d_traj_idx is None:
						continue

					# Add candidate
					doc_trajectory = doc_trajectories[d_traj_idx]["embeddings"]
					candidates_trajectories.append(doc_trajectory)
					candidates_metadata.append({
						"token": doc_trajectories[d_traj_idx]["token"],
						"token_id": doc_token_id,
						"pos": d_idx,
						"traj_idx": d_traj_idx
					})

				# If we found candidates for this query token
				if candidates_trajectories:
					all_query_trajectories.append(query_trajectory)
					all_query_tokens.append(query_token)
					all_query_token_ids.append(query_token_id)
					all_query_positions.append(q_pos)
					all_candidate_trajectories.append(candidates_trajectories)
					all_candidate_metadata.append(candidates_metadata)

			# If no valid trajectories were found
			if not all_query_trajectories:
				scores[i] = 0.0
				continue

			# Now compute DTW similarities and find the best matches
			max_similarities = []
			query_weights = []
			doc_weights = []

			# Process each query token and its candidates
			for q_idx in range(len(all_query_trajectories)):
				query_trajectory = all_query_trajectories[q_idx]
				candidates = all_candidate_trajectories[q_idx]
				candidate_metadata = all_candidate_metadata[q_idx]

				# Batch compute DTW similarities for this query against all its candidates
				# Stack candidates into a tensor for batch processing
				candidates_tensor = torch.stack(candidates)

				# Expand query to batch dimension
				query_expanded = query_trajectory.unsqueeze(0).expand(len(candidates), -1, -1)

				# Compute DTW distances in batch
				similarities = self.dtw_calculator(query_expanded, candidates_tensor)

				# Find best candidate
				best_idx = torch.argmax(similarities).item()
				best_similarity = similarities[best_idx].item()
				best_candidate = candidate_metadata[best_idx]

				# Add result to lists
				max_similarities.append(best_similarity)

				# Calculate token weights
				q_weight = self.dtw_weighter(
					[all_query_tokens[q_idx]],
					[query_first_layer["token_indices"][all_query_positions[q_idx]]],
					token_ids=[all_query_token_ids[q_idx]],
					is_query=is_query_query
				)

				d_weight = self.dtw_weighter(
					[best_candidate["token"]],
					[doc_first_layer["token_indices"][best_candidate["pos"]]],
					token_ids=[best_candidate["token_id"]],
					is_query=is_query_doc
				)

				query_weights.append(q_weight)
				doc_weights.append(d_weight)

			# If no matches were found
			if not max_similarities:
				scores[i] = 0.0
				continue

			# Convert results to tensors
			similarities_tensor = torch.tensor(max_similarities, device=self.device)
			query_weights_tensor = torch.cat(query_weights)
			doc_weights_tensor = torch.cat(doc_weights)

			# Apply weighting mode
			if self.weighting_mode == "query_only":
				combined_weights = query_weights_tensor
			elif self.weighting_mode == "none":
				combined_weights = torch.ones_like(query_weights_tensor) / len(query_weights_tensor)
			else:  # "full"
				combined_weights = query_weights_tensor * doc_weights_tensor

			# Normalize weights
			if self.weight_normalization == "linear":
				weight_sum = combined_weights.sum()
				if weight_sum > 0:
					normalized_weights = combined_weights / weight_sum
				else:
					normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)
			elif self.weight_normalization == "softmax":
				combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
				normalized_weights = F.softmax(combined_weights, dim=0)

			# Calculate final score
			scores[i] = (similarities_tensor * normalized_weights).sum()

		return scores

	def forward(self, query_data, doc_data, model_type="all"):
		"""
		Forward pass that can run all models or a specific model type.

		Args:
			query_data: Output from encode_multi_layer for query
			doc_ document
			model_type: "all", "layer_{idx}", or "dtw"

		Returns:
			Dict mapping model names to similarity scores
		"""
		if model_type == "all":
			# Run all models
			results = {}

			# Layer models
			for layer_idx in self.layer_indices:
				layer_name = f"layer_{layer_idx}"
				results[layer_name] = self.calculate_layer_similarity(
					query_data, doc_data, layer_idx)

			# DTW model
			results["dtw"] = self.calculate_dtw_similarity(query_data, doc_data)

			return results

		elif model_type.startswith("layer_"):
			# Run specific layer model
			layer_idx = int(model_type.split("_")[1])
			if layer_idx not in self.layer_indices:
				raise ValueError(f"Layer {layer_idx} not in configured layers: {self.layer_indices}")

			return {model_type: self.calculate_layer_similarity(query_data, doc_data, layer_idx)}

		elif model_type == "dtw":
			# Run DTW model
			return {"dtw": self.calculate_dtw_similarity(query_data, doc_data)}

		else:
			raise ValueError(f"Unknown model_type: {model_type}")

	def batch_forward(self, batch_query_data, batch_doc_data, model_type="all"):
		"""
		Batch forward pass for multiple query-document pairs.

		Args:
			batch_query_data: List of outputs from encode_multi_layer for queries
			batch_doc_data: List of outputs from encode_multi_layer for documents
			model_type: "all", "layer_{idx}", or "dtw"

		Returns:
			Dict mapping model names to tensors of similarity scores
		"""
		batch_size = len(batch_query_data)

		if model_type == "all":
			# Run all models
			results = {}

			# Layer models
			for layer_idx in self.layer_indices:
				layer_name = f"layer_{layer_idx}"
				results[layer_name] = self.batch_calculate_layer_similarity(
					batch_query_data, batch_doc_data, layer_idx)

			# DTW model
			results["dtw"] = self.batch_calculate_dtw_similarity(
				batch_query_data, batch_doc_data)

			return results

		elif model_type.startswith("layer_"):
			# Run specific layer model
			layer_idx = int(model_type.split("_")[1])
			if layer_idx not in self.layer_indices:
				raise ValueError(f"Layer {layer_idx} not in configured layers: {self.layer_indices}")

			return {model_type: self.batch_calculate_layer_similarity(
				batch_query_data, batch_doc_data, layer_idx)}

		elif model_type == "dtw":
			# Run DTW model
			return {"dtw": self.batch_calculate_dtw_similarity(
				batch_query_data, batch_doc_data)}

		else:
			raise ValueError(f"Unknown model_type: {model_type}")

	def rerank(
		self,
		query: str,
		documents: List[str],
		model_type: str = "all",
		batch_size: int = 16,
		return_scores: bool = True
	):
		"""
		Rerank a list of documents based on similarity to query.

		Args:
			query: Query string
			documents: List of document strings to rank
			model_type: Which model to use for ranking ("all", "layer_{idx}", "dtw")
			batch_size: Number of documents to process at once
			return_scores: Whether to return scores along with indices

		Returns:
			Dict mapping model names to rankings, where each ranking is either:
			- List of document indices sorted by relevance (if return_scores=False)
			- Tuple of (sorted_indices, scores) (if return_scores=True)
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")

		# Encode query (extract all layers at once) - mark as query
		query_data = self.encode_multi_layer(query, is_query=True)

		# Initialize results dictionary
		if model_type == "all":
			results = {f"layer_{idx}": [] for idx in self.layer_indices}
			results["dtw"] = []
		else:
			results = {model_type: []}

		# Process documents in batches
		all_doc_scores = {k: [] for k in results.keys()}

		for i in range(0, len(documents), batch_size):
			batch_docs = documents[i:i+batch_size]

			# Encode batch of documents - mark as documents (not queries)
			doc_data_list = self.encode_multi_layer(batch_docs, is_query=False)

			# Process each document in the batch
			for j, doc_data in enumerate(doc_data_list):
				# Get scores for relevant model(s)
				scores = self.forward(query_data, doc_data, model_type)

				# Store scores
				for model_name, score in scores.items():
					if isinstance(score, torch.Tensor):
						all_doc_scores[model_name].append(score.detach().cpu().item())
					else:
						all_doc_scores[model_name].append(score)

		# Calculate rankings for each model
		rankings = {}
		for model_name, scores in all_doc_scores.items():
			# Sort indices by descending score
			sorted_indices_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
			sorted_indices = [idx for idx, _ in sorted_indices_and_scores]

			if return_scores:
				sorted_scores = [scores[i] for i in sorted_indices]
				rankings[model_name] = (sorted_indices, sorted_scores)
			else:
				rankings[model_name] = sorted_indices

		return rankings

	def __enter__(self):
		"""Enable context manager usage with 'with' statement."""
		self.load_model()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Ensure model is unloaded when exiting context."""
		self.unload_model()