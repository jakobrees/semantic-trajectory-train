# multi_layer_reranker.py - with modifications for new weighting methods

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import os
import gc
import numpy as np
import pickle
import math

logger = logging.getLogger(__name__)

def get_device():
	"""Get the appropriate device for computation"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		return torch.device("mps")
	else:
		return torch.device("cpu")

class VocabLookupWeighter(nn.Module):
	"""Weighting module that handles token weights through a lookup table"""
	def __init__(
		self, 
		vocab_size: int, 
		init_weight: float = 1.0,
		weighting_mode: str = "full"
	):
		super(VocabLookupWeighter, self).__init__()
		self.token_weights = nn.Parameter(torch.ones(vocab_size) * init_weight)
		self.weighting_mode = weighting_mode

	def forward(
		self, 
		tokens: List[str] = None, 
		token_positions: List[int] = None, 
		token_ids: List[int] = None,
		is_query: bool = True,
		surprise_values: Optional[torch.Tensor] = None,
		original_positions: Optional[List[int]] = None
	) -> torch.Tensor:
		"""
		Look up weights for the given token IDs

		Args:
			tokens: List of token strings
			token_positions: List of token positions in original sequence
			token_ids: List of token IDs
			is_query: Whether this is a query (vs document)
			surprise_values: Optional tensor of surprise values (for surprise-based weighting)
			original_positions: Original positions in the input (for position-based weighting)

		Returns:
			Tensor of weights for each token
		"""
		if token_ids is None or len(token_ids) == 0:
			return torch.tensor([], device=self.token_weights.device)

		# Convert to tensor if needed
		if not isinstance(token_ids, torch.Tensor):
			token_ids_tensor = torch.tensor(token_ids, device=self.token_weights.device)
		else:
			token_ids_tensor = token_ids

		# Handle out-of-vocabulary tokens
		valid_mask = (token_ids_tensor >= 0) & (token_ids_tensor < len(self.token_weights))
		valid_indices = token_ids_tensor[valid_mask]

		# Initialize weights (default 1.0 for out-of-vocabulary)
		weights = torch.ones_like(token_ids_tensor, dtype=self.token_weights.dtype)

		# Look up weights for valid indices
		if len(valid_indices) > 0:
			weights[valid_mask] = self.token_weights[valid_indices]

		return weights

class PositionalWeighter(nn.Module):
	"""Weighting module that uses token positions in document/query"""
	def __init__(
		self, 
		b: float = 3.0,  # Base for exponential weighting (default: 3)
		c: float = 3.0,  # Coefficient for logarithmic component (default: 3)
	):
		super(PositionalWeighter, self).__init__()
		self.b = b
		self.c = c

	def forward(
		self, 
		token_positions: List[int] or torch.Tensor, 
		sequence_length: int,
		is_query: bool = True
	) -> torch.Tensor:
		"""
		Calculate weights based on token positions.
		Formula: weight_i = (b^token_pos_i / sum(b^i)) + 1 + (c*log10(token_count))/(token_count+0.25-token_pos)

		Args:
			token_positions: Positions of tokens in original sequence
			sequence_length: Total length of the sequence (token count)
			is_query: Whether this is a query (vs document)

		Returns:
			Tensor of weights for each token
		"""
		# Check if input is empty using a method that works for both lists and tensors
		if isinstance(token_positions, list) and not token_positions:
			return torch.tensor([], device=get_device())

		if isinstance(token_positions, torch.Tensor) and token_positions.numel() == 0:
			return torch.tensor([], device=token_positions.device)

		# Convert to tensor if needed
		if not isinstance(token_positions, torch.Tensor):
			pos_tensor = torch.tensor(token_positions, dtype=torch.float, device=get_device())
		else:
			pos_tensor = token_positions.float()

		# Calculate base term: b^token_pos_i
		b_power_pos = torch.pow(self.b, pos_tensor)

		# Calculate the normalization sum: sum(b^i) for i from 1 to token_count
		token_count = float(sequence_length)
		if token_count > 0:
			# Generate powers: [b^1, b^2, ..., b^token_count]
			indices = torch.arange(1, token_count + 1, device=pos_tensor.device)
			all_powers = torch.pow(self.b, indices)
			normalization_sum = all_powers.sum()
		else:
			normalization_sum = 1.0  # Avoid division by zero

		# Calculate first term: b^token_pos_i / sum(b^i)
		first_term = b_power_pos / normalization_sum

		# Calculate second term: 1
		second_term = 1.0

		# Calculate third term: (c*log10(token_count))/(token_count+0.25-token_pos)
		# Use log base 10 as specified
		log10_token_count = torch.log10(torch.tensor(max(1.0, token_count)))
		denominator = token_count + 0.25 - pos_tensor
		# Ensure denominator is positive to avoid division by zero or negative values
		denominator = torch.clamp(denominator, min=0.01)
		third_term = (self.c * log10_token_count) / denominator

		# Final weight: first_term + second_term + third_term
		weights = first_term + second_term + third_term

		return weights

class SurpriseWeighter(nn.Module):
	"""Weighting module that uses token surprise values"""
	def __init__(
		self,
		scaling_factor: float = 1.0
	):
		super(SurpriseWeighter, self).__init__()
		self.scaling_factor = scaling_factor

	def forward(
		self, 
		surprise_values: torch.Tensor,
		is_query: bool = True
	) -> torch.Tensor:
		"""
		Calculate weights based on token surprise values.
		Surprise = -log(probability)

		Args:
			surprise_values: Tensor of surprise values for each token
			is_query: Whether this is a query (vs document)

		Returns:
			Tensor of weights for each token
		"""
		if surprise_values is None or len(surprise_values) == 0:
			return torch.tensor([], device=get_device())

		# Simply return the surprise values, optionally scaled
		return surprise_values * self.scaling_factor

class LogFreqQueryWeighter(nn.Module):
	"""Weighting module that only uses log frequency weights for query tokens"""
	def __init__(
		self, 
		vocab_size: int, 
		default_weight: float = 1.0
	):
		super(LogFreqQueryWeighter, self).__init__()
		self.token_weights = nn.Parameter(torch.ones(vocab_size) * default_weight)

	def forward(
		self, 
		token_ids: List[int], 
		is_query: bool = True
	) -> torch.Tensor:
		"""
		Look up log frequency weights for query tokens only.
		For document tokens, returns uniform weights.

		Args:
			token_ids: List of token IDs
			is_query: Whether this is a query (vs document)

		Returns:
			Tensor of weights for each token
		"""
		if token_ids is None or len(token_ids) == 0:
			return torch.tensor([], device=self.token_weights.device)

		# If processing document tokens, return uniform weights
		if not is_query:
			return torch.ones(len(token_ids), device=self.token_weights.device)

		# Otherwise, look up query token weights
		# Convert to tensor if needed
		if not isinstance(token_ids, torch.Tensor):
			token_ids_tensor = torch.tensor(token_ids, device=self.token_weights.device)
		else:
			token_ids_tensor = token_ids

		# Handle out-of-vocabulary tokens
		valid_mask = (token_ids_tensor >= 0) & (token_ids_tensor < len(self.token_weights))
		valid_indices = token_ids_tensor[valid_mask]

		# Initialize weights (default 1.0)
		weights = torch.ones_like(token_ids_tensor, dtype=self.token_weights.dtype)

		# Look up weights for valid indices
		if len(valid_indices) > 0:
			weights[valid_mask] = self.token_weights[valid_indices]

		return weights

class FastDTWEmbeddingSimilarity(nn.Module):
	"""A robust implementation of DTW for embedding similarity calculation."""
	def __init__(self, distance_metric="cosine"):
		super(FastDTWEmbeddingSimilarity, self).__init__()
		self.distance_metric = distance_metric

	def forward(self, seq1, seq2):
		"""
		Calculate DTW similarity between two embedding sequences.
		Args:
			seq1: Tensor of shape [batch_size, seq_len1, embed_dim]
			seq2: Tensor of shape [batch_size, seq_len2, embed_dim]
		Returns:
			Similarity scores of shape [batch_size]
		"""
		# Ensure consistent batch size
		batch_size1 = seq1.size(0)
		batch_size2 = seq2.size(0)

		if batch_size1 != batch_size2:
			# Handle batch size mismatch explicitly
			logger.warning(f"Batch size mismatch: seq1 has size {batch_size1}, seq2 has size {batch_size2}")
			# Process only up to the minimum batch size
			batch_size = min(batch_size1, batch_size2)
			seq1 = seq1[:batch_size]
			seq2 = seq2[:batch_size]
		else:
			batch_size = batch_size1

		similarities = torch.zeros(batch_size, device=seq1.device)

		for i in range(batch_size):
			# Calculate similarity between individual sequences
			similarity = self.compute_single_dtw_similarity(seq1[i], seq2[i])
			similarities[i] = similarity

		return similarities

	def compute_single_dtw_similarity(self, seq1, seq2):
		"""Compute DTW similarity between two single sequences."""
		# Calculate similarity matrix
		if self.distance_metric == "cosine":
			seq1_norm = F.normalize(seq1, p=2, dim=1)
			seq2_norm = F.normalize(seq2, p=2, dim=1)
			sim_matrix = torch.mm(seq1_norm, seq2_norm.t())
			dist_matrix = 1.0 - sim_matrix
		else:
			# Fallback to euclidean distance
			dist_matrix = torch.cdist(seq1, seq2, p=2)

		# Compute DTW matrix
		seq_len1, seq_len2 = seq1.size(0), seq2.size(0)
		dtw_matrix = torch.zeros(seq_len1, seq_len2, device=seq1.device)

		# Initialize first row and column
		dtw_matrix[0, 0] = dist_matrix[0, 0]
		for j in range(1, seq_len2):
			dtw_matrix[0, j] = dtw_matrix[0, j-1] + dist_matrix[0, j]
		for i in range(1, seq_len1):
			dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + dist_matrix[i, 0]

		# Dynamic programming to fill the DTW matrix
		for i in range(1, seq_len1):
			for j in range(1, seq_len2):
				dtw_matrix[i, j] = dist_matrix[i, j] + min(
					dtw_matrix[i-1, j],    # insertion
					dtw_matrix[i, j-1],    # deletion
					dtw_matrix[i-1, j-1]   # match
				)

		# Convert distance to similarity
		dtw_distance = dtw_matrix[-1, -1]
		normalized_distance = dtw_distance / (seq_len1 + seq_len2)
		similarity = 1.0 / (1.0 + normalized_distance)

		return similarity

	def single_sequence_dtw(self, seq1, seq2):
		"""Direct interface for single sequence DTW similarity."""
		return self.compute_single_dtw_similarity(seq1, seq2)

	def to(self, device):
		"""Move module to specified device."""
		super().to(device)
		return self

class MultiLayerReranker(nn.Module):
	"""Neural reranker that extracts embeddings from multiple specified layers."""
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
		weight_normalization: str = "linear",
		weighting_mode: str = "full",
		calc_surprise: bool = True,  # New parameter to control surprise calculation
		pos_weighting_b: float = 0.9,  # Parameter for positional weighting
		pos_weighting_c: float = 1.0,  # Parameter for positional weighting
		surprise_scaling: float = 1.0  # Scaling factor for surprise values
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
		self.calc_surprise = calc_surprise
		self.pos_weighting_b = pos_weighting_b
		self.pos_weighting_c = pos_weighting_c
		self.surprise_scaling = surprise_scaling

		# Ensure DTW layers are included in layer indices
		for idx in dtw_layer_indices:
			if idx not in layer_indices:
				self.layer_indices.append(idx)
				logger.warning(f"Added DTW layer {idx} to layer_indices")

		# Sort layer indices for efficient extraction
		self.layer_indices = sorted(list(set(self.layer_indices)))

		# Device setup
		if device is None:
			self.device = get_device()
		else:
			self.device = torch.device(device)

		# Auto-select dtype if not specified
		if dtype is None:
			self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
		else:
			self.dtype = dtype

		logger.info(f"Using device: {self.device} with dtype: {self.dtype}")

		# Initialize model components
		self.tokenizer = None
		self.model = None
		self._is_loaded = False

		# Initialize token weighters for each layer and weighting type
		self.layer_weighters = nn.ModuleDict()
		self.dtw_weighter = None

		# Initialize positional weighters
		self.positional_weighter = PositionalWeighter(b=pos_weighting_b, c=pos_weighting_c)

		# Initialize surprise weighter
		self.surprise_weighter = SurpriseWeighter(scaling_factor=surprise_scaling)

		# Initialize query-only log frequency weighter
		self.query_log_weighter = None  # Will be initialized after model loading

		# Initialize DTW calculator
		self.dtw_calculator = FastDTWEmbeddingSimilarity(distance_metric=self.similarity_fn)
		logger.info("Using FastDTWEmbeddingSimilarity")

	def load_model(self):
		"""Load the model and tokenizer into memory"""
		if self._is_loaded:
			return

		logger.info(f"Loading tokenizer: {self.model_name}")
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

		# Fix for tokenizers without pad token
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			logger.info("Set pad_token to eos_token since it was not defined")

		logger.info(f"Loading model: {self.model_name}")
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
			logger.info(f"Initializing weighters with vocab size: {vocab_size}")

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

			# Initialize query-only log frequency weighter
			self.query_log_weighter = LogFreqQueryWeighter(
				vocab_size,
				default_weight=1.0
			)

			# Move to device
			self.to(self.device)

		# Move DTW calculator to device
		self.dtw_calculator.to(self.device)

		self._is_loaded = True
		logger.info(f"Model loaded successfully with weight_normalization={self.weight_normalization}, weighting_mode={self.weighting_mode}")

	def calculate_token_surprise(self, input_ids, attention_mask):
		"""
		Calculate the surprise value (-log probability) for each token in the sequence.

		Args:
			input_ids: Token IDs [batch_size, seq_length]
			attention_mask: Attention mask [batch_size, seq_length]

		Returns:
			Tensor of surprise values [batch_size, seq_length]
		"""
		if not self.calc_surprise:
			# Return dummy values if surprise calculation is disabled
			return torch.ones_like(input_ids, dtype=torch.float)

		batch_size, seq_length = input_ids.size()
		surprise_values = torch.ones((batch_size, seq_length), dtype=torch.float, device=input_ids.device)

		try:
			# Use the existing model to compute surprise values
			# For each position, we'll compute the likelihood of the next token
			with torch.no_grad():
				# Get hidden states
				outputs = self.model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					return_dict=True
				)

				# Get the last hidden states
				hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_size]

				# For each position, use the hidden state to predict the next token
				for batch_idx in range(batch_size):
					for pos in range(seq_length - 1):
						if attention_mask[batch_idx, pos] > 0:  # Only process real tokens
							next_token = input_ids[batch_idx, pos + 1]

							# Get the hidden state for this position
							hidden = hidden_states[batch_idx, pos, :]

							# Project to vocabulary (rough approximation for Language Model head)
							# Use a simplified projection using embedding weights transposed
							# This is approximate and assumes embedding weights ≈ output projection
							if hasattr(self.model, 'lm_head'):
								logits = self.model.lm_head(hidden.unsqueeze(0)).squeeze(0)
							else:
								# Approximation - project using embedding matrix
								logits = torch.matmul(hidden, self.model.get_input_embeddings().weight.t())

							# Calculate probability
							probs = F.softmax(logits, dim=0)
							token_prob = probs[next_token].item()

							# Calculate surprise
							token_surprise = -math.log(token_prob + 1e-10)
							surprise_values[batch_idx, pos] = token_surprise

					# Last token doesn't have a "next" token, use average surprise
					if attention_mask[batch_idx, -1] > 0:
						valid_positions = attention_mask[batch_idx, :-1].sum().item()
						if valid_positions > 0:
							surprise_values[batch_idx, -1] = surprise_values[batch_idx, :valid_positions].mean()

		except Exception as e:
			logger.warning(f"Failed to calculate token surprise: {e}")
			logger.warning("Using uniform surprise values instead")
			# Return uniform values as a fallback

		return surprise_values

	def initialize_token_weighter_with_frequencies(
		self,
		weighter: VocabLookupWeighter,
		weights_filepath: str,
		weight_type: str = "log_weights",
		fallback_weight: float = 21.5481
	):
		"""Initialize a VocabLookupWeighter with token frequency weights"""
		logger.info(f"Initializing token weights from {weights_filepath} using {weight_type}")

		# Load weights
		token_weights = self.load_token_weights(weights_filepath, weight_type, fallback_weight)
		if not token_weights:
			logger.warning("Failed to load token weights, using default initialization")
			return

		# Get current weights tensor
		weights_tensor = weighter.token_weights.data
		vocab_size = len(weights_tensor)

		# Create a new weights tensor
		new_weights = torch.ones_like(weights_tensor) * fallback_weight

		# Keep track of tokens initialized and missed
		tokens_initialized = 0
		tokens_missed = 0

		# Iterate through token IDs
		for token_id in range(vocab_size):
			# Check if token ID exists directly in weights dictionary
			if token_id in token_weights:
				new_weights[token_id] = token_weights[token_id]
				tokens_initialized += 1
			# Try with token ID as string
			elif str(token_id) in token_weights:
				new_weights[token_id] = token_weights[str(token_id)]
				tokens_initialized += 1
			else:
				# No weight found - use fallback
				tokens_missed += 1

		# Set the weights
		weighter.token_weights.data = new_weights
		logger.info(f"Initialized {tokens_initialized} token weights "
					f"({tokens_missed} tokens used fallback weight {fallback_weight})")

	def load_token_weights(self, weights_filepath: str, weight_type: str = "log_weights", fallback_weight: float = 21.5481) -> Dict:
		"""Load token weights from a pickle file"""
		if not os.path.exists(weights_filepath):
			logger.warning(f"Error: Token weights file not found at {weights_filepath}")
			return {}

		try:
			with open(weights_filepath, 'rb') as f:
				full_weight_data = pickle.load(f)
				if isinstance(full_weight_data, dict) and weight_type in full_weight_data:
					weights_dict = full_weight_data[weight_type]
					if isinstance(weights_dict, dict):
						logger.info(f"Loaded {len(weights_dict)} token weights ('{weight_type}') from {weights_filepath}")
						return weights_dict
					else:
						logger.warning(f"Error: Weight type '{weight_type}' in {weights_filepath} exists but is not a dictionary")
						return {}
				else:
					if isinstance(full_weight_data, dict):
						logger.warning(f"Error: Weight file {weights_filepath} loaded, but does not contain the key '{weight_type}'.")
					else:
						logger.warning(f"Error: Weight file {weights_filepath} did not load as a dictionary.")
					return {}
		except Exception as e:
			logger.error(f"Error loading token weights from {weights_filepath}: {e}")
			import traceback
			traceback.print_exc()
			return {}

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

		# Calculate surprise values if needed
		if self.calc_surprise:
			surprise_values = self.calculate_token_surprise(
				batch_encoding["input_ids"], 
				batch_encoding["attention_mask"]
			)
		else:
			surprise_values = None

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

			# Get surprise values for this sequence if available
			if surprise_values is not None:
				seq_surprise = surprise_values[batch_idx, :seq_length]
			else:
				seq_surprise = None

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
						filtered_surprise = None if seq_surprise is None else seq_surprise.new_zeros(0)
					else:
						# Get indices of tokens to keep
						keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

						# Use index_select for GPU efficiency
						select_indices = torch.tensor(keep_indices, device=embeddings.device)
						filtered_embeddings = torch.index_select(embeddings, 0, select_indices)
						filtered_tokens = [tokens[i] for i in keep_indices]
						filtered_token_ids = [token_ids[i] for i in keep_indices]
						filtered_token_indices = keep_indices

						# Filter surprise values if available
						if seq_surprise is not None:
							filtered_surprise = torch.index_select(seq_surprise, 0, select_indices)
						else:
							filtered_surprise = None
				else:
					filtered_embeddings = embeddings
					filtered_tokens = tokens
					filtered_token_ids = token_ids
					filtered_token_indices = list(range(seq_length))
					filtered_surprise = seq_surprise

				# Store layer data
				layer_data[layer_idx] = {
					"embeddings": filtered_embeddings,
					"tokens": filtered_tokens,
					"token_ids": filtered_token_ids,
					"token_indices": filtered_token_indices,
					"surprise_values": filtered_surprise,
					"sequence_length": seq_length,
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
						token_pos = layer_data[first_layer]["token_indices"][token_idx]

						# Get surprise value if available
						if layer_data[first_layer]["surprise_values"] is not None:
							surprise = layer_data[first_layer]["surprise_values"][token_idx].item()
						else:
							surprise = None

						# Extract embedding trajectory across layers
						trajectory = []
						for layer_idx in self.dtw_layer_indices:
							# Find matching token position in each layer
							token_pos_in_layer = None
							for i, tid in enumerate(layer_data[layer_idx]["token_ids"]):
								if tid == token_id:
									token_pos_in_layer = i
									break

							if token_pos_in_layer is not None:
								trajectory.append(layer_data[layer_idx]["embeddings"][token_pos_in_layer])

						if len(trajectory) == len(self.dtw_layer_indices):
							# Only include complete trajectories
							trajectories[token_idx] = {
								"token": token,
								"token_id": token_id,
								"token_position": token_pos,
								"surprise": surprise,
								"embeddings": torch.stack(trajectory)
							}

			# Create final result structure
			result = {
				"layer_data": layer_data,
				"trajectories": trajectories,
				"is_query": is_query,  # Store query/document flag at the top level too
				"sequence_length": seq_length
			}

			results.append(result)

		# Return single result for single input
		if is_single_text:
			return results[0]

		return results

	def calculate_layer_similarity(self, query_data, doc_data, layer_idx, weighting_mode=None):
		"""
		Calculate similarity between query and document for a specific layer.
		Args:
			query_data: Dictionary with query embeddings and tokens from encode_multi_layer
			doc_data: Dictionary with document embeddings and tokens
			layer_idx: Which layer to use for similarity calculation
			weighting_mode: Override instance weighting_mode ("uniform", "query_only", "full", "combined",
						   "query_log_only", "positional", "surprise")
		Returns:
			Tensor containing similarity score
		"""
		# Use provided weighting_mode or fall back to instance default
		weighting_mode = weighting_mode or self.weighting_mode

		# Get layer-specific data
		query_layer = query_data["layer_data"][layer_idx]
		doc_layer = doc_data["layer_data"][layer_idx]
		query_embeddings = query_layer["embeddings"]
		query_tokens = query_layer["tokens"]
		query_token_ids = query_layer["token_ids"]
		query_token_indices = query_layer["token_indices"]
		query_surprise = query_layer.get("surprise_values", None)
		query_seq_length = query_data["sequence_length"]

		doc_embeddings = doc_layer["embeddings"]
		doc_tokens = doc_layer["tokens"]
		doc_token_ids = doc_layer["token_ids"]
		doc_token_indices = doc_layer["token_indices"]
		doc_surprise = doc_layer.get("surprise_values", None)
		doc_seq_length = doc_data["sequence_length"]

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

		# Uniform weighting - just return mean similarity
		if weighting_mode == "uniform":
			return max_similarities.mean()

		# New weighting modes
		elif weighting_mode == "query_log_only":
			# Use query log frequency weights only
			query_weights = self.query_log_weighter(
				token_ids=query_token_ids,
				is_query=True
			)
			combined_weights = query_weights

		elif weighting_mode == "positional":
			# Calculate position-based weights
			query_pos_weights = self.positional_weighter(
				token_positions=query_token_indices,
				sequence_length=query_seq_length,
				is_query=True
			)

			# Get corresponding document token positions
			best_doc_indices = [doc_token_indices[idx.item()] if idx.item() < len(doc_token_indices) else 0 
							 for idx in max_indices]
			doc_pos_weights = self.positional_weighter(
				token_positions=best_doc_indices,
				sequence_length=doc_seq_length,
				is_query=False
			)

			# Combine weights
			combined_weights = query_pos_weights

		elif weighting_mode == "surprise":
			# Check if surprise values are available
			if query_surprise is None:
				logger.warning("Surprise values not available, using uniform weights")
				return max_similarities.mean()

			# Use surprise values as weights
			query_surprise_weights = self.surprise_weighter(
				surprise_values=query_surprise,
				is_query=True
			)

			# Get surprise values for the best matching document tokens
			if doc_surprise is not None:
				best_doc_surprise = torch.stack([doc_surprise[idx.item()] if idx.item() < len(doc_surprise) else 
												torch.tensor(1.0, device=self.device) 
												for idx in max_indices])
				doc_surprise_weights = self.surprise_weighter(
					surprise_values=best_doc_surprise,
					is_query=False
				)
				# Combine weights (average query and doc surprise)
				combined_weights = (query_surprise_weights + doc_surprise_weights) / 2.0
			else:
				# Just use query surprise if document surprise is not available
				combined_weights = query_surprise_weights

		else:
			# Original weighting modes from previous implementation
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
			if weighting_mode == "query_only":
				combined_weights = query_weights
			elif weighting_mode == "combined":
				# Average the query and document weights
				combined_weights = (query_weights + doc_weights) / 2.0
			elif weighting_mode == "full":
				combined_weights = query_weights * doc_weights
			else:  # "none" or any other value
				combined_weights = torch.ones_like(query_weights) / len(query_weights)

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

	def calculate_dtw_similarity(self, query_data, doc_data, top_k=5, weighting_mode=None):
		"""
		Calculate similarity using DTW on token trajectories.
		Args:
			query_data: Dictionary with query trajectories from encode_multi_layer
			doc_data: Dictionary with document trajectories
			top_k: Number of top candidates to consider for each query token
			weighting_mode: Override instance weighting_mode
		Returns:
			Tensor containing similarity score
		"""
		# Use provided weighting_mode or fall back to instance default
		weighting_mode = weighting_mode or self.weighting_mode

		query_trajectories = query_data["trajectories"]
		doc_trajectories = doc_data["trajectories"]
		query_seq_length = query_data["sequence_length"]
		doc_seq_length = doc_data["sequence_length"]

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

		# Process each query token trajectory
		max_similarities = []
		query_weights = []
		doc_weights = []
		query_positions = []
		doc_positions = []
		query_surprises = []
		doc_surprises = []

		for q_idx, q_traj_data in query_trajectories.items():
			query_token = q_traj_data["token"]
			query_token_id = q_traj_data["token_id"]
			query_trajectory = q_traj_data["embeddings"]
			query_position = q_traj_data["token_position"]
			query_surprise = q_traj_data.get("surprise", None)

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
				if d_idx >= len(doc_first_layer["token_ids"]):
					continue  # Skip if index is out of bounds

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
									  
			# If we found candidates, process DTW similarity
			if candidates:
				best_similarity = 0.0
				best_candidate = None

				for candidate in candidates:
					d_traj_idx, doc_token, doc_token_id, doc_pos = candidate
					doc_trajectory = doc_trajectories[d_traj_idx]["embeddings"]

					# Calculate DTW similarity using single sequence implementation
					similarity = self.dtw_calculator.single_sequence_dtw(
						query_trajectory, doc_trajectory
					)

					# Update best match
					if similarity > best_similarity:
						best_similarity = similarity
						best_candidate = candidate

				if best_similarity > 0:
					# Add to results
					max_similarities.append(best_similarity)

					# Store positions and surprise values for alternative weighting
					query_positions.append(query_position)
					d_traj_idx, doc_token, doc_token_id, doc_pos = best_candidate
					doc_position = doc_trajectories[d_traj_idx].get("token_position", doc_pos)
					doc_positions.append(doc_position)

					# Store surprise values if available
					if query_surprise is not None:
						query_surprises.append(query_surprise)

					doc_surprise = doc_trajectories[d_traj_idx].get("surprise", None)
					if doc_surprise is not None:
						doc_surprises.append(doc_surprise)

					# For uniform weighting, we don't need token weights
					if weighting_mode not in ["uniform", "query_log_only", "positional", "surprise"]:
						# Calculate token weights using original methods
						q_weight = self.dtw_weighter(
							[query_token],
							[query_position],
							token_ids=[query_token_id],
							is_query=is_query_query
						)

						d_weight = self.dtw_weighter(
							[doc_token],
							[doc_position],
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

		# For uniform weighting, just return mean similarity
		if weighting_mode == "uniform":
			return similarities.mean()

		# New weighting modes
		if weighting_mode == "query_log_only":
			# Get query token IDs for matched trajectories
			query_token_ids = [query_trajectories[q_idx]["token_id"] for q_idx in query_trajectories 
							  if q_idx in range(len(max_similarities))]
			if query_token_ids:
				# Calculate query-only log frequency weights
				combined_weights = self.query_log_weighter(
					token_ids=query_token_ids,
					is_query=True
				)
			else:
				# Fallback to uniform weights
				combined_weights = torch.ones(len(max_similarities), device=self.device) / len(max_similarities)

		elif weighting_mode == "positional":
			# Convert positions to tensors
			if query_positions:
				q_pos_tensor = torch.tensor(query_positions, device=self.device)
				# Calculate position-based weights
				combined_weights = self.positional_weighter(
					token_positions=q_pos_tensor,
					sequence_length=query_seq_length,
					is_query=True
				)
			else:
				# Fallback to uniform weights
				combined_weights = torch.ones(len(max_similarities), device=self.device) / len(max_similarities)

		elif weighting_mode == "surprise":
			# Check if we have surprise values
			if query_surprises:
				# Convert to tensor
				q_surprise_tensor = torch.tensor(query_surprises, device=self.device)
				combined_weights = self.surprise_weighter(
					surprise_values=q_surprise_tensor,
					is_query=True
				)
			else:
				# Fallback to uniform weights
				combined_weights = torch.ones(len(max_similarities), device=self.device) / len(max_similarities)

		else:
			# Original weighting modes from previous implementation
			query_weights = torch.cat(query_weights) if query_weights else torch.ones(len(max_similarities), device=self.device)
			doc_weights = torch.cat(doc_weights) if doc_weights else torch.ones(len(max_similarities), device=self.device)

			# Apply weighting mode
			if weighting_mode == "query_only":
				combined_weights = query_weights
			elif weighting_mode == "combined":
				# Average the query and document weights
				combined_weights = (query_weights + doc_weights) / 2.0
			elif weighting_mode == "full":
				combined_weights = query_weights * doc_weights
			else:  # "none" or any other value
				combined_weights = torch.ones_like(query_weights) / len(query_weights)

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

	def forward(self, query_data, doc_data, model_type="all", weighting_mode=None):
		"""
		Forward pass that can run all models or a specific model type.
		Args:
			query_data: Output from encode_multi_layer for query
			doc_data: Output from encode_multi_layer for document
			model_type: "all", "layer_{idx}", or "dtw"
			weighting_mode: Override instance weighting_mode
		Returns:
			Dict mapping model names to similarity scores
		"""
		# Use provided weighting_mode or fall back to instance default
		weighting_mode = weighting_mode or self.weighting_mode

		if model_type == "all":
			# Run all models
			results = {}

			# Layer models
			for layer_idx in self.layer_indices:
				layer_name = f"layer_{layer_idx}"
				results[layer_name] = self.calculate_layer_similarity(
					query_data, doc_data, layer_idx, weighting_mode)

			# DTW model
			results["dtw"] = self.calculate_dtw_similarity(
				query_data, doc_data, weighting_mode=weighting_mode)

			return results

		elif model_type.startswith("layer_"):
			# Run specific layer model
			layer_idx = int(model_type.split("_")[1])
			if layer_idx not in self.layer_indices:
				raise ValueError(f"Layer {layer_idx} not in configured layers: {self.layer_indices}")

			return {model_type: self.calculate_layer_similarity(
				query_data, doc_data, layer_idx, weighting_mode)}

		elif model_type == "dtw":
			# Run DTW model
			return {"dtw": self.calculate_dtw_similarity(
				query_data, doc_data, weighting_mode=weighting_mode)}

		else:
			raise ValueError(f"Unknown model_type: {model_type}")

	def rerank(
		self,
		query: str,
		documents: List[str],
		model_type: str = "all",
		batch_size: int = 16,
		return_scores: bool = True,
		weighting_mode: Optional[str] = None
	):
		"""
		Rerank a list of documents based on similarity to query.
		Args:
			query: Query string
			documents: List of document strings to rank
			model_type: Which model to use for ranking ("all", "layer_{idx}", "dtw")
			batch_size: Number of documents to process at once
			return_scores: Whether to return scores along with indices
			weighting_mode: Override instance weighting_mode
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
				scores = self.forward(query_data, doc_data, model_type, weighting_mode)

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

	def unload_model(self):
		"""Unload model and tokenizer from memory"""
		if not self._is_loaded:
			return

		logger.info("Unloading model...")
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

		logger.info("Model unloaded successfully.")

	def __enter__(self):
		"""Enable context manager usage with 'with' statement."""
		self.load_model()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Ensure model is unloaded when exiting context."""
		self.unload_model()