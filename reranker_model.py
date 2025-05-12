import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import gc
import logging
logger = logging.getLogger("reranker_model")

# Determine the best available device with proper support
def get_device():
	if torch.backends.mps.is_available():
		return torch.device("mps")
	elif torch.cuda.is_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")

# Base class for token weighting strategies
class TokenWeighter(nn.Module):
	"""Base class for token weighting strategies"""
	def __init__(self, weighting_mode="full"):
		super(TokenWeighter, self).__init__()
		self.weighting_mode = weighting_mode # "full", "query_only", or "none"

	def forward(self, tokens, token_indices, is_query=False, **kwargs):
		"""
		Calculate weights for tokens
		Args:
			tokens: List of token strings
			token_indices: Indices of tokens in the sequence
			is_query: Whether these tokens are from a query or document
			**kwargs: Additional inputs needed for specific weighting strategies
		Returns:
			Tensor of weights for each token
		"""
		raise NotImplementedError("Subclasses must implement this method")

	def batch_forward(self, batch_tokens, batch_token_indices, batch_is_query=None, **kwargs):
		"""
		Calculate weights for batches of tokens - default implementation processes each item separately
		Override this for more efficient batch implementations
		"""
		results = []
		batch_size = len(batch_tokens)

		if batch_is_query is None:
			batch_is_query = [False] * batch_size

		for i in range(batch_size):
			tokens = batch_tokens[i]
			token_indices = batch_token_indices[i]
			is_query = batch_is_query[i]

			# Extract batch-specific kwargs
			item_kwargs = {}
			for key, value in kwargs.items():
				if isinstance(value, list) and len(value) == batch_size:
					item_kwargs[key] = value[i]
				else:
					item_kwargs[key] = value

			weights = self.forward(tokens, token_indices, is_query=is_query, **item_kwargs)
			results.append(weights)

		return results

class VocabLookupWeighter(TokenWeighter):
	"""Learns a weight for each token in the vocabulary"""
	def __init__(self, vocab_size, init_value=1.0, weighting_mode="full"):
		super(VocabLookupWeighter, self).__init__(weighting_mode=weighting_mode)
		self.token_weights = nn.Parameter(torch.ones(vocab_size) * init_value)

	def forward(self, tokens, token_indices, is_query=False, **kwargs):
		"""Get weights based on weighting mode and token context"""
		token_ids = kwargs.get('token_ids')
		if token_ids is None:
			return torch.ones(len(tokens), device=self.token_weights.device)

		# Use index_select for efficient lookup that maintains gradients
		device = self.token_weights.device
		indices = torch.tensor(token_ids, dtype=torch.long, device=device)
		weights = torch.index_select(self.token_weights, 0, indices)

		# Apply weighting based on mode
		if self.weighting_mode == "none":
			# Return uniform weights (still keep gradients flowing)
			return torch.ones_like(weights)
		elif self.weighting_mode == "query_only" and not is_query:
			# For document tokens in query_only mode, return uniform weights
			return torch.ones_like(weights)

		# Otherwise return actual weights
		return weights

	def batch_forward(self, batch_tokens, batch_token_indices, batch_is_query=None, **kwargs):
		"""Efficient batch implementation for vocab lookup"""
		batch_token_ids = kwargs.get('batch_token_ids')

		if batch_token_ids is None:
			# Fall back to sequential processing
			return super().batch_forward(batch_tokens, batch_token_indices, batch_is_query, **kwargs)

		device = self.token_weights.device
		batch_weights = []

		if batch_is_query is None:
			batch_is_query = [False] * len(batch_tokens)

		for i, token_ids in enumerate(batch_token_ids):
			if not token_ids:
				# Handle empty token_ids case
				batch_weights.append(torch.ones(0, device=device))
				continue

			indices = torch.tensor(token_ids, dtype=torch.long, device=device)
			weights = torch.index_select(self.token_weights, 0, indices)

			# Apply weighting based on mode
			if self.weighting_mode == "none":
				weights = torch.ones_like(weights)
			elif self.weighting_mode == "query_only" and not batch_is_query[i]:
				weights = torch.ones_like(weights)

			batch_weights.append(weights)

		return batch_weights

class PositionalWeighter(TokenWeighter):
	"""Weights tokens based on their position in the sequence"""
	def __init__(self, max_length=512, init_slope=0.01, init_bias=0.5, weighting_mode="full"):
		super(PositionalWeighter, self).__init__(weighting_mode=weighting_mode)
		self.slope = nn.Parameter(torch.tensor(init_slope, dtype=torch.float32))
		self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
		self.max_length = max_length

	def forward(self, tokens, token_indices, is_query=False, **kwargs):
		"""Calculate weights based on position and weighting mode"""
		device = self.slope.device

		# Handle weighting modes first
		if self.weighting_mode == "none":
			return torch.ones(len(tokens), device=device)
		elif self.weighting_mode == "query_only" and not is_query:
			return torch.ones(len(tokens), device=device)

		# Calculate positional weights
		positions = torch.tensor(token_indices, dtype=torch.float32, device=device) / self.max_length
		weights = self.slope * positions + self.bias
		weights = F.softplus(weights)  # Ensure weights are positive

		return weights

	def batch_forward(self, batch_tokens, batch_token_indices, batch_is_query=None, **kwargs):
		"""Efficient batch implementation for positional weighting"""
		device = self.slope.device
		batch_weights = []

		if batch_is_query is None:
			batch_is_query = [False] * len(batch_tokens)

		for i, token_indices in enumerate(batch_token_indices):
			if not token_indices:
				# Handle empty token_indices case
				batch_weights.append(torch.ones(0, device=device))
				continue

			# Handle weighting modes first
			if self.weighting_mode == "none":
				batch_weights.append(torch.ones(len(token_indices), device=device))
				continue
			elif self.weighting_mode == "query_only" and not batch_is_query[i]:
				batch_weights.append(torch.ones(len(token_indices), device=device))
				continue

			# Calculate positional weights
			positions = torch.tensor(token_indices, dtype=torch.float32, device=device) / self.max_length
			weights = self.slope * positions + self.bias
			weights = F.softplus(weights)

			batch_weights.append(weights)

		return batch_weights

class SurpriseWeighter(TokenWeighter):
	"""Weights tokens based on their surprise (-log probability)"""
	def __init__(self, init_weight=1.0, init_bias=0.0, weighting_mode="full"):
		super(SurpriseWeighter, self).__init__(weighting_mode=weighting_mode)
		self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32))
		self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

	def forward(self, tokens, token_indices, is_query=False, **kwargs):
		"""Calculate weights based on token probabilities and weighting mode"""
		device = self.weight.device

		# Handle weighting modes first
		if self.weighting_mode == "none":
			return torch.ones(len(tokens), device=device)
		elif self.weighting_mode == "query_only" and not is_query:
			return torch.ones(len(tokens), device=device)

		# Calculate surprise-based weights
		token_probs = kwargs.get('token_probs')
		if token_probs is None or all(p is None for p in token_probs):
			return torch.ones(len(tokens), device=device)

		# Replace None values with a default probability
		cleaned_probs = [0.01 if p is None else p for p in token_probs]
		probs = torch.tensor(cleaned_probs, dtype=torch.float32, device=device)

		# Handle zero probabilities
		epsilon = 1e-10
		probs = torch.clamp(probs, min=epsilon)

		# Calculate surprise: -log(prob)
		surprise = -torch.log(probs)

		# Apply weighting: surprise * weight + bias
		weights = self.weight * surprise + self.bias

		# Ensure weights are positive
		weights = F.softplus(weights)

		return weights

	def batch_forward(self, batch_tokens, batch_token_indices, batch_is_query=None, **kwargs):
		"""Efficient batch implementation for surprise weighting"""
		batch_token_probs = kwargs.get('batch_token_probs')

		if batch_token_probs is None:
			# Fall back to sequential processing
			return super().batch_forward(batch_tokens, batch_token_indices, batch_is_query, **kwargs)

		device = self.weight.device
		batch_weights = []

		if batch_is_query is None:
			batch_is_query = [False] * len(batch_tokens)

		for i, token_probs in enumerate(batch_token_probs):
			# Handle weighting modes first
			if self.weighting_mode == "none":
				batch_weights.append(torch.ones(len(token_probs) if token_probs else 0, device=device))
				continue
			elif self.weighting_mode == "query_only" and not batch_is_query[i]:
				batch_weights.append(torch.ones(len(token_probs) if token_probs else 0, device=device))
				continue

			if not token_probs or all(p is None for p in token_probs):
				# Handle empty or all-None case
				batch_weights.append(torch.ones(len(token_probs) if token_probs else 0, device=device))
				continue

			# Replace None values with a default probability
			cleaned_probs = [0.01 if p is None else p for p in token_probs]
			probs = torch.tensor(cleaned_probs, dtype=torch.float32, device=device)

			# Handle zero probabilities
			epsilon = 1e-10
			probs = torch.clamp(probs, min=epsilon)

			# Calculate surprise: -log(prob)
			surprise = -torch.log(probs)

			# Apply weighting: surprise * weight + bias
			weights = self.weight * surprise + self.bias

			# Ensure weights are positive
			weights = F.softplus(weights)

			batch_weights.append(weights)

		return batch_weights

class CombinedWeighter(TokenWeighter):
	"""Combines multiple weighting strategies"""
	def __init__(self, weighters, weights=None, weighting_mode="full"):
		"""
		Initialize combined weighter
		Args:
			weighters: List of TokenWeighter instances
			weights: Optional weights for each weighter (will be learned if not provided)
			weighting_mode: Weighting mode to use
		"""
		super(CombinedWeighter, self).__init__(weighting_mode=weighting_mode)
		self.weighters = nn.ModuleList(weighters)
		if weights is None:
			self.strategy_weights = nn.Parameter(torch.ones(len(weighters), dtype=torch.float32))
		else:
			self.strategy_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

	def forward(self, tokens, token_indices, is_query=False, **kwargs):
		"""Combine weights from multiple strategies"""
		device = self.strategy_weights.device

		# Handle weighting modes
		if self.weighting_mode == "none":
			return torch.ones(len(tokens), device=device)
		elif self.weighting_mode == "query_only" and not is_query:
			return torch.ones(len(tokens), device=device)

		# Calculate weights from each strategy
		all_weights = []
		for weighter in self.weighters:
			weights = weighter(tokens, token_indices, is_query=is_query, **kwargs)
			all_weights.append(weights)

		# Stack weights and apply strategy weighting
		stacked_weights = torch.stack(all_weights)  # [num_strategies, num_tokens]
		strategy_weights = F.softplus(self.strategy_weights).view(-1, 1)  # [num_strategies, 1]

		# Weighted sum across strategies
		combined = torch.sum(stacked_weights * strategy_weights, dim=0)

		return combined

	def batch_forward(self, batch_tokens, batch_token_indices, batch_is_query=None, **kwargs):
		"""Efficient batch implementation for combined weighting"""
		device = self.strategy_weights.device
		batch_size = len(batch_tokens)

		if batch_is_query is None:
			batch_is_query = [False] * batch_size

		# Get batch weights from each weighter
		all_weighter_results = []
		for weighter in self.weighters:
			batch_weights = weighter.batch_forward(
				batch_tokens, 
				batch_token_indices, 
				batch_is_query=batch_is_query, 
				**kwargs
			)
			all_weighter_results.append(batch_weights)

		# Compute strategy weights
		strategy_weights = F.softplus(self.strategy_weights)  # [num_strategies]

		# Combine weights for each item in batch
		combined_batch_weights = []
		for i in range(batch_size):
			# Handle weighting modes
			if self.weighting_mode == "none":
				combined_batch_weights.append(torch.ones_like(all_weighter_results[0][i]))
				continue
			elif self.weighting_mode == "query_only" and not batch_is_query[i]:
				combined_batch_weights.append(torch.ones_like(all_weighter_results[0][i]))
				continue

			# Get weights from all strategies for this batch item
			item_weights = []
			for strategy_idx, strategy_result in enumerate(all_weighter_results):
				item_weights.append(strategy_result[i] * strategy_weights[strategy_idx])

			# Sum weights across strategies
			combined = torch.zeros_like(item_weights[0])
			for w in item_weights:
				combined += w

			combined_batch_weights.append(combined)

		return combined_batch_weights

class LlamaReranker(nn.Module):
	"""
	Neural reranker using Llama embeddings from a specified layer.
	Uses a flexible late-interaction approach with customizable token weighting.
	Optimized for parallel processing on GPU.
	"""

	def __init__(
		self,
		model_name: str = "meta-llama/Llama-2-7b-hf",
		layer_idx: int = 20,
		device: Optional[Union[str, torch.device]] = None,
		dtype: Optional[torch.dtype] = None,
		max_length: int = 512,
		normalize_embeddings: bool = True,
		token_weighter: Optional[TokenWeighter] = None,
		similarity_fn: str = "cosine",
		weight_normalization: str = "linear",
		weighting_mode="full"
	):
		"""Initialize the Llama Reranker model"""
		super(LlamaReranker, self).__init__()

		# Model configuration
		self.model_name = model_name
		self.layer_idx = layer_idx
		self.max_length = max_length
		self.normalize_embeddings = normalize_embeddings
		self.weighting_mode = weighting_mode

		# Device setup
		if device is None:
			self.device = get_device()
		else:
			self.device = device if isinstance(device, torch.device) else torch.device(device)

		# Auto-select dtype if not specified
		if dtype is None:
			self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32
		else:
			self.dtype = dtype

		logger.info(f"Using device: {self.device} with dtype: {self.dtype}")

		# Token weighter setup
		self.token_weighter = token_weighter
		if self.token_weighter:
			self.token_weighter.to(self.device)

		# Similarity and weighting settings
		self.similarity_fn = similarity_fn
		self.weight_normalization = weight_normalization

		# Initialize model components
		self.tokenizer = None
		self.model = None
		self._is_loaded = False

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
		self._is_loaded = True

		# Ensure weighter is on the correct device
		if self.token_weighter:
			self.token_weighter.to(self.device)

		logger.info("Model loaded successfully.")

	def unload_model(self):
		"""Unload the model and tokenizer from memory"""
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

	def encode(self, text, remove_special_tokens=True, is_query=False):
		"""
		Encode text and extract embeddings from the specified layer.

		Args:
			text: Text to encode
			remove_special_tokens: Whether to remove special tokens from outputs
			is_query: Whether this text is a query (True) or document (False)

		Returns:
			Dictionary with embeddings, tokens, and token info
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")

		# Tokenize input
		encoding = self.tokenizer(
			text,
			padding=False,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		)

		# Move to appropriate device
		encoding = {k: v.to(self.device) for k, v in encoding.items()}

		# Forward pass to get embeddings and hidden states
		with torch.no_grad():
			outputs = self.model(**encoding, return_dict=True)

		# Extract hidden states from the specified layer
		hidden_states = outputs.hidden_states[self.layer_idx]

		# Get logits if available (for token probabilities)
		logits = getattr(outputs, 'logits', None)

		# Get token IDs and attention mask for actual sequence length
		attention_mask = encoding["attention_mask"][0]
		seq_length = attention_mask.sum().item()
		token_ids = encoding["input_ids"][0][:seq_length].tolist()

		# Get token strings
		tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

		# Get embeddings for this sequence
		text_embeddings = hidden_states[0, :seq_length].clone()

		# Calculate token probabilities if available
		token_probs = [None] * seq_length
		if logits is not None:
			batch_logits = logits[0, :seq_length-1]
			if batch_logits.size(-1) == len(self.tokenizer):
				batch_probs = F.softmax(batch_logits, dim=-1)
				for pos in range(seq_length - 1):
					next_token_id = token_ids[pos + 1]
					token_probs[pos+1] = batch_probs[pos, next_token_id].item()

		# Normalize embeddings if requested
		if self.normalize_embeddings:
			text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

		# Filter out special tokens if requested
		if remove_special_tokens:
			special_tokens = set(self.tokenizer.all_special_tokens)
			keep_mask = [token not in special_tokens for token in tokens]
			if not any(keep_mask):
				# Handle edge case of all special tokens
				filtered_embeddings = text_embeddings.new_zeros((0, text_embeddings.size(1)))
				filtered_tokens = []
				filtered_token_ids = []
				filtered_token_probs = []
				filtered_token_indices = []
			else:
				# Get indices of tokens to keep
				keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
				# Use index_select for GPU efficiency
				select_indices = torch.tensor(keep_indices, device=text_embeddings.device)
				filtered_embeddings = torch.index_select(text_embeddings, 0, select_indices)
				filtered_tokens = [tokens[i] for i in keep_indices]
				filtered_token_ids = [token_ids[i] for i in keep_indices]
				filtered_token_probs = [token_probs[i] for i in keep_indices]
				filtered_token_indices = keep_indices
		else:
			filtered_embeddings = text_embeddings
			filtered_tokens = tokens
			filtered_token_ids = token_ids
			filtered_token_probs = token_probs
			filtered_token_indices = list(range(seq_length))

		# Create result dictionary
		result = {
			"embeddings": filtered_embeddings,
			"tokens": filtered_tokens,
			"token_ids": filtered_token_ids,
			"token_indices": filtered_token_indices,
			"token_probs": filtered_token_probs,
			"is_query": is_query  # Store the is_query flag
		}

		return result

	def batch_encode(self, texts, batch_size=16, remove_special_tokens=True):
		"""
		Encode texts in smaller batches to manage memory.

		Args:
			texts: List of texts to encode
			batch_size: Size of batches to process
			remove_special_tokens: Whether to remove special tokens

		Returns:
			List of encoded text dictionaries
		"""
		if not texts:
			return []

		all_results = []

		# Process in batches
		for i in range(0, len(texts), batch_size):
			batch_texts = texts[i:i+batch_size]
			batch_results = self.encode(batch_texts, remove_special_tokens)

			if isinstance(batch_results, dict):  # Single result case
				all_results.append(batch_results)
			else:  # Multiple results
				all_results.extend(batch_results)

		return all_results

	def calculate_similarity(self, query_data, doc_data):
		"""
		Calculate similarity between query and document using weighted MaxSim approach.

		Args:
			query_data: Dictionary with query embeddings and tokens
			doc_data: Dictionary with document embeddings and tokens

		Returns:
			Similarity score between query and document
		"""
		query_embeddings = query_data["embeddings"]
		query_tokens = query_data["tokens"]
		query_token_ids = query_data["token_ids"]
		query_token_indices = query_data["token_indices"]
		query_token_probs = query_data["token_probs"]

		doc_embeddings = doc_data["embeddings"]
		doc_tokens = doc_data["tokens"]
		doc_token_ids = doc_data["token_ids"]
		doc_token_indices = doc_data["token_indices"]
		doc_token_probs = doc_data["token_probs"]

		# Handle empty embeddings case
		if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
			return torch.tensor(0.0, device=self.device)

		# Calculate token-wise similarities matrix
		if self.similarity_fn == "cosine":
			# Embeddings should already be normalized if normalize_embeddings is True
			if not self.normalize_embeddings:
				query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
				doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

			similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())

		elif self.similarity_fn == "dot":
			similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())

		elif self.similarity_fn == "euclidean":
			# Use efficient distance calculation
			distances = torch.cdist(query_embeddings, doc_embeddings, p=2)
			# Convert distance to similarity
			similarity_matrix = 1.0 / (1.0 + distances)

		else:
			raise ValueError(f"Unsupported similarity function: {self.similarity_fn}")

		# For each query token, find the most similar document token
		max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

		# Calculate weights for tokens if weighter is provided
		if self.token_weighter is not None:
			# Calculate weights for query tokens
			query_weights = self.token_weighter(
				query_tokens, 
				query_token_indices,
				token_ids=query_token_ids,
				token_probs=query_token_probs,
				is_query=True
			)

			# Get weights for corresponding document tokens
			best_doc_indices = [doc_token_indices[idx.item()] if idx.item() < len(doc_token_indices) else 0 
							  for idx in max_indices]
			best_doc_tokens = [doc_tokens[idx.item()] if idx.item() < len(doc_tokens) else "" 
							 for idx in max_indices]
			best_doc_token_ids = [doc_token_ids[idx.item()] if idx.item() < len(doc_token_ids) else 0 
								for idx in max_indices]
			best_doc_token_probs = [doc_token_probs[idx.item()] if idx.item() < len(doc_token_probs) else None 
								  for idx in max_indices]

			best_doc_weights = self.token_weighter(
				best_doc_tokens,
				best_doc_indices,
				token_ids=best_doc_token_ids,
				token_probs=best_doc_token_probs,
				is_query=False
			)

			# Apply weighting scheme based on mode
			if self.weighting_mode == "query_only":
				combined_weights = query_weights
			elif self.weighting_mode == "none":
				combined_weights = torch.ones_like(query_weights) / len(query_weights)
			else:  # "full"
				combined_weights = query_weights * best_doc_weights

			# Apply weight normalization
			if self.weight_normalization == "linear":
				weight_sum = combined_weights.sum()
				if weight_sum > 0:
					normalized_weights = combined_weights / weight_sum
				else:
					normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)

			elif self.weight_normalization == "softmax":
				# Clamp inputs to avoid numerical issues
				combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
				normalized_weights = F.softmax(combined_weights, dim=0)

			else:
				raise ValueError(f"Unsupported weight normalization: {self.weight_normalization}")

			# Calculate weighted sum of similarities
			score = (max_similarities * normalized_weights).sum()

		else:
			# No weighting - simple average
			score = max_similarities.mean()

		return score

	def batch_calculate_similarity(self, query_data_batch, doc_data_batch):
		"""
		Calculate similarity scores for batches of queries and documents

		Args:
			query_data_batch: List of dictionaries with query data
			doc_data_batch: List of dictionaries with document data

		Returns:
			Tensor with similarity scores for each query-document pair
		"""
		batch_size = len(query_data_batch)
		scores = torch.zeros(batch_size, device=self.device)

		for i in range(batch_size):
			scores[i] = self.calculate_similarity(query_data_batch[i], doc_data_batch[i])

		return scores

	def rerank(
		self,
		query: str,
		documents: List[str],
		batch_size: int = 16,
		return_scores: bool = True
	):
		"""
		Rerank a list of documents based on similarity to query.
		Optimized for parallel processing.

		Args:
			query: Query string
			documents: List of document strings to rank
			batch_size: Number of documents to process at once
			return_scores: Whether to return scores along with indices

		Returns:
			List of document indices sorted by relevance, and optionally scores
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")

		# Encode query
		query_data = self.encode(query)

		# Store scores for each document
		scores = []

		# Process documents in batches for memory efficiency
		for i in range(0, len(documents), batch_size):
			batch_docs = documents[i:i+batch_size]

			# Encode batch of documents
			doc_data_list = self.encode(batch_docs)

			# Calculate similarity score for each document
			for doc_data in doc_data_list:
				score = self.calculate_similarity(query_data, doc_data)

				# Ensure score is detached
				if isinstance(score, torch.Tensor):
					scores.append(score.detach().cpu().item())
				else:
					scores.append(score)

		# Get sorted indices (descending by score)
		sorted_indices_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
		sorted_indices = [idx for idx, _ in sorted_indices_and_scores]

		# Return indices and optionally scores
		if return_scores:
			sorted_scores = [scores[i] for i in sorted_indices]
			return sorted_indices, sorted_scores
		else:
			return sorted_indices

	def forward(self, query_data, doc_data):
		"""
		Forward pass for training - returns similarity score as tensor.

		Args:
			query_data: Dictionary with query embeddings and tokens
			doc_data: Dictionary with document embeddings and tokens

		Returns:
			Tensor containing similarity score
		"""
		return self.calculate_similarity(query_data, doc_data)

	def batch_forward(self, query_data_batch, doc_data_batch):
		"""
		Batch forward pass for training - returns similarity scores as tensor.

		Args:
			query_data_batch: List of dictionaries with query data
			doc_data_batch: List of dictionaries with document data

		Returns:
			Tensor containing similarity scores
		"""
		return self.batch_calculate_similarity(query_data_batch, doc_data_batch)

	def __enter__(self):
		"""Enable context manager usage with 'with' statement."""
		self.load_model()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Ensure model is unloaded when exiting context."""
		self.unload_model()