"""
Token Embedding Visualization with Reverse PageRank Importance
- Extracts attention patterns and embeddings in a single pass
- Computes reverse PageRank to identify information-synthesizing tokens
- Visualizes token relationships with DTW and UMAP
- Scales points by importance in visualizations
- Generates Query vs. Document plots for each document
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import pickle
import torch
import numpy as np
import tempfile
import shutil
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
# UMAP import with error handling
try:
	from umap import UMAP
except ImportError:
	try:
		from umap.umap_ import UMAP
	except ImportError:
		print("ERROR: Could not import UMAP.")
		print("Please ensure 'umap-learn' is installed: pip install umap-learn")
		exit()
warnings.filterwarnings("ignore", category=UserWarning)
# Parallel processing imports
import multiprocessing
from functools import partial

# =============================================================================
# Configuration Section
# =============================================================================
# Model Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DEVICE = "mps"  # Use CPU if parallelizing DTW heavily
DTYPE = torch.float16

# Query & Documents Configuration
QUERY_TEXT = """What similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft."""
DOCUMENT_TEXTS = [
	"""Similarity laws for stressing heated wings. it will be shown that the differential equations for a heated plate with large temperature gradient and for a similar plate at constant temperature can be made the same by a proper modification of the thickness and the loading for the isothermal plate . this fact leads to the result that the stresses in the heated plate can be calculated from measured strains on the unheated plate by a series of relations, called the /similarity laws./ The application of this analog theory to solid wings under aerodynamic heating is discussed in detail. The loading on the unheated analog wing is, however, complicated and involves the novel concept of feedback and /body force/ loading. The problem of stressing a heated box-wing structure can be solved by the same analog method and is briefly discussed.""",
	"""Similarity laws for aerothermoelastic testing. The similarity laws for aerothermoelastic testing are presented in the range. These are obtained by making nondimensional the appropriate governing equations of the individual external aerodynamic flow, heat conduction to the interior, and stress-deflection problems which make up the combined aerothermoelastic problem. For the general aerothermoelastic model, where the model is placed in a high-stagnation-temperature wind tunnel, similitude is shown to be very difficult to achieve for a scale ratio other than unity. The primary conflict occurs between the free-stream mach number reynolds number aeroelastic parameter heat conduction parameter and thermal expansion parameter. Means of dealing with this basic conflict are presented. These include (1) looking at more specialized situations, such as the behavior of wing structures and of thin solid plate lifting surfaces, and panel flutter, where the aerothermoelastic similarity parameters assume less restrictive forms, (2) the use of /incomplete aerothermoelastic/ testing in which the pressure and/or heating rates are estimated in advance and applied artificially to the model, and (3) the use of /restricted purpose/ models investigating separately one or another facet of the complete aerothermoelastic problem. Some numerical examples of modeling for the general aerothermoelastic case as well as for the specialized situations mentioned in (1) above are given. Finally, extension of the aerothermoelastic similarity laws to higher speeds and temperatures is discussed.""",
	"""Pressure measurements at supersonic speeds on three uncambered conical wings of unit aspect ratio. Pressure measurements were made at mach numbers between 1.3 and 2.8 over a range of incidences on three simple models representing thick conical uncambered wings with sharp leading edges. These tests form part of an investigation into the effects of thickness and camber on slender wings. The aspect ratio of the models was unity in each case, and the spanwise cross sections were bounded by..  The measured pressure distributions are presented, along with overall lift and drag (excluding skin friction and base drag) obtained by integration."""
]

# Layer Configuration
# Layers for attention extraction (for PageRank)
ATTENTION_LAYERS = [0, 3, 6, 9, 12, 15, 20, 25, 30]  # To calculate PageRank
# Layers for embeddings (for DTW and UMAP)
EMBEDDING_LAYERS = [6, 9, 12, 13, 14, 15]  # Used for trajectories
# Single layer for direct embedding analysis
SINGLE_LAYER_IDX = 9  # Using one of the saved layers

# Attention Aggregation Configuration
ATTENTION_CONFIG = {
	"remove_self_attention": True,      # Remove diagonal elements (self-attention) BEFORE PageRank graph creation
	"renormalize_after_removal": True,  # Renormalize rows after removing self-attention
	"use_reverse_pagerank": True,       # Use reverse (column-wise) PageRank
}

# PageRank Configuration
PAGERANK_CONFIG = {
	"alpha": 0.85,          # Damping factor
	"max_iter": 1000,       # Maximum iterations
	"tol": 1e-6,            # Convergence tolerance
	"weight": 'weight',     # Edge attribute to use as weight
	"position_bias_correction": True,   # Apply position bias correction (off by default)
	"correction_strength": 1.15          # Strength for bias correction (higher = more compensation)
}

# Token Filtering Configuration
TOKEN_FILTERING = {
	"enabled": True,                 # Enable token filtering
	"exclude_special_tokens": True,  # Exclude model special tokens  
	"exclude_spaces": True,          # Exclude space tokens
	"custom_exclude": ["<s>", "</s>", "<0x0A>", "<pad>"],  # Custom tokens to exclude
	"min_token_length": 1            # Minimum length of token to include
}

# DTW Configuration
DTW_BAND_RADIUS = None     # None = no band constraint
DTW_METRIC = "cosine"      # Distance metric
DTW_NORMALIZE = False      # Don't normalize by sequence length

# Parallelization Config
NUM_WORKERS = max(1, int(os.cpu_count() * 0.75))  # Use 75% of CPU cores, minimum 1

# UMAP Configuration
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.15
UMAP_RANDOM_STATE = 42

# Token Importance Visualization Configuration
TOKEN_IMPORTANCE = {
	"min_point_size": 30,     # Minimum point size
	"max_point_size": 400,    # Maximum point size
	"min_label_size": 6,      # Minimum font size for labels
	"max_label_size": 12,     # Maximum font size for labels
}

# Visual Configuration
VISUALIZATION = {
	"figure_size": (15, 15),
	"query_color": "red",
	"doc_colors": ["blue", "green", "purple", "orange", "brown"],
	"alpha": 0.7,
	"point_size": 30,  # Default size if no importance scores
	"show_labels": True,
	"label_size": 8,
	"label_max_tokens": 10000,
}

# Output Configuration
OUTPUT_DIR = "dtw_umap_results"
SAVEFIG_DPI = 250

# Debug Mode
DEBUG = False  # Set to False for parallel runs

# =============================================================================
# Utility Functions
# =============================================================================
def time_function(func):
	"""Decorator to time function execution"""
	def wrapper(*args, **kwargs):
		start_time = time.time()
		print(f"Starting {func.__name__}...")
		result = func(*args, **kwargs)
		end_time = time.time()
		print(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
		return result
	return wrapper

def print_debug(message):
	"""Print debug messages if DEBUG is enabled."""
	if DEBUG:
		print(f"DEBUG: {message}")

def ensure_output_directory(output_dir):
	"""Create output directory if it doesn't exist."""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"Created output directory: {output_dir}")
	return output_dir

def filter_tokens(tokens, token_info=None, attention_matrix=None, embeddings=None):
	"""
	Filter out special tokens and others per configuration.

	Args:
		tokens: List of token strings
		token_info: Optional list of token metadata
		attention_matrix: Optional attention matrix (n_tokens, n_tokens)
		embeddings: Optional embeddings matrix (n_tokens, n_features) or list

	Returns:
		Tuple of (filtered_tokens, filtered_token_info, filtered_attention_matrix, filtered_embeddings)
	"""
	if not TOKEN_FILTERING["enabled"]:
		# Return original data
		return tokens, token_info, attention_matrix, embeddings

	keep_indices = []
	for i, token in enumerate(tokens):
		# Skip special tokens
		if TOKEN_FILTERING["exclude_special_tokens"] and (
			token in TOKEN_FILTERING["custom_exclude"] or 
			(token.startswith('<') and token.endswith('>'))
		):
			continue

		# Skip space tokens
		if TOKEN_FILTERING["exclude_spaces"] and (
			token.strip() == '' or token in [' ', 'Ġ', ' ']
		):
			continue

		# Skip tokens shorter than minimum length
		if len(token.strip()) < TOKEN_FILTERING["min_token_length"]:
			continue

		keep_indices.append(i)

	# Filter tokens
	filtered_tokens = [tokens[i] for i in keep_indices]

	# Filter token_info if provided
	filtered_token_info = None
	if token_info is not None:
		filtered_token_info = [token_info[i] for i in keep_indices]

	# Filter attention matrix if provided
	filtered_attention = None
	if attention_matrix is not None:
		filtered_attention = attention_matrix[np.ix_(keep_indices, keep_indices)]

	# Filter embeddings if provided
	filtered_embeddings = None
	if embeddings is not None:
		if isinstance(embeddings, list):
			# Handle list of embeddings
			filtered_embeddings = [embeddings[i] for i in keep_indices]
		else:
			# Handle numpy array
			filtered_embeddings = embeddings[keep_indices]

	print(f"Filtered {len(tokens) - len(filtered_tokens)} tokens out of {len(tokens)} total")
	return filtered_tokens, filtered_token_info, filtered_attention, filtered_embeddings

def remove_self_attention(attention_matrix, renormalize=True):
	"""
	Remove self-attention (diagonal elements) from the attention matrix.
	Args:
		attention_matrix: NumPy array of attention weights
		renormalize: Whether to renormalize rows after removing self-attention
	Returns:
		Modified attention matrix
	"""
	if attention_matrix is None:
		return None
	# Create a copy to avoid modifying the original
	result = attention_matrix.copy()
	# Get matrix shape
	seq_len = result.shape[0]
	# Zero out diagonal (self-attention)
	np.fill_diagonal(result, 0.0)
	# Renormalize rows (from tokens) to sum to 1 again
	if renormalize and seq_len > 0: # Add check for seq_len > 0
		# Calculate row sums (shape will be (seq_len,))
		row_sums = result.sum(axis=1)
		# Create a 1D boolean mask for rows with non-zero sums
		non_zero_mask = (row_sums != 0)
		# Check if any rows need renormalization
		if np.any(non_zero_mask):
			# Select rows with non-zero sums from the result matrix
			result_subset = result[non_zero_mask]
			# Select the corresponding non-zero row sums
			sums_subset = row_sums[non_zero_mask]
			# Perform division, ensuring sums_subset is broadcast correctly (shape becomes (k, 1))
			normalized_subset = result_subset / sums_subset[:, np.newaxis]
			# Assign the normalized subset back to the original matrix using the 1D mask
			result[non_zero_mask] = normalized_subset
		# Handle rows that became all zero after removing self-attention
		zero_rows_mask = ~non_zero_mask
		if np.any(zero_rows_mask):
			print(f"Note: {np.sum(zero_rows_mask)} tokens only had self-attention (zeros after diagonal removal)")
			# Option 1: Leave them as zeros (current behavior implicitly)
			# Option 2: Distribute uniformly (excluding self) - uncomment if desired
			# if seq_len > 1:
			#     uniform_value = 1.0 / (seq_len - 1)
			#     for i in np.where(zero_rows_mask)[0]:
			#          result[i, :] = uniform_value
			#          result[i, i] = 0.0 # Ensure diagonal remains zero
	return result

# =============================================================================
# Model Loading and Data Extraction Functions
# =============================================================================
@time_function
def load_model(model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE):
	"""Load model and tokenizer."""
	print(f"Loading tokenizer: {model_name}")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if tokenizer.pad_token is None:
		print("Setting pad_token to eos_token")
		tokenizer.pad_token = tokenizer.eos_token

	print(f"Loading model: {model_name} onto device: {device}")
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=dtype,
		device_map=device
	)
	model.eval()
	print(f"Model loaded successfully to {model.device}")
	return tokenizer, model

@time_function
def extract_data_from_texts(texts, model, tokenizer, attention_layers, embedding_layers, tmp_dir):
	"""
	Extract both attention patterns and embeddings from all texts in a single pass.

	Args:
		texts: List of texts to process
		model: HuggingFace model
		tokenizer: HuggingFace tokenizer
		attention_layers: List of layer indices to extract attention from
		embedding_layers: List of layer indices to extract embeddings from
		tmp_dir: Temporary directory to save data

	Returns:
		Dictionary with metadata for all texts
	"""
	# Ensure layers are valid
	model_num_layers = model.config.num_hidden_layers
	print(f"Model has {model_num_layers} layers")

	valid_attention_layers = []
	for layer in attention_layers:
		if 0 <= layer < model_num_layers:
			valid_attention_layers.append(layer)
		else:
			print(f"Attention layer {layer} out of bounds (0-{model_num_layers-1}), skipping")

	# Hidden states tuple includes initial embedding (index 0) + layer outputs (1 to num_layers)
	valid_embedding_layers = []
	for layer in embedding_layers:
		if 0 <= layer <= model_num_layers:
			valid_embedding_layers.append(layer)
		else:
			print(f"Embedding layer {layer} out of bounds (0-{model_num_layers}), skipping")

	print(f"Will extract attention from layers: {valid_attention_layers}")
	print(f"Will extract embeddings from layers: {valid_embedding_layers}")

	all_text_data = []

	# Process each text
	for text_idx, text in enumerate(texts):
		print(f"\nProcessing text {text_idx+1}/{len(texts)}: {text[:50]}...")

		# Tokenize the text
		inputs = tokenizer(text, return_tensors="pt")
		inputs = {k: v.to(model.device) for k, v in inputs.items()}
		input_ids = inputs["input_ids"][0]
		tokens = tokenizer.convert_ids_to_tokens(input_ids)

		print(f"Text {text_idx} tokenized into {len(tokens)} tokens")

		# Create token info
		token_info = [{"text_idx": text_idx, "token_idx": i, "token": token} 
					 for i, token in enumerate(tokens)]

		# Forward pass with both attention and hidden states
		with torch.no_grad():
			outputs = model(
				**inputs,
				output_attentions=True,
				output_hidden_states=True,
				return_dict=True
			)

		# Get attention from specified layers
		all_attentions = outputs.attentions  # Tuple of attention tensors

		# Get hidden states from specified layers
		all_hidden_states = outputs.hidden_states  # Tuple of hidden state tensors

		# Create file paths for this text
		attention_dir = os.path.join(tmp_dir, f"text_{text_idx}", "attention")
		embedding_dir = os.path.join(tmp_dir, f"text_{text_idx}", "embeddings")
		os.makedirs(attention_dir, exist_ok=True)
		os.makedirs(embedding_dir, exist_ok=True)

		# Save attention matrices for each layer
		attention_files = []
		for layer_idx in valid_attention_layers:
			if layer_idx < len(all_attentions):
				# Get attention tensor and move to CPU
				att_tensor = all_attentions[layer_idx].detach().cpu()

				# Save tensor to file
				file_path = os.path.join(attention_dir, f"layer_{layer_idx}.pt")
				torch.save(att_tensor, file_path)
				attention_files.append((layer_idx, file_path))

		# Save embeddings for each layer
		embedding_files = []
		for layer_idx in valid_embedding_layers:
			if layer_idx < len(all_hidden_states):
				# Get embedding tensor and move to CPU
				# Shape is [batch_size, seq_len, hidden_dim]
				emb_tensor = all_hidden_states[layer_idx][0].detach().cpu().to(torch.float16)

				# Save tensor to file
				file_path = os.path.join(embedding_dir, f"layer_{layer_idx}.pt")
				torch.save(emb_tensor, file_path)
				embedding_files.append((layer_idx, file_path))

		# Save basic token info
		token_info_path = os.path.join(tmp_dir, f"text_{text_idx}", "token_info.pkl")
		with open(token_info_path, 'wb') as f:
			pickle.dump({
				"tokens": tokens,
				"token_ids": input_ids.tolist(),
				"token_info": token_info
			}, f)

		# Add metadata
		all_text_data.append({
			"text_idx": text_idx,
			"text": text,
			"token_count": len(tokens),
			"token_info_path": token_info_path,
			"attention_dir": attention_dir,
			"attention_files": attention_files,
			"embedding_dir": embedding_dir,
			"embedding_files": embedding_files
		})

		# Clean up memory
		del outputs, all_attentions, all_hidden_states, inputs
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
			torch.mps.empty_cache()

	return all_text_data

# =============================================================================
# PageRank Importance Calculation
# =============================================================================
@time_function
def aggregate_attention_simple(attention_files):
	"""
	Simple averaging of attention matrices across heads and layers.

	Args:
		attention_files: List of tuples (layer_idx, file_path)

	Returns:
		Aggregated attention matrix (numpy array)
	"""
	print(f"Aggregating attention across {len(attention_files)} layers using simple averaging")

	if not attention_files:
		print("No attention files to aggregate")
		return None

	# Load first file to determine dimensions
	_, first_file = attention_files[0]
	first_tensor = torch.load(first_file)
	# Attention tensor shape: [batch_size, num_heads, seq_len, seq_len]
	seq_len = first_tensor.shape[2]

	# Initialize aggregated matrix
	aggregated_attention = torch.zeros((seq_len, seq_len), dtype=torch.float32)
	total_heads = 0

	# Load and aggregate each attention layer
	for layer_idx, file_path in attention_files:
		# Load attention tensor
		att_tensor = torch.load(file_path)

		# Squeeze batch dimension if needed
		if att_tensor.dim() == 4 and att_tensor.shape[0] == 1:
			att_tensor = att_tensor.squeeze(0)

		# Average across heads
		heads_avg = att_tensor.mean(dim=0)

		# Add to aggregate
		aggregated_attention += heads_avg
		total_heads += 1

	# Normalize by number of layers
	if total_heads > 0:
		aggregated_attention /= total_heads

	# Convert to numpy for easier handling
	aggregated_attention = aggregated_attention.numpy()

	return aggregated_attention

def apply_position_bias_correction(pagerank_scores, sequence_length, strength=PAGERANK_CONFIG["correction_strength"]):
	"""
	Apply position bias correction to PageRank scores.
	Note: As implemented, this multiplies by weights increasing with position,
		  which may amplify bias rather than correct it depending on goal.

	Args:
		pagerank_scores: Original PageRank scores
		sequence_length: Length of the sequence
		strength: Strength of the correction (higher = more compensation)

	Returns:
		Corrected PageRank scores
	"""
	# Create position weights that increase later in the sequence
	position_weights = np.arange(sequence_length + 1, 1, -1) ** strength

	# Normalize weights
	position_weights = position_weights / np.sum(position_weights)

	# Element-wise multiplication
	corrected_scores = pagerank_scores * position_weights

	# Re-normalize to sum to 1
	if np.sum(corrected_scores) > 0:
		corrected_scores = corrected_scores / np.sum(corrected_scores)

	return corrected_scores

@time_function
def compute_token_importance(attention_matrix, tokens, token_info):
	"""
	Compute token importance using reverse PageRank.

	Args:
		attention_matrix: Aggregated attention matrix
		tokens: List of token strings
		token_info: List of token metadata

	Returns:
		Array of importance scores for each token
	"""
	if attention_matrix is None or attention_matrix.shape[0] == 0:
		print("Error: Empty attention matrix for PageRank calculation")
		return np.ones(len(tokens)) / len(tokens)  # Uniform scores

	# Apply self-attention removal if configured - DO THIS BEFORE TRANSPOSE
	if ATTENTION_CONFIG["remove_self_attention"]:
		print("Removing self-attention (diagonal elements) before PageRank graph creation")
		attention_matrix = remove_self_attention(
			attention_matrix, 
			renormalize=ATTENTION_CONFIG["renormalize_after_removal"]
		)

	# For reverse PageRank, we transpose the matrix
	if ATTENTION_CONFIG["use_reverse_pagerank"]:
		print("Using reverse PageRank (transposing attention matrix)")
		attention_matrix = attention_matrix.T

		# Renormalize after transposing (columns become rows)
		row_sums = attention_matrix.sum(axis=1, keepdims=True)
		row_sums[row_sums == 0] = 1.0  # Avoid division by zero
		attention_matrix = attention_matrix / row_sums

	# Create graph from attention matrix
	G = nx.DiGraph()

	# Add nodes
	seq_len = attention_matrix.shape[0]
	for i in range(seq_len):
		G.add_node(i)

	# Add weighted edges (excluding self-loops is already handled by remove_self_attention)
	for i in range(seq_len):
		for j in range(seq_len):
			# Edge from i to j with weight = attention from i to j
			weight = float(attention_matrix[i, j])
			if weight > 0:
				G.add_edge(i, j, weight=weight)

	# Check if graph has edges
	if len(G.edges()) == 0:
		print("No edges in graph after processing. Cannot compute PageRank.")
		return np.ones(seq_len) / seq_len  # Return uniform distribution as fallback

	# Compute PageRank
	try:
		pagerank_scores = nx.pagerank(
			G, 
			alpha=PAGERANK_CONFIG["alpha"],
			max_iter=PAGERANK_CONFIG["max_iter"],
			tol=PAGERANK_CONFIG["tol"],
			weight=PAGERANK_CONFIG["weight"]
		)

		# Convert dictionary to array in order
		scores = np.zeros(seq_len)
		for idx, score in pagerank_scores.items():
			scores[idx] = score

		# Apply position bias correction if enabled
		if PAGERANK_CONFIG["position_bias_correction"]:
			print(f"Applying position bias correction (strength={PAGERANK_CONFIG['correction_strength']})")
			scores = apply_position_bias_correction(
				scores, 
				seq_len, 
				PAGERANK_CONFIG["correction_strength"]
			)

		return scores

	except Exception as e:
		print(f"Error computing PageRank: {e}")
		# Try with more relaxed parameters
		try:
			print("Trying PageRank with more relaxed parameters...")
			pagerank_scores = nx.pagerank(
				G, 
				alpha=0.85,  # Default damping
				max_iter=50,  # Fewer iterations
				tol=1e-4      # More relaxed tolerance
			)

			# Convert dictionary to array
			scores = np.zeros(seq_len)
			for idx, score in pagerank_scores.items():
				scores[idx] = score

			return scores

		except Exception as e2:
			print(f"PageRank calculation failed again: {e2}")
			return np.ones(seq_len) / seq_len  # Uniform fallback

@time_function
def process_text_for_importance(text_data, output_dir):
	"""
	Process a single text to calculate token importance using reverse PageRank.

	Args:
		text_data: Dictionary with text metadata
		output_dir: Output directory for results

	Returns:
		Dictionary with token importance data
	"""
	text_idx = text_data["text_idx"]
	print(f"\nCalculating token importance for Text {text_idx}")

	# Load token info
	with open(text_data["token_info_path"], 'rb') as f:
		token_data = pickle.load(f)

	tokens = token_data["tokens"]
	token_info = token_data["token_info"]

	# Aggregate attention across layers
	aggregated_attention = aggregate_attention_simple(text_data["attention_files"])

	# Filter tokens if configured
	filtered_tokens, filtered_token_info, filtered_attention, _ = filter_tokens(
		tokens, token_info, aggregated_attention
	)

	# Compute token importance using reverse PageRank
	# Note: Self-attention removal and transpose are handled inside compute_token_importance
	importance_scores = compute_token_importance(filtered_attention, filtered_tokens, filtered_token_info)

	# Save importance scores to CSV
	token_type = "reverse" if ATTENTION_CONFIG["use_reverse_pagerank"] else "forward"
	csv_path = os.path.join(output_dir, f"text_{text_idx}_{token_type}_importance.csv")

	df = pd.DataFrame({
		'token': [info["token"] for info in filtered_token_info],
		'display_token': [t.replace('Ġ', ' ').replace(' ', ' ') for t in filtered_tokens],
		'position': [info["token_idx"] for info in filtered_token_info],
		'importance_score': importance_scores
	})

	# Sort by importance
	df = df.sort_values('importance_score', ascending=False)
	df.to_csv(csv_path, index=False)
	print(f"Saved token importance scores to {csv_path}")

	# Create visualization of top tokens
	plot_token_importance(filtered_tokens, filtered_token_info, importance_scores,
						 f"Token Importance ({token_type.title()}) - Text {text_idx}",
						 os.path.join(output_dir, f"text_{text_idx}_{token_type}_importance.png"))

	# Save detailed token data with importance
	return {
		"text_idx": text_idx,
		"filtered_tokens": filtered_tokens,
		"filtered_token_info": filtered_token_info,
		"importance_scores": importance_scores,
		"csv_path": csv_path
	}

def plot_token_importance(tokens, token_info, importance_scores, title, output_path, top_n=None):
	"""
	Plot token importance based on PageRank scores as vertical bars with tokens in original sequence.
	Args:
		tokens: List of token strings
		token_info: List of token metadata
		importance_scores: Array of token importance scores
		title: Plot title
		output_path: Output file path
		top_n: Optional limit on number of tokens to display (None = show all)
	"""
	# Check for empty input
	if not tokens or len(tokens) == 0:
		print("Warning: Empty token list provided to plot_token_importance. Cannot create plot.")
		return
	
	# Determine how many tokens to display (either all or top_n)
	num_tokens = len(tokens)
	if top_n is not None and top_n < num_tokens:
		# Take the first N tokens in their original sequence
		display_count = top_n
	else:
		display_count = num_tokens

	# Create token-position pairs (no sorting by importance)
	token_data = list(zip(
		tokens[:display_count], 
		importance_scores[:display_count], 
		[info["token_idx"] for info in token_info[:display_count]]
	))
	
	# Unzip for plotting (maintaining original sequence order)
	display_tokens, display_scores, positions = zip(*token_data)
	
	# Clean tokens for display
	display_tokens = [t.replace('Ġ', ' ').replace(' ', ' ') for t in display_tokens]
	
	# Calculate figure width based on token count
	width_per_token = 0.4  # inches per token
	min_width = 12
	max_width = 24
	fig_width = min(max_width, max(min_width, display_count * width_per_token))
	
	# Create figure with appropriate size
	plt.figure(figsize=(fig_width, 10))
	
	# Use a color gradient for bars
	colors = plt.cm.viridis(np.linspace(0, 0.8, display_count))
	
	# Create vertical bar chart (x-positions are just indices)
	x_positions = np.arange(display_count)
	bars = plt.bar(x_positions, display_scores, color=colors, width=0.7)
	
	# Add value labels above each bar
	for i, bar in enumerate(bars):
		# Only show label if bar is tall enough
		if bar.get_height() > max(display_scores) * 0.05:  # If bar > 5% of max height
			plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
					f"{display_scores[i]:.4f}", ha='center', va='bottom', fontsize=8)
	
	# Set x-axis ticks and labels
	rotation_angle = 90 if display_count > 20 else 45
	plt.xticks(x_positions, display_tokens, rotation=rotation_angle, ha='right', fontsize=10)
	
	# Add grid lines for easier comparison
	plt.grid(axis='y', linestyle='--', alpha=0.3)
	
	# Add labels and title
	plt.ylabel('Importance Score', fontsize=12)
	plt.title(title, fontsize=14)
	
	# Ensure y-axis starts at zero
	plt.ylim(bottom=0)
	
	# Add padding at top for value labels
	y_max = max(display_scores) * 1.15  # 15% padding
	plt.ylim(top=y_max)
	
	# Adjust layout and save
	plt.tight_layout()
	plt.savefig(output_path, dpi=SAVEFIG_DPI)
	plt.close()
	print(f"Saved token importance plot to {output_path}")

# =============================================================================
# DTW Implementation for Token Trajectories
# =============================================================================
def dtw_distance(sequence1, sequence2, band_radius=None, distance_metric="cosine", normalize=True,
				 token_i_global_idx=-1, token_j_global_idx=-1):
	"""
	Compute DTW distance between two token embedding trajectories.

	Args:
		sequence1: First sequence of embeddings (layers x embedding_dim) as numpy array
		sequence2: Second sequence of embeddings (layers x embedding_dim) as numpy array
		band_radius: Sakoe-Chiba band radius (None for automatic full width)
		distance_metric: Distance metric to use ("euclidean", "cosine", "manhattan")
		normalize: Whether to normalize by sequence lengths (N+M)
		token_i_global_idx: Global index of the first token (for debugging)
		token_j_global_idx: Global index of the second token (for debugging)

	Returns:
		DTW distance (float)
	"""
	# Inputs are expected to be numpy arrays already (float32)
	sequence1 = np.asarray(sequence1, dtype=np.float32)
	sequence2 = np.asarray(sequence2, dtype=np.float32)
	n = sequence1.shape[0]
	m = sequence2.shape[0]

	# Check if sequences are empty
	if n == 0 or m == 0:
		print_debug(f"Empty sequence detected for tokens {token_i_global_idx} or {token_j_global_idx}. Seq lengths: {n}, {m}. Returning max distance.")
		return 1.0 if distance_metric == "cosine" else np.inf

	# Set band radius if not provided
	if band_radius is None:
		band_radius = max(n, m)  # Effectively no band constraint

	# Define distance function based on metric
	if distance_metric == "euclidean":
		def dist_fn(x, y, i_seq, j_seq):
			if np.isnan(x).any() or np.isinf(x).any():
				print_debug(f"NaN/Inf in vector x (token {token_i_global_idx}, seq_idx {i_seq})")
				return np.inf
			if np.isnan(y).any() or np.isinf(y).any():
				print_debug(f"NaN/Inf in vector y (token {token_j_global_idx}, seq_idx {j_seq})")
				return np.inf
			diff = x - y
			dist_sq = np.dot(diff, diff)
			dist = np.sqrt(dist_sq)
			if np.isnan(dist) or np.isinf(dist):
				print_debug(f"NaN/Inf in Euclidean dist calculation for tokens {token_i_global_idx}({i_seq}) vs {token_j_global_idx}({j_seq})")
				return np.inf
			return dist
	elif distance_metric == "cosine":
		def dist_fn(x, y, i_seq, j_seq):
			# Check for NaN/Inf in inputs
			x_has_nan_inf = np.isnan(x).any() or np.isinf(x).any()
			y_has_nan_inf = np.isnan(y).any() or np.isinf(y).any()
			if x_has_nan_inf: print_debug(f"NaN/Inf in vector x (token {token_i_global_idx}, seq_idx {i_seq})")
			if y_has_nan_inf: print_debug(f"NaN/Inf in vector y (token {token_j_global_idx}, seq_idx {j_seq})")
			if x_has_nan_inf or y_has_nan_inf: return 1.0  # Max cosine distance

			# Compute cosine distance
			norm_x = np.linalg.norm(x)
			norm_y = np.linalg.norm(y)
			is_zero_x = norm_x < 1e-10
			is_zero_y = norm_y < 1e-10

			if is_zero_x and is_zero_y:
				return 0.0  # Identical
			elif is_zero_x or is_zero_y:
				return 1.0  # Maximally dissimilar

			dot_product = np.dot(x, y)
			denominator = norm_x * norm_y

			# Denominator check
			if denominator < 1e-10:
				print_debug(f"Denominator near zero ({denominator=}) for tokens {token_i_global_idx}({i_seq}) vs {token_j_global_idx}({j_seq}). norm_x={norm_x}, norm_y={norm_y}. Returning dist 1.0")
				return 1.0

			similarity = dot_product / denominator
			# Clamp similarity due to potential float inaccuracies
			similarity = np.clip(similarity, -1.0, 1.0)
			distance = 1.0 - similarity

			if np.isnan(distance):
				print(f"!!! CRITICAL: NaN detected in cosine distance *despite checks* !!!")
				print(f"  Tokens: {token_i_global_idx}({i_seq}) vs {token_j_global_idx}({j_seq})")
				print(f"  norm_x={norm_x}, norm_y={norm_y}, dot_product={dot_product}, denominator={denominator}, similarity={similarity}")
				return 1.0  # Fallback

			return distance
	elif distance_metric == "manhattan":
		def dist_fn(x, y, i_seq, j_seq):
			if np.isnan(x).any() or np.isinf(x).any():
				print_debug(f"NaN/Inf in vector x (token {token_i_global_idx}, seq_idx {i_seq})")
				return np.inf
			if np.isnan(y).any() or np.isinf(y).any():
				print_debug(f"NaN/Inf in vector y (token {token_j_global_idx}, seq_idx {j_seq})")
				return np.inf

			# Manhattan distance: sum of absolute differences
			dist = np.sum(np.abs(x - y))

			if np.isnan(dist) or np.isinf(dist):
				print_debug(f"NaN/Inf in Manhattan dist calculation for tokens {token_i_global_idx}({i_seq}) vs {token_j_global_idx}({j_seq})")
				return np.inf

			return dist
	else:
		raise ValueError(f"Unsupported distance metric: {distance_metric}")

	# Initialize DTW matrix (Cost matrix)
	dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
	dp[0, 0] = 0.0

	# Fill the dp array using dynamic programming with band constraint
	for i in range(1, n + 1):
		# Determine band boundaries for column j
		j_start = max(1, i - band_radius)
		j_end = min(m + 1, i + band_radius + 1)

		for j in range(j_start, j_end):
			# Calculate cost using the distance function
			cost = dist_fn(sequence1[i-1], sequence2[j-1], i-1, j-1)

			if not np.isfinite(cost):
				continue  # Skip update if cost is invalid

			# Get costs from possible previous steps
			prev_costs = [
				dp[i-1, j],    # Move from top
				dp[i-1, j-1],  # Move from diagonal
				dp[i, j-1]     # Move from left
			]
			min_prev_cost = np.min(prev_costs)

			# Update only if a valid path exists to this cell
			if np.isfinite(min_prev_cost):
				dp[i, j] = cost + min_prev_cost

	# Final DTW distance
	final_distance = dp[n, m]

	# Check if a valid path was found to the end
	if not np.isfinite(final_distance):
		return 1.0 if distance_metric == "cosine" else np.inf

	# Normalization (Optional)
	if normalize:
		norm_factor = n + m
		if norm_factor > 0:
			return final_distance / norm_factor
		else:
			return 0.0
	else:
		return final_distance

# --- Worker Functions for Parallel DTW ---
# Global variables for worker processes
worker_all_tokens = None
worker_dtw_band_radius = None
worker_dtw_metric = None
worker_dtw_normalize = None

def init_worker(all_tokens_data, band_radius, metric, normalize):
	"""Initializer for multiprocessing pool workers."""
	global worker_all_tokens, worker_dtw_band_radius, worker_dtw_metric, worker_dtw_normalize
	worker_all_tokens = all_tokens_data
	worker_dtw_band_radius = band_radius
	worker_dtw_metric = metric
	worker_dtw_normalize = normalize

def compute_dtw_pair(indices):
	"""Task executed by worker processes."""
	global worker_all_tokens, worker_dtw_band_radius, worker_dtw_metric, worker_dtw_normalize
	i, j = indices
	try:
		dist = dtw_distance(
			worker_all_tokens[i],
			worker_all_tokens[j],
			band_radius=worker_dtw_band_radius,
			distance_metric=worker_dtw_metric,
			normalize=worker_dtw_normalize,
			token_i_global_idx=i,
			token_j_global_idx=j
		)
		return i, j, dist
	except Exception as e:
		print(f"!!! Error in worker process for pair ({i}, {j}): {e}")
		return i, j, np.nan  # Return NaN on error to indicate failure

@time_function
def extract_token_trajectories(all_text_data, layer_indices):
	"""
	Extract token embedding trajectories for all tokens across texts.

	Args:
		all_text_data: List of text metadata dictionaries
		layer_indices: Indices of layers to use for trajectories

	Returns:
		Tuple of (token_trajectories, all_token_info)
	"""
	print(f"Extracting token trajectories for layers: {layer_indices}")

	all_token_trajectories = []
	all_token_info = []

	for text_data in all_text_data:
		text_idx = text_data["text_idx"]

		# Load token info
		with open(text_data["token_info_path"], 'rb') as f:
			token_data = pickle.load(f)

		tokens = token_data["tokens"]
		token_info = token_data["token_info"]

		# Find embedding files for requested layers
		layer_file_dict = {layer_idx: file_path for layer_idx, file_path in text_data["embedding_files"]}

		# Check if all requested layers are available
		missing_layers = [layer for layer in layer_indices if layer not in layer_file_dict]
		if missing_layers:
			print(f"Warning: Layers {missing_layers} not found for text {text_idx}. Using available layers only.")

		valid_layers = [layer for layer in layer_indices if layer in layer_file_dict]
		if not valid_layers:
			print(f"Error: No valid layers found for text {text_idx}. Skipping.")
			continue

		# Load embeddings for each valid layer
		layer_embeddings = {}
		for layer in valid_layers:
			# Load embedding tensor
			emb_tensor = torch.load(layer_file_dict[layer])
			if emb_tensor.dim() == 1:  # Handle unexpected dimensions
				emb_tensor = emb_tensor.unsqueeze(0)

			# Convert to numpy float32
			layer_embeddings[layer] = emb_tensor.numpy().astype(np.float32)

		# Create trajectory for each token
		for token_idx, token in enumerate(tokens):
			# Extract embedding for each layer for this token
			token_trajectory = []
			for layer in valid_layers:
				if token_idx < layer_embeddings[layer].shape[0]:
					token_trajectory.append(layer_embeddings[layer][token_idx])
				else:
					print(f"Warning: Token index {token_idx} out of bounds for layer {layer} embeddings with shape {layer_embeddings[layer].shape}")

			# Skip if trajectory is empty
			if not token_trajectory:
				continue

			# Add to results
			all_token_trajectories.append(np.array(token_trajectory))
			all_token_info.append(token_info[token_idx])

	print(f"Extracted trajectories for {len(all_token_trajectories)} tokens across {len(all_text_data)} texts")
	return all_token_trajectories, all_token_info

@time_function
def compute_dtw_distance_matrix(token_trajectories, token_info, metric=DTW_METRIC, normalize=DTW_NORMALIZE, num_workers=NUM_WORKERS):
	"""
	Compute pairwise DTW distances between all token trajectories using multiprocessing.

	Args:
		token_trajectories: List of token embedding trajectories (numpy arrays)
		token_info: List of token metadata
		metric: Distance metric for DTW
		normalize: Whether to normalize distances
		num_workers: Number of worker processes to use

	Returns:
		Distance matrix as numpy array
	"""
	print(f"Computing DTW distance matrix using {num_workers} workers...")

	total_tokens = len(token_trajectories)

	# Initialize distance matrix
	distance_matrix = np.full((total_tokens, total_tokens), np.nan, dtype=np.float32)
	np.fill_diagonal(distance_matrix, 0.0)

	# Prepare tasks for parallel processing
	tasks = []
	for i in range(total_tokens):
		for j in range(i + 1, total_tokens):
			tasks.append((i, j))

	total_dtw_calcs = len(tasks)
	print(f"Prepared {total_dtw_calcs} DTW calculation tasks for parallel execution.")

	# Run parallel computation
	results_count = 0
	nan_warnings = 0
	inf_warnings = 0

	# Use try-finally to ensure pool cleanup
	pool = None
	try:
		# Create pool with initializer
		pool = multiprocessing.Pool(
			processes=num_workers,
			initializer=init_worker,
			initargs=(token_trajectories, DTW_BAND_RADIUS, metric, normalize)
		)

		# Process tasks using imap_unordered for progress tracking
		chunksize = 100  # Fixed chunk size
		print(f"Starting parallel computation with chunksize={chunksize}...")

		# Wrap the pool iterator with tqdm for a progress bar
		results_iterator = pool.imap_unordered(compute_dtw_pair, tasks, chunksize=chunksize)

		for i, j, dist in tqdm(results_iterator, total=total_dtw_calcs, desc="Computing DTW distances (parallel)"):
			results_count += 1

			if np.isnan(dist):
				if nan_warnings < 20:  # Limit warnings
					print(f"Warning: Worker returned NaN for pair ({i}, {j})")
				nan_warnings += 1
			elif np.isinf(dist):
				if inf_warnings < 20:
					print(f"Warning: Worker returned Inf for pair ({i}, {j})")
				inf_warnings += 1

			# Store result symmetrically
			distance_matrix[i, j] = dist
			distance_matrix[j, i] = dist

		# Ensure all tasks are processed
		pool.close()
		pool.join()
		print("Parallel computation finished.")

	except Exception as e:
		print(f"!!! Error during parallel processing: {e}")
		import traceback
		traceback.print_exc()
		if pool:
			pool.terminate()  # Forcefully stop workers on error
	finally:
		if pool:
			pool.close()
			pool.join()  # Ensure cleanup

	# Handle NaN/Inf values
	replacement_value = 1.0 if metric == "cosine" else np.inf  # Default replacement

	# Find valid values to determine appropriate replacement
	non_diag_mask = ~np.eye(total_tokens, dtype=bool)
	finite_mask = np.isfinite(distance_matrix)
	valid_distances = distance_matrix[non_diag_mask & finite_mask]

	if metric != "cosine" and len(valid_distances) > 0:
		replacement_value = np.max(valid_distances) * 1.1
	elif metric != "cosine":
		replacement_value = 100.0  # Fallback if no valid distances

	final_nan_count = np.isnan(distance_matrix).sum()
	final_inf_count = np.isinf(distance_matrix).sum()

	if final_nan_count > 0 or final_inf_count > 0:
		print(f"\nWarning: Found {final_nan_count} NaN and {final_inf_count} Inf values in the final distance matrix.")
		print(f"Replacing them with value: {replacement_value}")
		distance_matrix = np.nan_to_num(distance_matrix, nan=replacement_value, posinf=replacement_value, neginf=replacement_value)

	return distance_matrix

# =============================================================================
# UMAP and Visualization Functions
# =============================================================================
@time_function
def run_umap(data, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
			 is_distance_matrix=False, random_state=UMAP_RANDOM_STATE):
	"""
	Run UMAP dimensionality reduction. Handles NaN/Inf in input.

	Args:
		data: Input data matrix (embeddings or distance matrix)
		n_neighbors: Number of neighbors for UMAP
		min_dist: Minimum distance parameter for UMAP
		is_distance_matrix: Whether input is a precomputed distance matrix
		random_state: Random seed for reproducibility

	Returns:
		2D UMAP embedding (numpy array)
	"""
	metric_to_use = 'precomputed' if is_distance_matrix else 'cosine'
	print(f"Running UMAP with metric='{metric_to_use}' (n_neighbors={n_neighbors}, min_dist={min_dist})...")

	# Ensure data is numpy array
	if not isinstance(data, np.ndarray):
		print("Converting input data to NumPy array.")
		data = np.array(data)

	# Check input data validity
	if data.size == 0:
		print("Error: UMAP input data is empty.")
		return np.array([]).reshape(0, 2)

	if data.ndim != 2:
		print(f"Error: UMAP input data must be 2D, but got shape {data.shape}.")
		if data.ndim == 1:
			print("Reshaping 1D input to 2D (n_samples, 1 feature).")
			data = data.reshape(-1, 1)
		else:
			return np.array([]).reshape(0, 2)

	# Handle NaN/Inf values
	nan_count = np.isnan(data).sum()
	inf_count = np.isinf(data).sum()

	if nan_count > 0 or inf_count > 0:
		print(f"Warning: Input data contains {nan_count} NaN and {inf_count} Inf values.")

		if is_distance_matrix:
			# For distance matrix, replace with a value indicating max distance
			finite_vals = data[np.isfinite(data)]
			if finite_vals.size > 0:
				max_dist = np.max(finite_vals)
				replacement = max_dist * 1.1
				print(f"Replacing NaN/Inf in distance matrix with {replacement:.4f}")
			else:
				replacement = 1.0
				print(f"No finite distances found. Replacing NaN/Inf in distance matrix with {replacement}")
			data = np.nan_to_num(data, nan=replacement, posinf=replacement, neginf=replacement)
		else:
			# For embedding features, replace with median value
			finite_vals = data[np.isfinite(data)]
			if finite_vals.size > 0:
				median_val = np.median(finite_vals)
				print(f"Replacing NaN/Inf in features with median value: {median_val:.4f}")
				replacement = median_val
			else:
				replacement = 0.0
				print(f"No finite features found. Replacing NaN/Inf in features with {replacement}")
			data = np.nan_to_num(data, nan=replacement, posinf=replacement, neginf=replacement)

		# Double-check after replacement
		if np.isnan(data).any() or np.isinf(data).any():
			print("!!! ERROR: NaN/Inf values still present after replacement. Cannot run UMAP.")
			return np.array([]).reshape(0, 2)

	# Create and run UMAP reducer
	reducer = UMAP(
		n_components=2,
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		metric=metric_to_use,
		random_state=random_state,
		low_memory=True,
		verbose=True,
		spread=1.5,
	)

	try:
		embedding = reducer.fit_transform(data)
		print(f"UMAP completed, output shape: {embedding.shape}")

		# Final check on UMAP output
		if np.isnan(embedding).any() or np.isinf(embedding).any():
			print("!!! Warning: UMAP output contains NaN or Inf values!")
			# Replace with zeros for plotting
			embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

		return embedding

	except Exception as e:
		print(f"!!! Error during UMAP fitting: {e}")
		print(f"Data shape: {data.shape}, dtype: {data.dtype}")
		print(f"Is distance matrix: {is_distance_matrix}")
		return np.array([]).reshape(0, 2)

def plot_embedding(embedding, token_info, title, output_path, token_importance=None):
	"""
	Plot UMAP embedding with color-coding for query vs document tokens.
	Size of points varies based on token importance if provided.

	Args:
		embedding: 2D UMAP embedding (numpy array)
		token_info: List of token metadata (ensure length matches embedding rows)
		title: Plot title
		output_path: Path to save the figure
		token_importance: Optional array of importance scores for scaling point sizes
	"""
	print(f"Creating visualization: {title}")
	if embedding is None or embedding.shape[0] == 0:
		print("Error: Embedding data is empty or None. Cannot plot.")
		return
	if len(token_info) != embedding.shape[0]:
		print(f"Error: Mismatch between embedding rows ({embedding.shape[0]}) and token info ({len(token_info)}). Cannot plot.")
		return

	# Check for NaN/Inf in embedding and handle if necessary
	if np.isnan(embedding).any() or np.isinf(embedding).any():
		print("Warning: NaN/Inf values found in UMAP embedding result. Replacing with 0 for plotting.")
		embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

	# Process token importance scores if provided
	min_size = TOKEN_IMPORTANCE["min_point_size"]
	max_size = TOKEN_IMPORTANCE["max_point_size"]
	min_font = TOKEN_IMPORTANCE["min_label_size"]
	max_font = TOKEN_IMPORTANCE["max_label_size"]

	# Initialize with default sizes
	point_sizes = np.ones(embedding.shape[0]) * VISUALIZATION["point_size"]
	normalized_importance = None

	if token_importance is not None and len(token_importance) == embedding.shape[0]:
		print(f"Using token importance scores to scale point sizes (min: {min_size}, max: {max_size})")

		# Normalize importance scores to [0, 1] range
		if np.max(token_importance) > np.min(token_importance):
			normalized_importance = (token_importance - np.min(token_importance)) / (np.max(token_importance) - np.min(token_importance))
		else:
			normalized_importance = np.ones_like(token_importance) * 0.5

		# Scale to desired point size range
		point_sizes = min_size + normalized_importance * (max_size - min_size)

	# Setup figure
	fig, ax = plt.subplots(figsize=VISUALIZATION["figure_size"], dpi=SAVEFIG_DPI)

	# Get unique text indices present in the current token_info subset
	text_indices = sorted(list(set(info["text_idx"] for info in token_info)))

	# Plot each text's tokens
	plotted_labels = set()
	for text_idx in text_indices:
		# Get indices within the current subset that correspond to this text_idx
		current_subset_indices = [i for i, info in enumerate(token_info) if info["text_idx"] == text_idx]
		if not current_subset_indices:  # Skip if no tokens from this text_idx in the subset
			continue

		# Get color and label
		if text_idx == 0:
			color = VISUALIZATION["query_color"]
			label = "Query"
		else:
			doc_color_idx = (text_idx - 1) % len(VISUALIZATION["doc_colors"])
			color = VISUALIZATION["doc_colors"][doc_color_idx]
			label = f"Document {text_idx}"

		# Plot points for this text_idx using the subset indices, with varying sizes
		ax.scatter(
			embedding[current_subset_indices, 0],
			embedding[current_subset_indices, 1],
			c=color,
			s=point_sizes[current_subset_indices],  # Use calculated sizes
			alpha=VISUALIZATION["alpha"],
			label=label if label not in plotted_labels else ""  # Add label only once
		)
		plotted_labels.add(label)

		# Add token labels if enabled and number of tokens is manageable
		num_tokens_in_subset = len(current_subset_indices)
		if VISUALIZATION["show_labels"] and num_tokens_in_subset <= VISUALIZATION["label_max_tokens"]:
			for subset_idx in current_subset_indices:
				token_str = token_info[subset_idx]["token"]

				# Basic cleaning for display
				token_display = token_str.replace('Ġ', '').replace(' ', ' ').strip()
				if not token_display: token_display = '_'  # Represent space/empty token

				# Adjust font size based on importance if available
				if normalized_importance is not None:
					# Scale font size between min and max points based on importance
					font_size = min_font + normalized_importance[subset_idx] * (max_font - min_font)
				else:
					font_size = VISUALIZATION["label_size"]

				ax.text(
					embedding[subset_idx, 0],
					embedding[subset_idx, 1],
					token_display,
					fontsize=font_size,
					ha='center',
					va='center',
					alpha=0.6,  # Slightly dimmer labels
					zorder=5  # Draw labels on top
				)

	# Add a note about point sizes in the title if importance scores were used
	if token_importance is not None:
		token_type = "Reverse PageRank" if ATTENTION_CONFIG["use_reverse_pagerank"] else "PageRank"
		title = f"{title}\n(Point sizes reflect token importance - {token_type})"

	# Configure plot
	ax.legend(markerscale=2)  # Make legend markers larger
	ax.set_title(title, fontsize=16)
	ax.set_xlabel("UMAP Dimension 1")
	ax.set_ylabel("UMAP Dimension 2")
	ax.grid(True, linestyle='--', alpha=0.5)  # Add subtle grid

	# Save figure
	plt.tight_layout()
	try:
		plt.savefig(output_path, dpi=SAVEFIG_DPI)
		print(f"Figure saved to: {output_path}")
	except Exception as e:
		print(f"Error saving figure to {output_path}: {e}")
	plt.close(fig)  # Close the figure to free memory

@time_function
def extract_single_layer_embeddings(all_text_data, layer_idx):
	"""
	Extract embeddings from a single layer for all tokens.

	Args:
		all_text_data: List of text metadata dictionaries
		layer_idx: Layer index to use

	Returns:
		Tuple of (embeddings matrix, token_info)
	"""
	print(f"Extracting single layer embeddings from layer {layer_idx}...")

	all_embeddings = []
	all_token_info = []

	for text_data in all_text_data:
		text_idx = text_data["text_idx"]

		# Load token info
		with open(text_data["token_info_path"], 'rb') as f:
			token_data = pickle.load(f)

		tokens = token_data["tokens"]
		token_info = token_data["token_info"]

		# Find embedding file for requested layer
		layer_file = None
		for l_idx, file_path in text_data["embedding_files"]:
			if l_idx == layer_idx:
				layer_file = file_path
				break

		if layer_file is None:
			print(f"Warning: Layer {layer_idx} not found in saved files for text {text_idx}. Saved layers: {[l_idx for l_idx, _ in text_data['embedding_files']]}. Skipping.")
			continue

		# Load embedding tensor
		emb_tensor = torch.load(layer_file)
		embeddings = emb_tensor.numpy().astype(np.float32)

		# Validate shapes
		if len(embeddings) != len(tokens):
			print(f"Warning: Mismatch between embeddings count ({len(embeddings)}) and tokens count ({len(tokens)}) for text {text_idx}")

		# Add embeddings and token info to results
		for i, (token, info) in enumerate(zip(tokens, token_info)):
			if i < len(embeddings):
				all_embeddings.append(embeddings[i])
				all_token_info.append(info)
			else:
				# This should not happen if len(embeddings) == len(tokens)
				print(f"Warning: Token index {i} out of bounds for embeddings with shape {embeddings.shape}")

	if not all_embeddings:
		print(f"Error: No embeddings found for layer {layer_idx}.")
		return np.array([]).reshape(0, 0), []

	# Convert to numpy array
	embeddings_matrix = np.array(all_embeddings)
	print(f"Extracted {len(all_embeddings)} token embeddings from layer {layer_idx}")

	return embeddings_matrix, all_token_info

# =============================================================================
# Main Analysis Functions
# =============================================================================
@time_function
def analyze_single_layer(all_text_data, importance_data_list, output_dir, layer_idx=SINGLE_LAYER_IDX):
	"""
	Analyze embeddings from a single layer with UMAP projection.
	Generates Query vs. Document plots for each document.

	Args:
		all_text_data: List of text metadata dictionaries (used for extracting embeddings)
		importance_data_list: List of token importance data dictionaries (calculated previously)
		output_dir: Output directory for visualizations
		layer_idx: Layer index to analyze

	Returns:
		List of output visualization paths
	"""
	print(f"\n--- Running Single Layer Analysis (Layer {layer_idx}) ---")

	# Extract embeddings for the layer for ALL texts
	embeddings_full, token_info_full = extract_single_layer_embeddings(all_text_data, layer_idx)

	if embeddings_full.size == 0:
		print(f"Skipping single layer analysis as no embeddings were found for layer {layer_idx}.")
		return []

	# Filter tokens for ALL texts
	tokens_full = [info["token"] for info in token_info_full]
	_, filtered_token_info_full, _, filtered_embeddings_full = filter_tokens(tokens_full, token_info_full, embeddings=embeddings_full)

	if filtered_embeddings_full is None or filtered_embeddings_full.size == 0:
		print("Skipping single layer analysis as no tokens remained after filtering.")
		return []

	# Build token importance lookup by (text_idx, token_idx) from the importance_data_list
	importance_lookup = {}
	for imp_data in importance_data_list:
		current_text_idx = imp_data["text_idx"]
		for i, (token_info_item, score) in enumerate(zip(imp_data["filtered_token_info"], imp_data["importance_scores"])):
			original_token_idx = token_info_item["token_idx"]
			importance_lookup[(current_text_idx, original_token_idx)] = score

	# Get number of documents (excluding the query)
	num_docs = len(all_text_data) - 1
	output_paths = []

	# Generate Query vs Document Plots
	# We will filter the full dataset down to query + one document for each plot
	for doc_idx in range(1, num_docs + 1):
		print(f"\nProcessing Single Layer: Query vs Document {doc_idx}")

		# Select indices corresponding to query (0) and current doc_idx from the *filtered* list
		subset_indices = [i for i, info in enumerate(filtered_token_info_full)
						  if info["text_idx"] == 0 or info["text_idx"] == doc_idx]
						  
		if not subset_indices:
			print(f"No tokens found for Query vs Document {doc_idx} after filtering. Skipping plot.")
			continue

		# Extract embeddings and info for this subset
		subset_embeddings = filtered_embeddings_full[subset_indices]
		subset_token_info = [filtered_token_info_full[i] for i in subset_indices]

		# Create importance scores array aligned with the subset tokens
		subset_importance_scores = np.zeros(len(subset_token_info))
		for i, info in enumerate(subset_token_info):
			key = (info["text_idx"], info["token_idx"])
			if key in importance_lookup:
				subset_importance_scores[i] = importance_lookup[key]
			# else: score remains 0 (or some default low value)

		# Adjust n_neighbors if needed (UMAP requires n_samples > n_neighbors)
		n_samples = subset_embeddings.shape[0]
		current_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n_samples - 1))
		if n_samples > 0 and current_n_neighbors < UMAP_N_NEIGHBORS:
			print(f"Warning: Number of samples ({n_samples}) is less than UMAP_N_NEIGHBORS ({UMAP_N_NEIGHBORS}). Adjusting n_neighbors to {current_n_neighbors}.")

		# Run UMAP on this subset
		projection = run_umap(
			subset_embeddings,
			n_neighbors=current_n_neighbors,
			min_dist=UMAP_MIN_DIST,
			is_distance_matrix=False,
			random_state=UMAP_RANDOM_STATE
		)

		if projection.size == 0:
			print(f"UMAP failed for Query vs Document {doc_idx}. Skipping plot.")
			continue

		# Plot with importance-scaled point sizes
		output_path = os.path.join(output_dir, f"single_layer_{layer_idx}_query_vs_doc{doc_idx}_umap.png")
		plot_title = f"Token Embeddings - Layer {layer_idx} - Query vs Document {doc_idx} (n={len(subset_token_info)})"
		plot_embedding(projection, subset_token_info, plot_title, output_path, token_importance=subset_importance_scores)
		output_paths.append(output_path)

	return output_paths

@time_function
def analyze_token_trajectories(all_text_data, importance_data_list, output_dir, layer_indices=EMBEDDING_LAYERS):
	"""
	Analyze token trajectories across layers using DTW and UMAP.
	Generates Query vs. Document plots for each document.

	Args:
		all_text_data: List of text metadata dictionaries (used for extracting trajectories)
		importance_data_list: List of token importance data dictionaries (calculated previously)
		output_dir: Output directory for visualizations
		layer_indices: Layer indices to use for trajectories

	Returns:
		List of output visualization paths
	"""
	print(f"\n--- Running Token Trajectory Analysis (Layers {layer_indices}) ---")

	# Extract token trajectories for ALL texts
	token_trajectories_full, token_info_full = extract_token_trajectories(all_text_data, layer_indices)

	if not token_trajectories_full:
		print("No token trajectories extracted. Skipping trajectory analysis.")
		return []

	# Filter tokens for ALL texts
	tokens_full = [info["token"] for info in token_info_full]
	_, filtered_token_info_full, _, filtered_trajectories_full = filter_tokens(tokens_full, token_info_full, embeddings=token_trajectories_full)

	if not filtered_trajectories_full:
		print("No tokens remained after filtering. Skipping trajectory analysis.")
		return []

	# Compute DTW distance matrix for ALL filtered tokens
	distance_matrix_full = compute_dtw_distance_matrix(
		filtered_trajectories_full,
		filtered_token_info_full,
		metric=DTW_METRIC,
		normalize=DTW_NORMALIZE,
		num_workers=NUM_WORKERS
	)

	if distance_matrix_full.size == 0:
		print("Failed to compute DTW distance matrix. Skipping trajectory analysis.")
		return []

	# Build token importance lookup by (text_idx, token_idx) from the importance_data_list
	importance_lookup = {}
	for imp_data in importance_data_list:
		current_text_idx = imp_data["text_idx"]
		for i, (token_info_item, score) in enumerate(zip(imp_data["filtered_token_info"], imp_data["importance_scores"])):
			original_token_idx = token_info_item["token_idx"]
			importance_lookup[(current_text_idx, original_token_idx)] = score

	# Get number of documents (excluding the query)
	num_docs = len(all_text_data) - 1
	output_paths = []

	# Generate Query vs Document Plots
	# We will filter the full distance matrix down to query + one document for each plot
	for doc_idx in range(1, num_docs + 1):
		print(f"\nProcessing Trajectory: Query vs Document {doc_idx}")

		# Select indices corresponding to query (0) and current doc_idx from the *filtered* list
		subset_indices = [i for i, info in enumerate(filtered_token_info_full)
						  if info["text_idx"] == 0 or info["text_idx"] == doc_idx]
						  
		if not subset_indices:
			print(f"No tokens found for Query vs Document {doc_idx} after filtering. Skipping plot.")
			continue

		# Extract distance submatrix and info for this subset
		subset_distance_matrix = distance_matrix_full[np.ix_(subset_indices, subset_indices)]
		subset_token_info = [filtered_token_info_full[i] for i in subset_indices]

		# Create importance scores array aligned with the subset tokens
		subset_importance_scores = np.zeros(len(subset_token_info))
		for i, info in enumerate(subset_token_info):
			key = (info["text_idx"], info["token_idx"])
			if key in importance_lookup:
				subset_importance_scores[i] = importance_lookup[key]
			# else: score remains 0

		# Adjust n_neighbors if needed (UMAP requires n_samples > n_neighbors)
		n_samples = subset_distance_matrix.shape[0]
		current_n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n_samples - 1))
		if n_samples > 0 and current_n_neighbors < UMAP_N_NEIGHBORS:
			print(f"Warning: Number of samples ({n_samples}) is less than UMAP_N_NEIGHBORS ({UMAP_N_NEIGHBORS}). Adjusting n_neighbors to {current_n_neighbors}.")

		# Run UMAP on this subset distance matrix
		projection = run_umap(
			subset_distance_matrix,
			n_neighbors=current_n_neighbors,
			min_dist=UMAP_MIN_DIST,
			is_distance_matrix=True,  # Indicate precomputed distances
			random_state=UMAP_RANDOM_STATE
		)

		if projection.size == 0:
			print(f"UMAP failed for Query vs Document {doc_idx}. Skipping plot.")
			continue

		# Plot with importance-scaled point sizes
		layer_str = "-".join(map(str, layer_indices))
		output_path = os.path.join(output_dir, f"trajectory_layers_{layer_str}_query_vs_doc{doc_idx}_umap.png")
		plot_title = f"Token Trajectories (DTW) - Layers {layer_str} - Query vs Doc {doc_idx} (n={len(subset_token_info)})"
		plot_embedding(projection, subset_token_info, plot_title, output_path, token_importance=subset_importance_scores)
		output_paths.append(output_path)

	return output_paths

@time_function
def run_analysis():
	"""
	Run the complete analysis pipeline:
	1. Extract data from model for all texts
	2. Calculate token importance with reverse PageRank
	3. Perform UMAP visualization with importance-scaled points
	"""
	print("Starting complete analysis pipeline...")
	output_dir = ensure_output_directory(OUTPUT_DIR)

	# Create temporary directory for intermediate data
	tmp_dir = tempfile.mkdtemp(prefix="token_analysis_")
	print(f"Created temporary directory: {tmp_dir}")

	try:
		# Load model
		tokenizer, model = load_model()

		# Extract data from all texts
		all_texts = [QUERY_TEXT] + DOCUMENT_TEXTS
		print(f"\n--- Extracting data from {len(all_texts)} texts ---")
		all_text_data = extract_data_from_texts(
			all_texts, 
			model, 
			tokenizer, 
			ATTENTION_LAYERS, 
			EMBEDDING_LAYERS, 
			tmp_dir
		)

		# Unload model to free memory
		print("\n--- Unloading model ---")
		model_device = model.device
		del model, tokenizer
		gc.collect()
		if torch.cuda.is_available() and model_device != torch.device("cpu"):
			torch.cuda.empty_cache()
		if DEVICE == "mps" and model_device != torch.device("cpu"):
			try:
				torch.mps.empty_cache()
			except AttributeError:
				pass

		# Calculate token importance for each text
		print("\n--- Calculating Token Importance ---")
		importance_data_list = []
		for text_data in all_text_data:
			importance_data_list.append(process_text_for_importance(text_data, output_dir))

		# Perform UMAP analysis
		visualizations = []

		# Single layer analysis
		single_layer_vis = analyze_single_layer(
			all_text_data, 
			importance_data_list, 
			output_dir, 
			layer_idx=SINGLE_LAYER_IDX
		)
		visualizations.extend(single_layer_vis)

		# Trajectory analysis
		trajectory_vis = analyze_token_trajectories(
			all_text_data, 
			importance_data_list, 
			output_dir, 
			layer_indices=EMBEDDING_LAYERS
		)
		visualizations.extend(trajectory_vis)

		# Print summary
		print("\n--- Analysis Results ---")
		print(f"Token importance calculated for {len(importance_data_list)} texts.")
		print(f"Created {len(visualizations)} visualizations:")
		for viz in visualizations:
			print(f"  - {viz}")

		return {
			"importance_data_list": importance_data_list,
			"visualizations": visualizations
		}

	finally:
		# Clean up temporary directory
		print(f"\n--- Cleaning up temporary directory ---")
		try:
			shutil.rmtree(tmp_dir)
			print(f"Removed temporary directory: {tmp_dir}")
		except Exception as e:
			print(f"Failed to remove temporary directory: {e}")

# =============================================================================
# Script Execution
# =============================================================================
if __name__ == "__main__":
	# Add guard for multiprocessing on Windows
	multiprocessing.freeze_support()

	script_start_time = time.time()

	print("="*80)
	print(" Token Embedding Visualization with Reverse PageRank Importance")
	print("="*80)

	print(f"Model: {MODEL_NAME}")
	print(f"Device: {DEVICE}")
	print(f"Analyzing {len(DOCUMENT_TEXTS) + 1} texts (1 query + {len(DOCUMENT_TEXTS)} documents)")
	print(f"Attention Layers: {ATTENTION_LAYERS} (for PageRank)")
	print(f"Embedding Layers: {EMBEDDING_LAYERS} (for trajectories)")
	print(f"Single Layer Analysis: Layer {SINGLE_LAYER_IDX}")
	print(f"Using {'reverse' if ATTENTION_CONFIG['use_reverse_pagerank'] else 'forward'} PageRank")
	print(f"Parallel DTW with {NUM_WORKERS} workers")
	print(f"Output Directory: {OUTPUT_DIR}")
	print("-"*80)

	results = run_analysis()

	script_end_time = time.time()
	print("="*80)
	print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")
	print("="*80)