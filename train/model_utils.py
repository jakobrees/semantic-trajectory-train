import os
import json
import torch
import logging
from typing import Dict, List, Optional, Union, Any
import time

logger = logging.getLogger("model_utils")

def save_checkpoint(
	model,
	step: int,
	output_dir: str,
	optimizers: Dict = None,
	losses: Dict = None,
	metrics: Dict = None
):
	"""
	Save model checkpoints to disk with separate directories for each model.

	Args:
		model: The MultiLayerReranker model
		step: Current training step
		output_dir: Base directory for outputs
		optimizers: Dict of optimizers to save
		losses: Dict of current losses
		metrics: Dict of other metrics to save
	"""
	# Base checkpoint directory
	checkpoint_base = f"{output_dir}/checkpoints"
	os.makedirs(checkpoint_base, exist_ok=True)

	# 1. Save each layer model in its own directory
	for layer_idx in model.layer_indices:
		# Create dedicated directory for this layer model
		model_dir = f"{checkpoint_base}/single_layer_{layer_idx}"
		os.makedirs(model_dir, exist_ok=True)

		# Save this layer's weighter
		layer_name = f"layer_{layer_idx}"
		save_path = f"{model_dir}/weighter_step_{step}.pt"
		torch.save(model.layer_weighters[layer_name].state_dict(), save_path)

		# Save optimizer if provided
		if optimizers and layer_name in optimizers:
			optim_path = f"{model_dir}/optimizer_step_{step}.pt"
			torch.save(optimizers[layer_name].state_dict(), optim_path)

		# Save layer-specific metadata
		metadata = {
			"layer_idx": layer_idx,
			"step": step,
			"timestamp": time.time(),
			"model_name": model.model_name,
			"layer_type": "single",
			"normalize_embeddings": model.normalize_embeddings,
			"similarity_fn": model.similarity_fn,
			"weight_normalization": model.weight_normalization,
			"weighting_mode": model.weighting_mode
		}

		# Add loss if available
		if losses and layer_name in losses:
			metadata["loss"] = losses[layer_name]

		with open(f"{model_dir}/metadata.json", "w") as f:
			json.dump(metadata, f, indent=2)

	# 2. Save DTW model separately
	dtw_dir = f"{checkpoint_base}/dtw_model"
	os.makedirs(dtw_dir, exist_ok=True)

	# Save DTW weighter
	dtw_path = f"{dtw_dir}/weighter_step_{step}.pt"
	torch.save(model.dtw_weighter.state_dict(), dtw_path)

	# Save DTW optimizer if provided
	if optimizers and "dtw" in optimizers:
		dtw_optim_path = f"{dtw_dir}/optimizer_step_{step}.pt"
		torch.save(optimizers["dtw"].state_dict(), dtw_optim_path)

	# Save DTW-specific metadata
	dtw_metadata = {
		"step": step,
		"timestamp": time.time(),
		"model_name": model.model_name,
		"dtw_layers": model.dtw_layer_indices,
		"layer_type": "dtw",
		"normalize_embeddings": model.normalize_embeddings,
		"similarity_fn": model.similarity_fn,
		"weight_normalization": model.weight_normalization
	}

	# Add loss if available
	if losses and "dtw" in losses:
		dtw_metadata["loss"] = losses["dtw"]

	with open(f"{dtw_dir}/metadata.json", "w") as f:
		json.dump(dtw_metadata, f, indent=2)

	# 3. Save a reference file at the checkpoint base level
	reference = {
		"step": step,
		"timestamp": time.time(),
		"available_models": {
			"single_layers": model.layer_indices,
			"dtw": {
				"using_layers": model.dtw_layer_indices
			}
		}
	}

	# Add metrics if provided
	if metrics:
		reference["metrics"] = metrics

	with open(f"{checkpoint_base}/checkpoint_reference_{step}.json", "w") as f:
		json.dump(reference, f, indent=2)

	logger.info(f"Saved all model checkpoints at step {step}")

def load_model_from_checkpoint(
	checkpoint_path: str,
	model_type: str = "single_layer", 
	layer_idx: Optional[int] = None,
	device = None,
	step: Optional[int] = None,
	load_optimizer: bool = False
):
	"""
	Load a specific model from saved checkpoints.

	Args:
		checkpoint_path: Base path to checkpoints directory
		model_type: 'single_layer' or 'dtw'
		layer_idx: Layer index (required for single_layer model type)
		device: Device to load model to
		step: Specific step to load (loads latest if None)
		load_optimizer: Whether to load optimizer state

	Returns:
		Tuple of (loaded_model, optimizer) if load_optimizer=True, else just model
	"""
	from reranker_model import get_device, VocabLookupWeighter

	if device is None:
		device = get_device()

	# Determine the specific checkpoint directory
	if model_type == "single_layer":
		if layer_idx is None:
			raise ValueError("layer_idx required for single_layer model type")
		model_dir = f"{checkpoint_path}/single_layer_{layer_idx}"
	elif model_type == "dtw":
		model_dir = f"{checkpoint_path}/dtw_model"
	else:
		raise ValueError(f"Unknown model_type: {model_type}")

	if not os.path.exists(model_dir):
		raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

	# Load metadata to get model configuration
	with open(f"{model_dir}/metadata.json", "r") as f:
		metadata = json.load(f)

	# Find checkpoint file
	if step is None:
		# Find the latest checkpoint
		checkpoint_files = [f for f in os.listdir(model_dir) 
						   if f.startswith("weighter_step_") and f.endswith(".pt")]
		if not checkpoint_files:
			raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

		# Sort by step number
		latest_checkpoint = sorted(
			checkpoint_files,
			key=lambda x: int(x.split("_")[2].split(".")[0])
		)[-1]

		checkpoint_path = f"{model_dir}/{latest_checkpoint}"
		step = int(latest_checkpoint.split("_")[2].split(".")[0])
	else:
		# Use specified step
		checkpoint_path = f"{model_dir}/weighter_step_{step}.pt"
		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	logger.info(f"Loading checkpoint from {checkpoint_path}")

	# Create appropriate model based on type
	if model_type == "single_layer":
		# Create a minimal model with just the needed layer
		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
		vocab_size = len(tokenizer)

		# Create appropriate weighter
		weighter = VocabLookupWeighter(vocab_size=vocab_size)

		# Load weights
		weighter.load_state_dict(torch.load(checkpoint_path, map_location=device))
		weighter.to(device)

		# Create simple reranker with just this layer
		from reranker_model import LlamaReranker
		reranker = LlamaReranker(
			model_name=metadata["model_name"],
			layer_idx=layer_idx,
			device=device,
			token_weighter=weighter,
			normalize_embeddings=metadata.get("normalize_embeddings", True),
			similarity_fn=metadata.get("similarity_fn", "cosine"),
			weight_normalization=metadata.get("weight_normalization", "linear")
		)

		# Load optimizer if requested
		if load_optimizer:
			import torch.optim as optim
			optimizer = optim.Adam(weighter.parameters())

			# Check if optimizer state exists
			optim_path = f"{model_dir}/optimizer_step_{step}.pt"
			if os.path.exists(optim_path):
				optimizer.load_state_dict(torch.load(optim_path, map_location=device))
				logger.info(f"Loaded optimizer state from {optim_path}")
			else:
				logger.warning(f"Optimizer state not found at {optim_path}")

			return reranker, optimizer

		return reranker

	elif model_type == "dtw":
		# Create a model that uses DTW similarity
		# We'll need to import our multi-layer model
		from transformers import AutoTokenizer
		from multi_layer_reranker import MultiLayerReranker

		tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
		vocab_size = len(tokenizer)

		# Create DTW weighter
		weighter = VocabLookupWeighter(vocab_size=vocab_size)

		# Load weights
		weighter.load_state_dict(torch.load(checkpoint_path, map_location=device))
		weighter.to(device)

		# Create DTW-based reranker
		reranker = MultiLayerReranker(
			model_name=metadata["model_name"],
			layer_indices=metadata["dtw_layers"],  # Only load needed layers
			dtw_layer_indices=metadata["dtw_layers"],
			device=device,
			normalize_embeddings=metadata.get("normalize_embeddings", True),
			similarity_fn=metadata.get("similarity_fn", "cosine"),
			weight_normalization=metadata.get("weight_normalization", "linear")
		)

		# Assign loaded weighter
		reranker.dtw_weighter = weighter

		# Load optimizer if requested
		if load_optimizer:
			import torch.optim as optim
			optimizer = optim.Adam(weighter.parameters())

			# Check if optimizer state exists
			optim_path = f"{model_dir}/optimizer_step_{step}.pt"
			if os.path.exists(optim_path):
				optimizer.load_state_dict(torch.load(optim_path, map_location=device))
				logger.info(f"Loaded optimizer state from {optim_path}")
			else:
				logger.warning(f"Optimizer state not found at {optim_path}")

			return reranker, optimizer

		return reranker

def plot_losses(loss_log_file, output_dir):
	"""Generate loss plots and save to disk"""
	try:
		import matplotlib
		matplotlib.use('Agg')  # Non-interactive backend
		import matplotlib.pyplot as plt
		import pandas as pd

		# Read TSV log
		df = pd.read_csv(loss_log_file, sep='\t')

		# Plot losses
		plt.figure(figsize=(12, 8))
		for col in df.columns:
			if col != 'step':
				plt.plot(df['step'], df[col], label=col)

		plt.xlabel('Step')
		plt.ylabel('Loss')
		plt.title('Training Losses')
		plt.legend()
		plt.grid(True)

		# Save to disk
		plt.savefig(f"{output_dir}/logs/loss_plot.png")
		plt.close()

		logger.info(f"Generated loss plot at {output_dir}/logs/loss_plot.png")

	except ImportError:
		logger.warning("Matplotlib or pandas not available for plotting")
	except Exception as e:
		logger.error(f"Error generating plot: {e}")

def clear_gpu_memory():
	"""Explicitly clear temporary GPU memory"""
	if torch.cuda.is_available():
		torch.cuda.empty_cache()