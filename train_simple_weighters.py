#!/usr/bin/env python3
"""Train LlamaReranker models with multiple weighting schemes on MSMARCO hard negatives"""

import gzip
import json
import logging
import os
import random
import sys
import argparse
import torch
import pickle
import gc
import csv
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from typing import Dict, List, Any, Optional, Tuple

# Setup preliminary logging
print("Initializing training script...")

try:
	from beir import LoggingHandler, util
	from beir.datasets.data_loader import GenericDataLoader

	# Import model components - verify these are in your path
	from reranker_model import LlamaReranker, PositionalWeighter, SurpriseWeighter, VocabLookupWeighter
	from model_utils import save_checkpoint, clear_gpu_memory

	# Import plotting libraries
	import matplotlib
	matplotlib.use('Agg')  # Use non-interactive backend
	import matplotlib.pyplot as plt

except ImportError as e:
	print(f"ERROR: Failed to import required modules: {e}")
	print("Make sure BEIR is installed and your model code is in the Python path")
	sys.exit(1)

# Setup logging
logging.basicConfig(
	format="%(asctime)s - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
	handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

def load_token_weights(weights_filepath: str, weight_type: str = "log_weights", fallback_weight: float = 21.5481) -> Dict:
	"""
	Load token weights from a pickle file, correctly handling the nested
	structure saved by token_freq.py.

	Args:
		weights_filepath: Path to the pickle file containing token weights.
		weight_type: Type of weight to use ('log_weights' or 'reciprocal_weights').
		fallback_weight: Default weight to use for tokens not in dictionary

	Returns:
		Dict containing the requested token weights, or an empty dict if loading fails.
	"""
	if not os.path.exists(weights_filepath):
		logger.warning(f"Error: Token weights file not found at {weights_filepath}")
		return {}

	try:
		with open(weights_filepath, 'rb') as f:
			# Load the entire data structure from the pickle file
			full_weight_data = pickle.load(f)

			# Check if the loaded data is a dictionary and contains the requested weight_type key
			if isinstance(full_weight_data, dict) and weight_type in full_weight_data:
				# Extract the specific weight dictionary
				weights_dict = full_weight_data[weight_type]

				# Validate that the extracted part is also a dictionary
				if isinstance(weights_dict, dict):
					logger.info(f"Loaded {len(weights_dict)} token weights ('{weight_type}') from {weights_filepath}")
					return weights_dict
				else:
					logger.warning(f"Error: Weight type '{weight_type}' in {weights_filepath} exists but is not a dictionary")
					return {}
			else:
				# Handle cases where the file might have a different structure or is missing the key
				if isinstance(full_weight_data, dict):
					logger.warning(f"Error: Weight file {weights_filepath} loaded, but does not contain the key '{weight_type}'. Available keys: {list(full_weight_data.keys())}")
				else:
					logger.warning(f"Error: Weight file {weights_filepath} did not load as a dictionary. Cannot extract weights.")
				return {}
	except Exception as e:
		logger.error(f"Error loading token weights from {weights_filepath}: {e}")
		import traceback
		traceback.print_exc()  # Print detailed traceback for unexpected errors
		return {}

def initialize_token_weighter_with_frequencies(
	tokenizer,
	weighter: VocabLookupWeighter,
	weights_filepath: str,
	weight_type: str = "log_weights",
	fallback_weight: float = 21.5481
):
	"""
	Initialize a VocabLookupWeighter with token frequency weights

	Args:
		tokenizer: Tokenizer to map tokens to IDs
		weighter: The VocabLookupWeighter to initialize
		weights_filepath: Path to token frequency weights file
		weight_type: Type of weight to use ('log_weights' or 'reciprocal_weights')
		fallback_weight: Default weight to use for tokens not in dictionary
	"""
	logger.info(f"Initializing token weights from {weights_filepath} using {weight_type}")

	# Load weights
	token_weights = load_token_weights(weights_filepath, weight_type, fallback_weight)
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

# New function to create and save loss plots
def plot_losses(loss_log_file, output_dir, model_names, weighter_type, weighting_mode):
	"""
	Generate and save loss plots

	Args:
		loss_log_file: Path to the loss CSV file
		output_dir: Directory to save the plots
		model_names: List of model names to plot
		weighter_type: Current weighter type
		weighting_mode: Current weighting mode
	"""
	try:
		# Create plots directory if it doesn't exist
		plots_dir = os.path.join(output_dir, "plots", f"{weighter_type}_{weighting_mode}")
		os.makedirs(plots_dir, exist_ok=True)

		# Read the loss log file
		steps = []
		losses = {model_name: [] for model_name in model_names}

		with open(loss_log_file, 'r') as f:
			reader = csv.DictReader(f)
			for row in reader:
				steps.append(int(row['step']))
				for model_name in model_names:
					if model_name in row:
						losses[model_name].append(float(row[model_name]))

		if not steps:
			logger.warning(f"No data found in loss log file: {loss_log_file}")
			return

		# Generate individual plots for each model
		for model_name in model_names:
			if not losses[model_name]:
				continue

			plt.figure(figsize=(10, 6))
			plt.plot(steps, losses[model_name], 'b-', linewidth=2)
			plt.title(f'Training Loss - {model_name} ({weighter_type}, {weighting_mode} weighting)')
			plt.xlabel('Training Steps')
			plt.ylabel('Loss')
			plt.grid(True, linestyle='--', alpha=0.7)

			# Add moving average
			window_size = min(5, len(losses[model_name]))
			if window_size > 1:
				moving_avg = []
				for i in range(len(losses[model_name])):
					start_idx = max(0, i - window_size + 1)
					moving_avg.append(sum(losses[model_name][start_idx:i+1]) / (i - start_idx + 1))
				plt.plot(steps, moving_avg, 'r-', linewidth=1.5, alpha=0.7, label='Moving Average')
				plt.legend()

			# Save the plot
			plot_file = os.path.join(plots_dir, f"{model_name}_loss.png")
			plt.savefig(plot_file, dpi=300, bbox_inches='tight')
			plt.close()

			logger.info(f"Saved loss plot for {model_name} to {plot_file}")

		# Generate combined plot for all models
		plt.figure(figsize=(12, 8))
		for model_name in model_names:
			if losses[model_name]:
				plt.plot(steps, losses[model_name], linewidth=2, label=model_name)

		plt.title(f'Training Losses - All Models ({weighter_type}, {weighting_mode} weighting)')
		plt.xlabel('Training Steps')
		plt.ylabel('Loss')
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.legend()

		# Save the combined plot
		combined_plot_file = os.path.join(plots_dir, "combined_loss.png")
		plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
		plt.close()

		logger.info(f"Saved combined loss plot to {combined_plot_file}")

	except Exception as e:
		logger.error(f"Error generating loss plots: {e}")
		import traceback
		traceback.print_exc()

# Dataset class for triplets with hard negatives
class MSMARCOHardNegativeDataset(Dataset):
	def __init__(self, queries, corpus):
		self.queries = queries
		self.queries_ids = list(queries.keys())
		self.corpus = corpus

		# Prepare the data
		for qid in self.queries:
			self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
			self.queries[qid]["harStream error: Response payload is not completed: <TransferEncodingError: 400, message='Not enough data for satisfy transfer length header.'>d_neg"] = list(self.queries[qid]["hard_neg"])
			random.shuffle(self.queries[qid]["hard_neg"])

	def __getitem__(self, idx):
		query = self.queries[self.queries_ids[idx]]
		query_text = query["query"]

		# Get a positive document
		pos_idx = random.randint(0, len(query["pos"])-1)
		pos_id = query["pos"][pos_idx]
		pos_text = self.corpus[pos_id]["text"]

		# Get a hard negative document
		neg_idx = random.randint(0, len(query["hard_neg"])-1)  
		neg_id = query["hard_neg"][neg_idx]
		neg_text = self.corpus[neg_id]["text"]

		return {
			"query": query_text, 
			"positive": pos_text, 
			"negative": neg_text
		}

	def __len__(self):
		return len(self.queries)

# Loss function for late interaction with margin
class MarginMSELoss(torch.nn.Module):
	def __init__(self, model, margin=0.05, lambda_factor=0.1):
		super(MarginMSELoss, self).__init__()
		self.model = model
		self.margin = margin
		self.lambda_factor = lambda_factor
		self.mse = torch.nn.MSELoss()

	def forward(self, query_data, pos_data, neg_data):
		"""
		Combines margin ranking loss with MSE
		- Margin: ensures pos_score > neg_score + margin
		- MSE: pushes pos_score toward 1 and neg_score toward 0
		"""
		# Get scores - query data will be marked as is_query=True
		pos_score = self.model(query_data, pos_data)
		neg_score = self.model(query_data, neg_data)

		# Margin component: max(0, margin - (pos_score - neg_score))
		margin_loss = torch.relu(self.margin - (pos_score - neg_score))

		# MSE component to push scores toward ideal values (1 for pos, 0 for neg)
		pos_mse = self.mse(pos_score, torch.tensor(1.0, device=pos_score.device))
		neg_mse = self.mse(neg_score, torch.tensor(0.0, device=neg_score.device))
		mse_loss = pos_mse + neg_mse

		# Combine losses
		total_loss = margin_loss + self.lambda_factor * mse_loss
		return total_loss

def train_with_hard_negatives(
	model_name="meta-llama/Llama-2-7b-hf",
	layer_indices=[0, 3, 6, 9, 12, 15, 18, 21],
	weighter_types=["token", "positional", "surprise"],
	weighting_modes=["full", "query_only", "none"],
	data_path="./datasets/msmarco",
	batch_size=16,
	gradient_accumulation_steps=4,
	learning_rate=1e-4,
	weight_decay=0.01,
	max_steps=10000,
	save_steps=1000,
	evaluation_steps=200,
	ce_score_margin=3.0, 
	num_negs_per_system=5,
	max_length=350,
	output_dir="./model_checkpoints/msmarco_simple_rerankers",
	token_weights_filepath=None,
	token_weight_type="log_weights",
	token_fallback_weight=21.5481,
	margin=0.05,
	lambda_factor=0.1
):
	"""Main training function for a fixed number of steps"""
	logger.info("Starting training process...")

	# Setup output directories
	os.makedirs(output_dir, exist_ok=True)

	# Create logs directory for logging losses
	logs_dir = os.path.join(output_dir, "logs")
	os.makedirs(logs_dir, exist_ok=True)

	# Load MSMARCO data
	logger.info(f"Loading MSMARCO data from {data_path}")
	corpus, queries, _ = GenericDataLoader(data_path).load(split="train")
	logger.info(f"Loaded {len(corpus)} passages and {len(queries)} queries")

	# Load hard negatives file
	triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
	if not os.path.isfile(triplets_filepath):
		logger.info(f"Hard negatives file not found, downloading...")
		url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
		util.download_url(url, triplets_filepath)

	# Process triplets to get training data
	logger.info("Processing hard negatives file...")
	train_queries = {}
	with gzip.open(triplets_filepath, "rt", encoding="utf8") as fIn:
		for line in tqdm(fIn, total=502939, desc="Loading hard negatives"):
			data = json.loads(line)

			# Get positive passages
			pos_pids = [item["pid"] for item in data["pos"]]
			pos_min_ce_score = min([item["ce-score"] for item in data["pos"]])
			ce_score_threshold = pos_min_ce_score - ce_score_margin

			# Get hard negatives
			neg_pids = set()
			for system_negs in data["neg"].values():
				negs_added = 0
				for item in system_negs:
					if item["ce-score"] > ce_score_threshold:
						continue
					pid = item["pid"]
					if pid not in neg_pids:
						neg_pids.add(pid)
						negs_added += 1
						if negs_added >= num_negs_per_system:
							break

			if len(pos_pids) > 0 and len(neg_pids) > 0:
				# Make sure query ID exists in our queries
				if data["qid"] in queries:
					train_queries[data["qid"]] = {
						"query": queries[data["qid"]],
						"pos": pos_pids,
						"hard_neg": list(neg_pids),
					}

	logger.info(f"Prepared {len(train_queries)} queries with hard negatives")

	# Initialize dataset and dataloader
	train_dataset = MSMARCOHardNegativeDataset(train_queries, corpus)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	logger.info(f"Created dataloader with {len(train_dataset)} samples")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Loop through weighter types and weighting modes
	for weighter_type in weighter_types:
		for weighting_mode in weighting_modes:
			logger.info(f"Training with weighter type: {weighter_type}, weighting mode: {weighting_mode}")

			# Initialize models for each layer
			models = {}
			tokenizer = None  # Will be set when first model is loaded

			for layer_idx in layer_indices:
				model_name_key = f"layer_{layer_idx}"

				# Create the appropriate weighter based on type
				if weighter_type == "token":
					# For token weighter, we'll need to initialize with the vocab size
					# This will be done after the model is loaded and we have access to tokenizer
					token_weighter = None  # Will create after loading the model
				elif weighter_type == "positional":
					token_weighter = PositionalWeighter(max_length=max_length, weighting_mode=weighting_mode)
				elif weighter_type == "surprise":
					token_weighter = SurpriseWeighter(weighting_mode=weighting_mode)
				else:
					raise ValueError(f"Unknown weighter type: {weighter_type}")

				# Create the reranker model
				models[model_name_key] = LlamaReranker(
					model_name=model_name,
					layer_idx=layer_idx,
					max_length=max_length,
					token_weighter=token_weighter,  # May be None for token weighter
					device=device
				)

				# Load model
				models[model_name_key].load_model()

				# If using token weighter, we need to create and initialize it after model is loaded
				if weighter_type == "token" and token_weighter is None:
					# Get tokenizer from model
					if tokenizer is None:
						tokenizer = models[model_name_key].tokenizer

					# Create token weighter with correct vocab size
					vocab_size = len(tokenizer)
					token_weighter = VocabLookupWeighter(
						vocab_size=vocab_size, 
						weighting_mode=weighting_mode
					)

					# Initialize with token frequency weights if provided
					if token_weights_filepath and os.path.exists(token_weights_filepath):
						initialize_token_weighter_with_frequencies(
							tokenizer=tokenizer,
							weighter=token_weighter,
							weights_filepath=token_weights_filepath,
							weight_type=token_weight_type,
							fallback_weight=token_fallback_weight
						)

					# Assign weighter to model
					models[model_name_key].token_weighter = token_weighter
					models[model_name_key].token_weighter.to(device)

			# Set up optimizers for each model
			optimizers = {}
			for model_key, model in models.items():
				optimizers[model_key] = torch.optim.AdamW(
					model.token_weighter.parameters(),
					lr=learning_rate,
					weight_decay=weight_decay
				)

			# Set up loss log file
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			loss_log_file = os.path.join(logs_dir, f"{weighter_type}_{weighting_mode}_loss_log_{timestamp}.csv")

			# Create CSV file with header
			with open(loss_log_file, 'w', newline='') as f:
				fieldnames = ['step'] + list(models.keys())
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()

			logger.info(f"Loss log will be saved to {loss_log_file}")

			# Training loop
			logger.info(f"Training for {max_steps} steps with weighter type: {weighter_type}, weighting mode: {weighting_mode}")
			global_step = 0
			epoch = 0

			# Initialize gradient accumulation counter
			accumulated_batches = 0

			# Zero gradients for all optimizers at the beginning
			for optimizer in optimizers.values():
				optimizer.zero_grad()

			while global_step < max_steps:
				epoch += 1  # Increment epoch at the beginning
				logger.info(f"Starting epoch {epoch}")

				# Track losses for each model
				epoch_losses = {model_key: 0.0 for model_key in models.keys()}
				steps = 0

				for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
					# Extract query and document texts
					query_texts = [item for item in batch["query"]]
					pos_texts = [item for item in batch["positive"]]
					neg_texts = [item for item in batch["negative"]]

					# Process each model
					for model_key, model in models.items():
						# Encode batch texts - mark query data as is_query=True 
						query_data_list = [model.encode(text, is_query=True) for text in query_texts]
						pos_data_list = [model.encode(text, is_query=False) for text in pos_texts]
						neg_data_list = [model.encode(text, is_query=False) for text in neg_texts]

						# Compute batch loss
						batch_loss = torch.tensor(0.0, device=device)

						for i in range(len(query_data_list)):
							# Create loss function
							loss_fn = MarginMSELoss(model, margin=margin, lambda_factor=lambda_factor)
							loss = loss_fn(query_data_list[i], pos_data_list[i], neg_data_list[i])
							batch_loss += loss

						# Average loss across batch
						batch_loss = batch_loss / len(query_data_list)

						# Scale loss for gradient accumulation
						batch_loss = batch_loss / gradient_accumulation_steps

						# Backward pass
						batch_loss.backward()

						# Track loss
						epoch_losses[model_key] += batch_loss.item() * gradient_accumulation_steps  # Scale back for logging

					steps += 1
					accumulated_batches += 1

					# Perform optimizer step after accumulating enough gradients
					if accumulated_batches % gradient_accumulation_steps == 0:
						# Clip gradients and optimize
						for model_key, model in models.items():
							torch.nn.utils.clip_grad_norm_(
								model.token_weighter.parameters(),
								max_norm=1.0
							)
							optimizers[model_key].step()
							optimizers[model_key].zero_grad()

						# Reset accumulation counter
						accumulated_batches = 0

						# Increment global step
						global_step += 1

						# Logging and evaluation
						if global_step % evaluation_steps == 0:
							# Calculate average losses
							avg_losses = {}
							log_msg = f"Step {global_step}, "

							for model_key in models.keys():
								avg_loss = epoch_losses[model_key] / steps
								avg_losses[model_key] = avg_loss
								log_msg += f"{model_key} loss: {avg_loss:.4f}, "

							logger.info(log_msg)

							# Write to CSV log file
							with open(loss_log_file, 'a', newline='') as f:
								writer = csv.DictWriter(f, fieldnames=['step'] + list(models.keys()))
								row_data = {'step': global_step}
								row_data.update(avg_losses)
								writer.writerow(row_data)

							# Generate and save loss plots
							plot_losses(loss_log_file, output_dir, list(models.keys()), weighter_type, weighting_mode)

							# Reset tracking
							epoch_losses = {model_key: 0.0 for model_key in models.keys()}
							steps = 0

						# Save checkpoints
						if global_step % save_steps == 0:
							checkpoint_path = f"{output_dir}/{weighter_type}_{weighting_mode}"
							os.makedirs(checkpoint_path, exist_ok=True)

							# Save each model checkpoint
							for model_key, model in models.items():
								model_dir = f"{checkpoint_path}/{model_key}"
								os.makedirs(model_dir, exist_ok=True)

								# Save model weights
								torch.save(
									model.token_weighter.state_dict(),
									f"{model_dir}/weighter_step_{global_step}.pt"
								)

								# Save optimizer state
								torch.save(
									optimizers[model_key].state_dict(),
									f"{model_dir}/optimizer_step_{global_step}.pt"
								)

								# Save metadata
								with open(f"{model_dir}/metadata.json", "w") as f:
									json.dump({
										"step": global_step,
										"weighter_type": weighter_type,
										"weighting_mode": weighting_mode,
										"layer_idx": int(model_key.split("_")[1]),
										"model_name": model_name,
										"timestamp": datetime.now().isoformat()
									}, f, indent=2)

						# Clean memory
						clear_gpu_memory()

					# Check if we have reached max_steps
					if global_step >= max_steps:
						break

				# If we have reached max_steps, break the outer loop as well
				if global_step >= max_steps:
					break

			# Generate final plots
			plot_losses(loss_log_file, output_dir, list(models.keys()), weighter_type, weighting_mode)

			# Save final models
			checkpoint_path = f"{output_dir}/{weighter_type}_{weighting_mode}"
			os.makedirs(checkpoint_path, exist_ok=True)

			# Save each model checkpoint
			for model_key, model in models.items():
				model_dir = f"{checkpoint_path}/{model_key}"
				os.makedirs(model_dir, exist_ok=True)

				# Save model weights
				torch.save(
					model.token_weighter.state_dict(),
					f"{model_dir}/weighter_step_{global_step}.pt"
				)

				# Save optimizer state
				torch.save(
					optimizers[model_key].state_dict(),
					f"{model_dir}/optimizer_step_{global_step}.pt"
				)

				# Save metadata
				with open(f"{model_dir}/metadata.json", "w") as f:
					json.dump({
						"step": global_step,
						"weighter_type": weighter_type,
						"weighting_mode": weighting_mode,
						"layer_idx": int(model_key.split("_")[1]),
						"model_name": model_name,
						"timestamp": datetime.now().isoformat()
					}, f, indent=2)

			# Unload all models
			for model in models.values():
				model.unload_model()

	logger.info("Training completed!")

def main():
	"""Main entry point with argument parsing"""
	parser = argparse.ArgumentParser(description="Train LlamaReranker with multiple weighting schemes")

	# Model configuration
	parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
					   help="HuggingFace model identifier")
	parser.add_argument("--layer_indices", type=int, nargs="+", 
					   default=[0, 3, 6, 9, 12, 15, 18, 21],
					   help="Layer indices to use")

	# Weighting configuration
	parser.add_argument("--weighter_types", type=str, nargs="+",
					   default=["token", "positional", "surprise"],
					   choices=["token", "positional", "surprise"],
					   help="Types of weighter to train")
					   
	parser.add_argument("--weighting_modes", type=str, nargs="+",
					   default=["full", "query_only", "none"],
					   choices=["full", "query_only", "none"],
					   help="Weighting modes to train")

	# Training configuration
	parser.add_argument("--data_path", type=str, default="./datasets/msmarco",
					   help="Path to MSMARCO dataset")
	parser.add_argument("--batch_size", type=int, default=16,
					   help="Training batch size")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
					   help="Number of batches to accumulate before optimizer step")
	parser.add_argument("--learning_rate", type=float, default=1e-4,
					   help="Learning rate")
	parser.add_argument("--weight_decay", type=float, default=0.01,
					   help="Weight decay for L2 regularization")
	parser.add_argument("--max_steps", type=int, default=10000,
					   help="Train for this many steps")
	parser.add_argument("--save_steps", type=int, default=1000,
					   help="Save checkpoint every N steps")
	parser.add_argument("--evaluation_steps", type=int, default=200,
					   help="Log evaluation metrics every N steps")

	# Hard negative settings
	parser.add_argument("--ce_score_margin", type=float, default=3.0,
					   help="Margin for cross-encoder scores when selecting negatives")
	parser.add_argument("--num_negs_per_system", type=int, default=5,
					   help="Number of negatives to use from each system")

	# Token weight initialization
	parser.add_argument("--token_weights_filepath", type=str, default=None,
					   help="Path to token frequency weights pickle file")
	parser.add_argument("--token_weight_type", type=str, default="log_weights",
					   choices=["log_weights", "reciprocal_weights"],
					   help="Type of token weight to use")
	parser.add_argument("--token_fallback_weight", type=float, default=21.5481,
					   help="Weight to use for tokens not in the weights dictionary")

	# Loss function parameters
	parser.add_argument("--margin", type=float, default=0.05,
					   help="Margin for ranking loss (should be between 0 and 1)")
	parser.add_argument("--lambda_factor", type=float, default=0.1,
					   help="Weight factor for MSE component of the loss")

	# Other settings
	parser.add_argument("--max_length", type=int, default=350,
					   help="Maximum sequence length")
	parser.add_argument("--output_dir", type=str, default="./model_checkpoints/msmarco_simple_rerankers",
					   help="Directory to save model checkpoints")

	args = parser.parse_args()
	logger.info(f"Arguments: {args}")

	# Set CUDA allocation configuration to avoid fragmentation
	if torch.cuda.is_available():
		os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
		logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid memory fragmentation")

	# Convert arguments and call train function
	train_with_hard_negatives(
		model_name=args.model_name,
		layer_indices=args.layer_indices,
		weighter_types=args.weighter_types,
		weighting_modes=args.weighting_modes,
		data_path=args.data_path,
		batch_size=args.batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		max_steps=args.max_steps,
		save_steps=args.save_steps,
		evaluation_steps=args.evaluation_steps,
		ce_score_margin=args.ce_score_margin,
		num_negs_per_system=args.num_negs_per_system,
		max_length=args.max_length,
		output_dir=args.output_dir,
		token_weights_filepath=args.token_weights_filepath,
		token_weight_type=args.token_weight_type,
		token_fallback_weight=args.token_fallback_weight,
		margin=args.margin,
		lambda_factor=args.lambda_factor
	)

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		logger.error(f"Error in main function: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)