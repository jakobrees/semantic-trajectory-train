#!/usr/bin/env python3
"""Train MultiLayerReranker using MS MARCO with hard negatives from BEIR"""

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
from typing import Dict, List, Any, Optional

# Setup preliminary logging
print("Initializing training script...")

try:
	from beir import LoggingHandler, util
	from beir.datasets.data_loader import GenericDataLoader
	
	# Import model components - verify these are in your path
	from multi_layer_reranker import MultiLayerReranker
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

def initialize_weighter_with_token_frequencies(
	model: MultiLayerReranker,
	weights_filepath: str,
	weight_type: str = "log_weights",
	fallback_weight: float = 21.5481,
	model_types: List[str] = ["dtw"]
):
	"""
	Initialize model weighters with token frequency weights
	
	Args:
		model: MultiLayerReranker model
		weights_filepath: Path to token frequency weights file
		weight_type: Type of weight to use ('log_weights' or 'reciprocal_weights')
		fallback_weight: Default weight to use for tokens not in dictionary
		model_types: Which model types to initialize (e.g., ["dtw", "layer_6"])
	"""
	logger.info(f"Initializing token weights from {weights_filepath} using {weight_type}")
	
	# Load weights
	token_weights = load_token_weights(weights_filepath, weight_type, fallback_weight)
	if not token_weights:
		logger.warning("Failed to load token weights, using default initialization")
		return
	
	# Get tokenizer vocabulary to map token IDs
	tokenizer = model.tokenizer
	
	# Initialize each weighter
	for model_type in model_types:
		logger.info(f"Initializing weights for {model_type} model")
		
		if model_type == "dtw":
			weighter = model.dtw_weighter
		elif model_type.startswith("layer_"):
			layer_idx = int(model_type.split("_")[1])
			weighter = model.layer_weighters[model_type]
		else:
			logger.warning(f"Unknown model type: {model_type}, skipping")
			continue
		
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
		
		logger.info(f"Initialized {tokens_initialized} token weights for {model_type} " 
				   f"({tokens_missed} tokens used fallback weight {fallback_weight})")

# New function to create and save loss plots
def plot_losses(loss_log_file, output_dir, model_types, weighting_mode):
	"""
	Generate and save loss plots
	
	Args:
		loss_log_file: Path to the loss CSV file
		output_dir: Directory to save the plots
		model_types: List of model types to plot
		weighting_mode: Current weighting mode
	"""
	try:
		# Create plots directory if it doesn't exist
		plots_dir = os.path.join(output_dir, "plots", weighting_mode)
		os.makedirs(plots_dir, exist_ok=True)
		
		# Read the loss log file
		steps = []
		losses = {model_type: [] for model_type in model_types}
		
		with open(loss_log_file, 'r') as f:
			reader = csv.DictReader(f)
			for row in reader:
				steps.append(int(row['step']))
				for model_type in model_types:
					if model_type in row:
						losses[model_type].append(float(row[model_type]))
		
		if not steps:
			logger.warning(f"No data found in loss log file: {loss_log_file}")
			return
		
		# Generate individual plots for each model type
		for model_type in model_types:
			if not losses[model_type]:
				continue
				
			plt.figure(figsize=(10, 6))
			plt.plot(steps, losses[model_type], 'b-', linewidth=2)
			plt.title(f'Training Loss - {model_type} ({weighting_mode} weighting)')
			plt.xlabel('Training Steps')
			plt.ylabel('Loss')
			plt.grid(True, linestyle='--', alpha=0.7)
			
			# Add moving average
			window_size = min(5, len(losses[model_type]))
			if window_size > 1:
				moving_avg = []
				for i in range(len(losses[model_type])):
					start_idx = max(0, i - window_size + 1)
					moving_avg.append(sum(losses[model_type][start_idx:i+1]) / (i - start_idx + 1))
				plt.plot(steps, moving_avg, 'r-', linewidth=1.5, alpha=0.7, label='Moving Average')
				plt.legend()
			
			# Save the plot
			plot_file = os.path.join(plots_dir, f"{model_type}_loss.png")
			plt.savefig(plot_file, dpi=300, bbox_inches='tight')
			plt.close()
			
			logger.info(f"Saved loss plot for {model_type} to {plot_file}")
		
		# Generate combined plot for all model types
		plt.figure(figsize=(12, 8))
		for model_type in model_types:
			if losses[model_type]:
				plt.plot(steps, losses[model_type], linewidth=2, label=model_type)
		
		plt.title(f'Training Losses - All Models ({weighting_mode} weighting)')
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
			self.queries[qid]["hard_neg"] = list(self.queries[qid]["hard_neg"])
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
	def __init__(self, model, model_type="dtw", margin=0.10, lambda_factor=0.1):
		super(MarginMSELoss, self).__init__()
		self.model = model
		self.model_type = model_type
		self.margin = margin
		self.lambda_factor = lambda_factor
		self.mse = torch.nn.MSELoss()
	
	def forward(self, query_data, pos_data, neg_data):
		"""
		Combines margin ranking loss with MSE
		- Margin: ensures pos_score > neg_score + margin
		- MSE: pushes pos_score toward 1 and neg_score toward 0
		"""
		# Get scores (forward returns a dict, with the key being model_type)
		pos_scores = self.model.forward(query_data, pos_data, model_type=self.model_type)
		pos_score = pos_scores[self.model_type]
		
		neg_scores = self.model.forward(query_data, neg_data, model_type=self.model_type)
		neg_score = neg_scores[self.model_type]
		
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
	dtw_layer_indices=[6, 9, 12, 15],
	train_model_types=["dtw", "layer_6", "layer_12", "layer_18"],
	weighting_modes=["full"],
	data_path="./datasets/msmarco",
	batch_size=16,
	gradient_accumulation_steps=4,  # Accumulate gradients for 4 batches
	learning_rate=1e-4,
	weight_decay=0.01,
	max_steps=10000,
	save_steps=1000,
	evaluation_steps=200,
	ce_score_margin=3.0, 
	num_negs_per_system=5,
	max_length=350,
	output_dir="./model_checkpoints/msmarco_hard_negs",
	token_weights_filepath=None,
	token_weight_type="log_weights",
	token_fallback_weight=21.5481,
	margin=0.10,
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
	
	# Train different model types with different weighting modes
	for weighting_mode in weighting_modes:
		logger.info(f"Training with weighting mode: {weighting_mode}")
		
		# Initialize model with this weighting mode
		model = MultiLayerReranker(
			model_name=model_name,
			layer_indices=layer_indices,
			dtw_layer_indices=dtw_layer_indices,
			max_length=max_length,
			weighting_mode=weighting_mode  # IMPORTANT: Pass the weighting mode to the model
		)
		logger.info(f"Initialized model with layers: {layer_indices} and weighting mode: {weighting_mode}")
		
		# Load the model
		logger.info("Loading model into memory...")
		model.load_model()
		
		# Initialize token weights if a weights file is provided
		if token_weights_filepath and os.path.exists(token_weights_filepath):
			logger.info(f"Initializing token weights from {token_weights_filepath}")
			initialize_weighter_with_token_frequencies(
				model=model,
				weights_filepath=token_weights_filepath,
				weight_type=token_weight_type,
				fallback_weight=token_fallback_weight,
				model_types=train_model_types
			)
		
		# Set up optimizers for each model type
		optimizers = {}
		for model_type in train_model_types:
			if model_type == "dtw":
				logger.info(f"Setting up optimizer for DTW model")
				optimizers[model_type] = torch.optim.AdamW(
					model.dtw_weighter.parameters(),
					lr=learning_rate,
					weight_decay=weight_decay
				)
			elif model_type.startswith("layer_"):
				layer_idx = int(model_type.split("_")[1])
				logger.info(f"Setting up optimizer for layer {layer_idx} model")
				optimizers[model_type] = torch.optim.AdamW(
					model.layer_weighters[f"layer_{layer_idx}"].parameters(),
					lr=learning_rate,
					weight_decay=weight_decay
				)
		
		# Set up loss log file with weighting mode in the name
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		loss_log_file = os.path.join(logs_dir, f"{weighting_mode}_loss_log_{timestamp}.csv")
		
		# Create CSV file with header
		with open(loss_log_file, 'w', newline='') as f:
			fieldnames = ['step'] + train_model_types
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
		
		logger.info(f"Loss log will be saved to {loss_log_file}")
		
		# Initialize loss history for plotting
		all_steps = []
		all_losses = {model_type: [] for model_type in train_model_types}
		
		# Training loop
		logger.info(f"Training for {max_steps} steps with weighting mode: {weighting_mode}")
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
			
			# Track losses for each model type
			epoch_losses = {model_type: 0.0 for model_type in train_model_types}
			steps = 0
			
			for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
				# Extract query and document texts
				query_texts = [item for item in batch["query"]]
				pos_texts = [item for item in batch["positive"]]
				neg_texts = [item for item in batch["negative"]]
				
				# Encode all texts
				query_data_list = model.encode_multi_layer(query_texts)
				pos_data_list = model.encode_multi_layer(pos_texts)
				neg_data_list = model.encode_multi_layer(neg_texts)
				
				# For each model type, compute loss and update
				for model_type in train_model_types:
					# Compute losses for the batch
					batch_loss = torch.tensor(0.0, device=model.device)
					
					for i in range(len(query_data_list)):
						# Create loss function for this model type
						loss_fn = MarginMSELoss(model, model_type=model_type, margin=margin, lambda_factor=lambda_factor)
						loss = loss_fn(query_data_list[i], pos_data_list[i], neg_data_list[i])
						batch_loss += loss
					
					# Average loss across batch
					batch_loss = batch_loss / len(query_data_list)
					
					# Scale loss for gradient accumulation
					batch_loss = batch_loss / gradient_accumulation_steps
					
					# Backward pass
					batch_loss.backward()
					
					# Track loss
					epoch_losses[model_type] += batch_loss.item() * gradient_accumulation_steps  # Scale back for logging
				
				steps += 1
				accumulated_batches += 1
				
				# Perform optimizer step after accumulating enough gradients
				if accumulated_batches % gradient_accumulation_steps == 0:
					# Clip gradients and optimize
					for model_type in train_model_types:
						torch.nn.utils.clip_grad_norm_(
							model.dtw_weighter.parameters() if model_type == "dtw" 
							else model.layer_weighters[model_type].parameters(),
							max_norm=1.0
						)
						optimizers[model_type].step()
						optimizers[model_type].zero_grad()
					
					# Reset accumulation counter
					accumulated_batches = 0
					
					# Increment global step
					global_step += 1
					
					# Logging and evaluation
					if global_step % evaluation_steps == 0:
						# Calculate average losses
						avg_losses = {}
						log_msg = f"Step {global_step}, "
						
						for model_type in train_model_types:
							avg_loss = epoch_losses[model_type] / steps
							avg_losses[model_type] = avg_loss
							log_msg += f"{model_type} loss: {avg_loss:.4f}, "
							
							# Store for plotting
							all_losses[model_type].append(avg_loss)
						
						all_steps.append(global_step)
						logger.info(log_msg)
						
						# Write to CSV log file
						with open(loss_log_file, 'a', newline='') as f:
							writer = csv.DictWriter(f, fieldnames=['step'] + train_model_types)
							row_data = {'step': global_step}
							row_data.update(avg_losses)
							writer.writerow(row_data)
						
						# Generate and save loss plots
						plot_losses(loss_log_file, output_dir, train_model_types, weighting_mode)
						
						# Reset tracking
						epoch_losses = {model_type: 0.0 for model_type in train_model_types}
						steps = 0
					
					# Save checkpoints
					if global_step % save_steps == 0:
						checkpoint_path = f"{output_dir}/{weighting_mode}"
						os.makedirs(checkpoint_path, exist_ok=True)
						
						# Prepare losses dict for saving
						losses = {}
						for model_type in train_model_types:
							losses[model_type] = epoch_losses[model_type] / steps if steps > 0 else 0.0
						
						# Save model checkpoints
						save_checkpoint(
							model,
							global_step,
							checkpoint_path,
							optimizers=optimizers,
							losses=losses
						)
					
					# Clean memory
					clear_gpu_memory()
				
				# Check if we have reached max_steps
				if global_step >= max_steps:
					break
			
			# If we have reached max_steps, break the outer loop as well
			if global_step >= max_steps:
				break
		
		# Generate final plots
		plot_losses(loss_log_file, output_dir, train_model_types, weighting_mode)
		
		# Save final model for this weighting mode
		checkpoint_path = f"{output_dir}/{weighting_mode}"
		os.makedirs(checkpoint_path, exist_ok=True)
		
		# Prepare final losses
		final_losses = {}
		for model_type in train_model_types:
			final_losses[model_type] = epoch_losses[model_type] / steps if steps > 0 else 0.0
		
		# Save final checkpoints
		save_checkpoint(
			model,
			global_step,
			checkpoint_path,
			optimizers=optimizers,
			losses=final_losses
		)
		
		# Unload model before next weighting mode
		model.unload_model()
		
	logger.info("Training completed!")

def main():
	"""Main entry point with argument parsing"""
	parser = argparse.ArgumentParser(description="Train MultiLayerReranker with MSMARCO hard negatives")
	
	# Model configuration
	parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
					   help="HuggingFace model identifier")
	parser.add_argument("--layer_indices", type=int, nargs="+", 
					   default=[0, 3, 6, 9, 12, 15, 18, 21],
					   help="Layer indices to use")
	parser.add_argument("--dtw_layer_indices", type=int, nargs="+", 
					   default=[6, 9, 12, 15],
					   help="Layer indices to use for DTW")
	
	# Training configuration
	parser.add_argument("--train_model_types", type=str, nargs="+", 
					   default=["dtw", "layer_6", "layer_12", "layer_18"],
					   help="Which model types to train")
	parser.add_argument("--weighting_modes", type=str, nargs="+",
					   default=["full"],
					   choices=["full", "query_only", "none"],
					   help="Token weighting modes to train")
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
	
	# Loss function parameters
	parser.add_argument("--margin", type=float, default=0.10,
					   help="Margin for ranking loss (should be between 0 and 1)")
	parser.add_argument("--lambda_factor", type=float, default=0.1,
					   help="Weight factor for MSE component of the loss")
	
	# Token weight initialization
	parser.add_argument("--token_weights_filepath", type=str, default=None,
					   help="Path to token frequency weights pickle file")
	parser.add_argument("--token_weight_type", type=str, default="log_weights",
					   choices=["log_weights", "reciprocal_weights"],
					   help="Type of token weight to use")
	parser.add_argument("--token_fallback_weight", type=float, default=21.5481,
					   help="Weight to use for tokens not in the weights dictionary")
	
	# Other settings
	parser.add_argument("--max_length", type=int, default=350,
					   help="Maximum sequence length")
	parser.add_argument("--output_dir", type=str, default="./model_checkpoints/msmarco_hard_negs",
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
		dtw_layer_indices=args.dtw_layer_indices,
		train_model_types=args.train_model_types,
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