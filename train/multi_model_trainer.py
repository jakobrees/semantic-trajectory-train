import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional
import logging
import time
import os
from tqdm import tqdm
import json
import argparse
import sys
import gc

# Import our modules
from msmarco_loader import MSMARCOTripleLoader
from multi_layer_reranker import MultiLayerReranker
from model_utils import save_checkpoint, plot_losses, clear_gpu_memory

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler("training.log")
	]
)
logger = logging.getLogger('multi_model_trainer')

# Process all layers in parallel
def process_all_layers_in_parallel(reranker, layer_indices, query_pos_data, query_neg_data, 
								pos_data_list, neg_data_list, target, loss_fn, 
								gradient_accumulation_steps, optimizers):
	"""
	Process all layers in parallel for better GPU utilization.
	Returns the total loss and individual layer losses.
	"""
	batch_losses = {}
	total_loss = 0.0

	# Create a list to hold all layer computations
	layer_computations = []

	# First, compute all layer similarities in parallel
	for layer_idx in layer_indices:
		layer_name = f"layer_{layer_idx}"

		# Get similarity scores in batch
		pos_scores = reranker.batch_calculate_layer_similarity(query_pos_data, pos_data_list, layer_idx)
		neg_scores = reranker.batch_calculate_layer_similarity(query_neg_data, neg_data_list, layer_idx)

		# Compute loss
		loss = loss_fn(pos_scores, neg_scores, target)

		# Scale loss for gradient accumulation
		scaled_loss = loss / gradient_accumulation_steps

		# Store for later
		layer_computations.append({
			'layer_name': layer_name,
			'layer_idx': layer_idx,
			'pos_scores': pos_scores,
			'neg_scores': neg_scores,
			'loss': loss,
			'scaled_loss': scaled_loss
		})

	# Now process the gradients sequentially (can't easily avoid this)
	for comp in layer_computations:
		layer_name = comp['layer_name']
		scaled_loss = comp['scaled_loss']
		loss = comp['loss']

		# Check for invalid loss
		if not torch.isfinite(scaled_loss):
			logger.warning(
				f"Non-finite loss detected for {layer_name}: {loss.item()}. "
				f"Skipping update."
			)
			continue

		# Backward pass (retain graph for other layers and DTW)
		scaled_loss.backward(retain_graph=True)

		# Track loss
		batch_losses[layer_name] = loss.item()
		total_loss += loss.item()

	# Process DTW trajectories in parallel
	pos_dtw_scores = reranker.batch_calculate_dtw_similarity(query_pos_data, pos_data_list)
	neg_dtw_scores = reranker.batch_calculate_dtw_similarity(query_neg_data, neg_data_list)

	# Compute DTW loss
	dtw_loss = loss_fn(pos_dtw_scores, neg_dtw_scores, target)

	# Scale loss for gradient accumulation
	dtw_scaled_loss = dtw_loss / gradient_accumulation_steps

	# Check for invalid loss
	if not torch.isfinite(dtw_scaled_loss):
		logger.warning(
			f"Non-finite loss detected for DTW: {dtw_loss.item()}. "
			f"Skipping update."
		)
	else:
		# Backward pass (no retain_graph needed for last model)
		dtw_scaled_loss.backward()

		# Track loss
		batch_losses["dtw"] = dtw_loss.item()
		total_loss += dtw_loss.item()

	return total_loss, batch_losses

def train_multi_layer_reranker(
	model_name: str = "meta-llama/Llama-2-7b-hf",
	layer_indices: List[int] = [0, 3, 6, 9, 12, 15, 18, 21],
	dtw_layer_indices: List[int] = [6, 9, 12, 15],
	triples_path: str = "./triples.train.small.tsv",
	learning_rate: float = 1e-4,
	weight_decay: float = 1e-5,
	batch_size: int = 16,
	gradient_accumulation_steps: int = 1,
	steps: int = 1000,
	eval_steps: int = 25,
	logging_steps: int = 10,
	save_steps: int = 100,
	output_dir: str = "./model_checkpoints",
	resume_from: Optional[str] = None,
	resume_step: Optional[int] = None,
	memory_cleanup_steps: int = 10
):
	"""
	Train multiple reranker models simultaneously with optimized batch processing.

	Args:
		model_name: HuggingFace model identifier
		layer_indices: Which layers to extract embeddings from
		dtw_layer_indices: Which layers to use for DTW trajectories
		triples_path: Path to MS MARCO triples file
		learning_rate: Learning rate for optimizers
		batch_size: External batch size for training data
		gradient_accumulation_steps: Number of batches to accumulate before updating weights
		steps: Number of training steps
		eval_steps: How often to evaluate and log metrics
		logging_steps: How often to log basic progress
		save_steps: How often to save checkpoints
		output_dir: Where to save outputs
		resume_from: Directory to resume training from
		resume_step: Step to resume from
		memory_cleanup_steps: How often to explicitly clean up memory
	"""
	# Calculate effective batch size
	effective_batch_size = batch_size * gradient_accumulation_steps
	logger.info(f"Training with batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
	logger.info(f"Effective batch size: {effective_batch_size}")

	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(f"{output_dir}/logs", exist_ok=True)

	# Initialize model
	reranker = MultiLayerReranker(
		model_name=model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices
	)
	device = reranker.device
	logger.info(f"Using device: {device} with dtype: {reranker.dtype}")

	# Setup loss log file
	loss_log_file = f"{output_dir}/logs/loss_log.tsv"

	# Create header for TSV loss log if not resuming
	if not resume_from:
		with open(loss_log_file, "w") as f:
			header = ["step"] + [f"layer_{idx}" for idx in layer_indices] + ["dtw"]
			f.write("\t".join(header) + "\n")

	# Load model
	reranker.load_model()

	# Initialize data loader
	data_loader = MSMARCOTripleLoader(
		triples_path=triples_path,
		additional_negatives_per_query=3,
		batch_size=batch_size
	)

	# Create optimizers for each model
	optimizers = {}

	# Layer models
	for layer_idx in layer_indices:
		model_name = f"layer_{layer_idx}"
		optimizers[model_name] = optim.AdamW(
			reranker.layer_weighters[model_name].parameters(), 
			lr=learning_rate,
			weight_decay=weight_decay
		)

	# DTW model
	optimizers["dtw"] = optim.Adam(
		reranker.dtw_weighter.parameters(), 
		lr=learning_rate,
		weight_decay=weight_decay
	)

	# Resume from checkpoint if specified
	start_step = 0
	if resume_from and resume_step:
		logger.info(f"Resuming training from {resume_from} at step {resume_step}")

		# Load layer weighters
		for layer_idx in layer_indices:
			layer_name = f"layer_{layer_idx}"
			checkpoint_path = f"{resume_from}/single_layer_{layer_idx}/weighter_step_{resume_step}.pt"
			optimizer_path = f"{resume_from}/single_layer_{layer_idx}/optimizer_step_{resume_step}.pt"

			if os.path.exists(checkpoint_path):
				reranker.layer_weighters[layer_name].load_state_dict(
					torch.load(checkpoint_path, map_location=device)
				)
				logger.info(f"Loaded {layer_name} weighter from {checkpoint_path}")

				if os.path.exists(optimizer_path):
					optimizers[layer_name].load_state_dict(
						torch.load(optimizer_path, map_location=device)
					)
					logger.info(f"Loaded {layer_name} optimizer from {optimizer_path}")
			else:
				logger.warning(f"Checkpoint not found for {layer_name}: {checkpoint_path}")

		# Load DTW weighter
		dtw_checkpoint_path = f"{resume_from}/dtw_model/weighter_step_{resume_step}.pt"
		dtw_optimizer_path = f"{resume_from}/dtw_model/optimizer_step_{resume_step}.pt"

		if os.path.exists(dtw_checkpoint_path):
			reranker.dtw_weighter.load_state_dict(
				torch.load(dtw_checkpoint_path, map_location=device)
			)
			logger.info(f"Loaded DTW weighter from {dtw_checkpoint_path}")

			if os.path.exists(dtw_optimizer_path):
				optimizers["dtw"].load_state_dict(
					torch.load(dtw_optimizer_path, map_location=device)
				)
				logger.info(f"Loaded DTW optimizer from {dtw_optimizer_path}")
		else:
			logger.warning(f"Checkpoint not found for DTW: {dtw_checkpoint_path}")

		start_step = resume_step

	# Use MarginRankingLoss
	loss_fn = nn.MarginRankingLoss(margin=0.2)

	# Training loop
	step = start_step
	losses = {name: 0.0 for name in optimizers.keys()}
	total_loss = 0.0

	# Training loop
	logger.info(f"Starting training from step {start_step}")
	progress_bar = tqdm(total=steps, initial=start_step, desc="Training")
	start_time = time.time()

	# Get data iterator
	data_iterator = data_loader.stream_training_data()

	# Reset gradient accumulation counter
	accumulated_batches = 0

	# Zero gradients for all optimizers at the beginning
	for optimizer in optimizers.values():
		optimizer.zero_grad()

	while step < steps:
		# Collect batch data
		try:
			batch = next(data_iterator)
		except StopIteration:
			# Restart iterator if we reach the end of the dataset
			logger.info("Reached end of dataset, restarting")
			data_iterator = data_loader.stream_training_data()
			batch = next(data_iterator)

		# Process batch data for efficient GPU utilization
		query_texts = []
		pos_docs = []
		neg_docs = []

		# Extract triplets from batch and organize them
		for query, passage, label in batch:
			# Group by query
			if label == 1:  # Positive
				query_texts.append(query)
				pos_docs.append(passage)
			else:  # Negative
				query_texts.append(query)
				neg_docs.append(passage)

		# Ensure we have a balanced set of positives and negatives
		min_size = min(len(pos_docs), len(neg_docs))
		if min_size == 0:
			continue  # Skip if we don't have both positives and negatives

		# Truncate to same length (balanced triplets)
		query_texts = query_texts[:min_size*2]  # Same query appears for both pos and neg
		pos_docs = pos_docs[:min_size]
		neg_docs = neg_docs[:min_size]

		# Batch encode all texts at once for maximum efficiency
		# First, prepare query pairs (each query appears twice - once for pos, once for neg)
		query_texts_paired = []
		for query in query_texts[:min_size]:  # Just take unique queries
			query_texts_paired.extend([query, query])  # Each query appears twice

		# Encode queries in one batch
		query_data_list = reranker.encode_multi_layer(query_texts_paired)

		# Encode positive and negative docs in one batch
		doc_batch = pos_docs + neg_docs
		doc_data_list = reranker.encode_multi_layer(doc_batch)

		# Split into positive and negative doc data
		pos_data_list = doc_data_list[:min_size]
		neg_data_list = doc_data_list[min_size:]

		# Pair the query data correctly
		query_pos_data = query_data_list[:min_size]
		query_neg_data = query_data_list[min_size:]

		# Prepare target tensor (all ones)
		target = torch.ones(min_size, device=device)

		# Process all layers and DTW in parallel
		batch_total_loss, batch_losses = process_all_layers_in_parallel(
			reranker, layer_indices, query_pos_data, query_neg_data, 
			pos_data_list, neg_data_list, target, loss_fn, 
			gradient_accumulation_steps, optimizers
		)

		# Add batch losses to accumulated losses
		for name, loss_val in batch_losses.items():
			losses[name] += loss_val
		total_loss += batch_total_loss

		# Increment accumulation counter
		accumulated_batches += 1

		# Apply gradients and optimize after enough batches are accumulated
		if accumulated_batches >= gradient_accumulation_steps:
			# Apply gradient clipping and optimizer step for all models
			for layer_idx in layer_indices:
				layer_name = f"layer_{layer_idx}"
				torch.nn.utils.clip_grad_norm_(
					reranker.layer_weighters[layer_name].parameters(), 
					max_norm=1.0
				)
				optimizers[layer_name].step()
				optimizers[layer_name].zero_grad()

			# Apply for DTW model
			torch.nn.utils.clip_grad_norm_(
				reranker.dtw_weighter.parameters(), 
				max_norm=1.0
			)
			optimizers["dtw"].step()
			optimizers["dtw"].zero_grad()

			# Reset accumulation counter
			accumulated_batches = 0

			# Update step count and progress bar
			step += 1
			progress_bar.update(1)

			# Basic logging
			if step % logging_steps == 0:
				elapsed = time.time() - start_time
				examples_per_sec = (logging_steps * effective_batch_size) / elapsed
				avg_loss = total_loss / (logging_steps * (len(layer_indices) + 1))

				logger.info(
					f"Step {step}/{steps} | "
					f"Loss: {avg_loss:.4f} | "
					f"Examples/sec: {examples_per_sec:.1f}"
				)

				# Reset for next logging period
				start_time = time.time()
				total_loss = 0.0

			# Evaluation and detailed logging
			if step % eval_steps == 0:
				# Log losses
				avg_losses = {name: losses[name] / eval_steps for name in losses}

				log_msg = f"Step {step} evaluation:"
				for name, loss_val in avg_losses.items():
					log_msg += f" {name}: {loss_val:.4f}"
				logger.info(log_msg)

				# Log memory usage
				if torch.cuda.is_available():
					allocated = torch.cuda.memory_allocated() / 1e9
					reserved = torch.cuda.memory_reserved() / 1e9
					logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

				# Write to TSV log file
				with open(loss_log_file, "a") as f:
					values = [str(step)] + [str(avg_losses[f"layer_{idx}"]) 
										  for idx in layer_indices] + [str(avg_losses["dtw"])]
					f.write("\t".join(values) + "\n")

				# Generate plot periodically
				if step % (eval_steps * 4) == 0:
					plot_losses(loss_log_file, output_dir)

				# Reset loss accumulators
				losses = {name: 0.0 for name in optimizers.keys()}

			# Save checkpoint
			if step % save_steps == 0:
				save_checkpoint(
					reranker, 
					step, 
					output_dir,
					optimizers=optimizers,
					losses=avg_losses if step % eval_steps == 0 else None
				)

			# Memory cleanup
			if step % memory_cleanup_steps == 0:
				clear_gpu_memory()

			# Check if we've reached max steps
			if step >= steps:
				break

	# Close progress bar
	progress_bar.close()

	# Save final models and metrics
	logger.info(f"Training complete. Saving final models at step {step}")
	save_checkpoint(
		reranker, 
		step, 
		output_dir,
		optimizers=optimizers
	)

	# Generate final loss plot
	plot_losses(loss_log_file, output_dir)

	# Unload model to free memory
	reranker.unload_model()

	logger.info(f"Training completed after {step} steps")

	return reranker