#!/usr/bin/env python
# evaluate_weighting_modes.py
"""
Evaluation script that tests multiple weighting strategies with a single model forward pass
Example usage:
  python evaluate_weighting_modes.py \
	--dataset quora \
	--split test \
	--initial_ranking_file ./quora_bm25_results.tsv \
	--checkpoint_dir ./model_checkpoints/checkpoints \
	--weighting_modes uniform learned combined \
	--log_weights_file ./token_stats/token_weights.pkl \
	--output_dir ./results/weighting_comparison
"""

import argparse
import logging
import os
import torch
import json
import pickle
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from datetime import timedelta
import time

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Setup logging
logging.basicConfig(
	format="%(asctime)s - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
	handlers=[LoggingHandler()],
)

logger = logging.getLogger(__name__)

def get_device():
	"""Get the appropriate device for computation"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		return torch.device("mps")
	else:
		return torch.device("cpu")

def print_gpu_utilization():
	"""Print current GPU memory usage."""
	if torch.cuda.is_available():
		total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
		reserved = torch.cuda.memory_reserved() / 1e9
		allocated = torch.cuda.memory_allocated() / 1e9
		free = total_memory - reserved
		logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free (Total: {total_memory:.2f}GB)")
	else:
		logger.info("No GPU available")

def load_initial_rankings(file_path: str) -> Dict[str, Dict[str, float]]:
	"""Load pre-computed initial rankings from a file"""
	results = {}
	try:
		with open(file_path, 'r') as f:
			for line in tqdm(f, desc="Loading initial rankings"):
				parts = line.strip().split()
				if len(parts) >= 3:
					qid, pid = parts[0], parts[1]
					# Handle different ranking file formats
					if len(parts) >= 4:  # If there's a score field
						score = float(parts[3])
					else:  # Otherwise use 1/rank as score
						rank = int(parts[2])
						score = 1000.0 - rank  # Higher is better
					if qid not in results:
						results[qid] = {}
					results[qid][pid] = score
	except Exception as e:
		logger.error(f"Error loading initial rankings: {e}")
		raise
	logger.info(f"Loaded rankings for {len(results)} queries")
	return results

def save_results_file(results: Dict[str, Dict[str, float]], file_path: str):
	"""Save results in TREC format"""
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	with open(file_path, 'w') as f:
		for query_id, doc_scores in results.items():
			# Sort documents by score
			sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
			for rank, (doc_id, score) in enumerate(sorted_docs, 1):
				f.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")
	logger.info(f"Saved results for {len(results)} queries to {file_path}")

def check_available_models(checkpoint_dir: str, layer_indices: List[int]) -> List[str]:
	"""
	Check which models are available in the checkpoint directory
	Args:
		checkpoint_dir: Directory containing model checkpoints
		layer_indices: List of layer indices to check
	Returns:
		List of available model types ("dtw", "layer_X", etc.)
	"""
	available_models = []
	
	logger.info(f"Checking for models in: {checkpoint_dir}")
	
	# First verify the checkpoint_dir exists
	if not os.path.exists(checkpoint_dir):
		logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
		return []
	
	# List directories to understand structure
	subdir_names = [d for d in os.listdir(checkpoint_dir) 
				   if os.path.isdir(os.path.join(checkpoint_dir, d))]
	logger.info(f"Found subdirectories: {subdir_names}")
	
	# Check for DTW model
	dtw_dir = os.path.join(checkpoint_dir, "dtw_model")
	if os.path.exists(dtw_dir):
		# Look for any .pt files (more flexible)
		pt_files = [f for f in os.listdir(dtw_dir) if f.endswith('.pt')]
		weighter_files = [f for f in pt_files if 'weighter' in f]
		
		if weighter_files:
			logger.info(f"Found DTW checkpoint files: {weighter_files[:3]}... (total: {len(weighter_files)})")
			available_models.append("dtw")
		elif pt_files:
			logger.info(f"Found DTW .pt files (non-standard naming): {pt_files[:3]}... (total: {len(pt_files)})")
			available_models.append("dtw")
		else:
			logger.warning(f"DTW model directory exists but contains no checkpoint files")
	else:
		logger.warning(f"DTW model directory not found at {dtw_dir}")
	
	# Check for layer models
	for layer_idx in layer_indices:
		layer_dir = os.path.join(checkpoint_dir, f"single_layer_{layer_idx}")
		if os.path.exists(layer_dir):
			# Look for any .pt files (more flexible)
			pt_files = [f for f in os.listdir(layer_dir) if f.endswith('.pt')]
			weighter_files = [f for f in pt_files if 'weighter' in f]
			
			if weighter_files:
				logger.info(f"Found layer_{layer_idx} checkpoint files: {weighter_files[:3]}... (total: {len(weighter_files)})")
				available_models.append(f"layer_{layer_idx}")
			elif pt_files:
				logger.info(f"Found layer_{layer_idx} .pt files (non-standard naming): {pt_files[:3]}... (total: {len(pt_files)})")
				available_models.append(f"layer_{layer_idx}")
			else:
				logger.warning(f"Layer {layer_idx} directory exists but contains no checkpoint files")
	
	if not available_models:
		logger.error(f"No valid models found in {checkpoint_dir}")
		# List all subdirectories to help diagnose issues
		if os.path.exists(checkpoint_dir):
			dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
			files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
			logger.info(f"Subdirectories found in checkpoint dir: {dirs}")
			logger.info(f"Files found in checkpoint dir: {files}")
	else:
		logger.info(f"Successfully found {len(available_models)} models: {available_models}")
	
	return available_models

def load_model(checkpoint_dir, model_name, layer_indices, dtw_layer_indices, weighter_step=None):
	"""Load the MultiLayerReranker model with trained weights"""
	try:
		from multi_layer_reranker import MultiLayerReranker
	except ImportError:
		logger.error("Failed to import MultiLayerReranker. Make sure the model code is in your Python path.")
		return None, []
	
	# First check which models are available
	available_model_types = check_available_models(checkpoint_dir, layer_indices)
	if not available_model_types:
		logger.error(f"No valid models with checkpoints found in {checkpoint_dir}")
		return None, []
	
	logger.info(f"Found {len(available_model_types)} available models: {', '.join(available_model_types)}")
	
	device = get_device()
	logger.info(f"Using device: {device}")
	
	# Initialize the model
	model = MultiLayerReranker(
		model_name=model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		device=device
	)
	
	# Load model into memory
	model.load_model()
	print_gpu_utilization()
	
	# Load weights for each model
	for model_type in available_model_types:
		if model_type == "dtw":
			model_dir = os.path.join(checkpoint_dir, "dtw_model")
			weighter = model.dtw_weighter
		elif model_type.startswith("layer_"):
			layer_idx = int(model_type.split("_")[1])
			model_dir = os.path.join(checkpoint_dir, f"single_layer_{layer_idx}")
			weighter = model.layer_weighters[model_type]
		else:
			continue
		
		# Find checkpoint files
		weighter_files = sorted(
			[f for f in os.listdir(model_dir) if f.startswith('weighter_step_') and f.endswith('.pt')],
			key=lambda x: int(x.split('_')[2].split('.')[0])
		)
		
		if not weighter_files:
			logger.warning(f"No weighter files found in {model_dir}, skipping {model_type}")
			continue
		
		# Select specific step or use latest
		if weighter_step is not None:
			matching_files = [f for f in weighter_files if f'weighter_step_{weighter_step}.pt' == f]
			if matching_files:
				checkpoint_file = os.path.join(model_dir, matching_files[0])
				step = weighter_step
			else:
				logger.warning(f"No checkpoint for step {weighter_step} found, using latest instead")
				checkpoint_file = os.path.join(model_dir, weighter_files[-1])  # Use latest
				step = int(weighter_files[-1].split('_')[2].split('.')[0])
		else:
			# Use latest
			checkpoint_file = os.path.join(model_dir, weighter_files[-1])
			step = int(weighter_files[-1].split('_')[2].split('.')[0])
		
		logger.info(f"Loading {model_type} weights from {checkpoint_file} (step {step})")
		
		try:
			# Load weights - handle different state dict formats
			state_dict = torch.load(checkpoint_file, map_location=device)
			
			if isinstance(state_dict, dict):
				if 'token_weights' in state_dict:
					# Direct weights format - just set the weights
					weighter.token_weights.data = state_dict['token_weights']
					logger.info(f"Loaded {model_type} token weights directly (shape: {state_dict['token_weights'].shape})")
				else:
					# Try to load using standard load_state_dict
					weighter.load_state_dict(state_dict)
					logger.info(f"Loaded {model_type} weights using standard load_state_dict")
			else:
				logger.error(f"Unexpected checkpoint format for {model_type}: {type(state_dict)}")
				continue
				
			logger.info(f"Successfully loaded {model_type} weighter from step {step}")
		except Exception as e:
			logger.error(f"Error loading {model_type} weights: {e}")
			if model_type in available_model_types:
				available_model_types.remove(model_type)
	
	model.eval()  # Set to evaluation mode
	print_gpu_utilization()  # Check GPU usage after loading
	
	if not available_model_types:
		logger.error("No models could be loaded successfully")
		return None, []
	
	return model, available_model_types

def initialize_with_log_weights(model, log_weights_file, fallback_weight=21.5481):
	"""Initialize token weights with log frequency weights"""
	if not os.path.exists(log_weights_file):
		logger.error(f"Log weights file not found: {log_weights_file}")
		return False
		
	logger.info(f"Initializing token weights from {log_weights_file}")
	
	# Initialize all weighters
	for layer_key in model.layer_weighters:
		model.initialize_token_weighter_with_frequencies(
			model.layer_weighters[layer_key],
			log_weights_file,
			weight_type="log_weights",
			fallback_weight=fallback_weight
		)
		
	# Initialize DTW weighter
	model.initialize_token_weighter_with_frequencies(
		model.dtw_weighter,
		log_weights_file,
		weight_type="log_weights",
		fallback_weight=fallback_weight
	)
	
	return True

def rerank_with_all_weighting_modes(
	model,
	available_model_types: List[str],
	corpus: Dict[str, Dict[str, str]],
	queries: Dict[str, str],
	results: Dict[str, Dict[str, float]],
	weighting_modes: List[str],
	top_k: int = 100,
	batch_size: int = 32,
	max_queries: Optional[int] = None,
	query_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
	"""
	Rerank documents using different weighting modes in a single pass
	
	Args:
		model: Loaded MultiLayerReranker model
		available_model_types: List of model types ("dtw", "layer_X")
		corpus: Dictionary mapping doc_id to document data
		queries: Dictionary mapping query_id to query text
		results: Initial retrieval results {query_id: {doc_id: score}}
		weighting_modes: List of weighting modes to evaluate
		top_k: How many documents to rerank per query
		batch_size: Batch size for document encoding
		max_queries: Maximum number of queries to process
		query_ids: List of specific query IDs to process
	
	Returns:
		Nested dictionary: {weighting_mode: {model_type: {query_id: {doc_id: score}}}}
	"""
	# Select queries to process
	if query_ids is not None:
		# Filter for valid query IDs
		valid_qids = set(queries.keys()) & set(results.keys())
		query_ids = [qid for qid in query_ids if qid in valid_qids]
		if not query_ids:
			logger.warning("None of the specified query IDs were found in the dataset")
			# Return empty results
			empty_results = {}
			for mode in weighting_modes:
				empty_results[mode] = {}
				for model_type in available_model_types:
					empty_results[mode][model_type] = {}
			return empty_results
		limited_queries = {qid: queries[qid] for qid in query_ids}
		logger.info(f"Processing {len(limited_queries)} specified queries")
	elif max_queries is not None and max_queries < len(results):
		# Randomly select queries
		logger.info(f"Limiting to {max_queries} of {len(results)} queries")
		query_ids = list(results.keys())
		if max_queries < len(query_ids):
			query_ids = random.sample(query_ids, max_queries)
		limited_queries = {qid: queries[qid] for qid in query_ids}
	else:
		# Process all queries
		limited_queries = queries
		query_ids = list(limited_queries.keys())
	
	# Filter query_ids to only those in results
	query_ids = [qid for qid in query_ids if qid in results]
	
	# Initialize nested results dictionaries for all weighting modes
	all_results = {}
	for mode in weighting_modes:
		all_results[mode] = {}
		for model_type in available_model_types:
			all_results[mode][model_type] = {}
	
	logger.info(f"Reranking top-{top_k} documents for {len(query_ids)} queries")
	logger.info(f"Models: {', '.join(available_model_types)}")
	logger.info(f"Weighting modes: {', '.join(weighting_modes)}")
	
	# If no queries to process, return empty results
	if not query_ids:
		logger.warning("No queries to process, returning empty results")
		return all_results
	
	# Start timing
	start_time = time.time()
	
	# Process each query
	for idx, query_id in enumerate(tqdm(query_ids, desc="Reranking queries")):
		query_text = limited_queries[query_id]
		
		# Get top-k doc_ids from initial retrieval
		sorted_docs = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)[:top_k]
		candidate_doc_ids = [doc_id for doc_id, _ in sorted_docs]
		
		# Get document texts
		candidate_docs = []
		valid_doc_ids = []
		for doc_id in candidate_doc_ids:
			if doc_id in corpus:
				# Handle both text-only and title+text formats
				if "title" in corpus[doc_id] and corpus[doc_id]["title"]:
					doc_text = f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}"
				else:
					doc_text = corpus[doc_id]["text"]
				candidate_docs.append(doc_text)
				valid_doc_ids.append(doc_id)
		
		if not candidate_docs:
			# No valid documents found for this query
			for mode in weighting_modes:
				for model_type in available_model_types:
					all_results[mode][model_type][query_id] = {}
			continue
		
		# Encode query once
		query_data = model.encode_multi_layer(query_text, is_query=True)
		
		# Process documents in batches
		doc_scores_by_mode_and_model = {
			mode: {model_type: {} for model_type in available_model_types}
			for mode in weighting_modes
		}
		
		for i in range(0, len(candidate_docs), batch_size):
			batch_docs = candidate_docs[i:i+batch_size]
			batch_doc_ids = valid_doc_ids[i:i+batch_size]
			
			# Encode batch of documents (only once regardless of weighting mode)
			doc_data_list = model.encode_multi_layer(batch_docs, is_query=False)
			
			# For each document, evaluate with all models and weighting modes
			for j, doc_data in enumerate(doc_data_list):
				doc_id = batch_doc_ids[j]
				
				# Process each weighting mode
				for weighting_mode in weighting_modes:
					# Get all model scores in a single pass
					scores = model.forward(query_data, doc_data, model_type="all", weighting_mode=weighting_mode)
					
					# Store scores for each model type
					for model_type, score in scores.items():
						if model_type in available_model_types:
							if isinstance(score, torch.Tensor):
								score_val = score.detach().cpu().item()
							else:
								score_val = score
							doc_scores_by_mode_and_model[weighting_mode][model_type][doc_id] = score_val
		
		# Store results for this query
		for weighting_mode in weighting_modes:
			for model_type in available_model_types:
				all_results[weighting_mode][model_type][query_id] = doc_scores_by_mode_and_model[weighting_mode][model_type]
		
		# Print progress and GPU usage periodically
		if (idx + 1) % 10 == 0:
			print_gpu_utilization()
			
			# Calculate ETA
			elapsed = time.time() - start_time
			queries_per_sec = (idx + 1) / elapsed
			remaining_queries = len(query_ids) - (idx + 1)
			eta_seconds = remaining_queries / queries_per_sec if queries_per_sec > 0 else 0
			
			logger.info(f"Processed {idx+1}/{len(query_ids)} queries "
						 f"({queries_per_sec:.2f} queries/sec, "
						 f"ETA: {timedelta(seconds=int(eta_seconds))})")
	
	# Final timing
	total_time = time.time() - start_time
	if len(query_ids) > 0:
		avg_queries_per_sec = len(query_ids)/total_time
	else:
		avg_queries_per_sec = 0.0
	logger.info(f"Reranking completed in {timedelta(seconds=int(total_time))}, "
				 f"average {avg_queries_per_sec:.2f} queries/sec")
	
	return all_results

def evaluate_and_save_results(
	all_results: Dict,
	evaluator: EvaluateRetrieval,
	initial_results: Dict,
	qrels: Dict,
	output_dir: str,
	dataset_name: str,
	save_runs: bool = True,
	save_metrics: bool = True
):
	"""
	Evaluate results for all weighting modes and models, save runs and metrics
	
	Args:
		all_results: Nested dictionary of reranking results
		evaluator: BEIR evaluator object
		initial_results: Initial retrieval results for comparison
		qrels: Relevance judgments
		output_dir: Directory to save results
		dataset_name: Name of the dataset
		save_runs: Whether to save run files
		save_metrics: Whether to save metrics files
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Check if there are any overlapping queries between results and qrels
	valid_qrels = {qid: rels for qid, rels in qrels.items() if qid in initial_results}
	if not valid_qrels:
		logger.error("No overlap between query IDs in results and relevance judgments!")
		logger.error("Cannot compute evaluation metrics. Saving run files only.")
		
		# Save run files without evaluation
		if save_runs:
			for weighting_mode, model_results in all_results.items():
				for model_type, reranked_results in model_results.items():
					run_file = os.path.join(output_dir, f"{dataset_name}_{model_type}_{weighting_mode}.run")
					save_results_file(reranked_results, run_file)
					
		# Create a minimal summary with just configuration info
		summary = {
			"dataset": dataset_name,
			"error": "No overlap between query IDs in results and relevance judgments",
			"queries_in_results": len(initial_results),
			"queries_in_qrels": len(qrels),
			"models_evaluated": [model for model in list(all_results.values())[0].keys()] if all_results else []
		}
		
		# Save summary
		summary_file = os.path.join(output_dir, f"{dataset_name}_weighting_modes_summary.json")
		with open(summary_file, 'w') as f:
			json.dump(summary, f, indent=2)
			
		logger.info(f"Saved minimal summary to {summary_file}")
		return
	
	# First evaluate initial results
	logger.info("Evaluating initial retrieval results")
	try:
		init_ndcg, init_map, init_recall, init_precision = evaluator.evaluate(
			valid_qrels, initial_results, evaluator.k_values
		)
		evaluation_successful = True
	except Exception as e:
		logger.error(f"Error evaluating initial results: {e}")
		evaluation_successful = False
		init_ndcg, init_map, init_recall, init_precision = {}, {}, {}, {}
	
	# Create summary metrics dictionary
	summary = {
		"dataset": dataset_name,
		"initial": {},
		"weighting_modes": {}
	}
	
	# Add initial retrieval metrics to summary
	if evaluation_successful:
		for k in evaluator.k_values:
			ndcg_key = f"NDCG@{k}"
			map_key = f"MAP@{k}"
			recall_key = f"Recall@{k}"
			precision_key = f"P@{k}"
			
			if ndcg_key in init_ndcg:
				summary["initial"][f"ndcg@{k}"] = init_ndcg[ndcg_key]
				summary["initial"][f"map@{k}"] = init_map[map_key]
				summary["initial"][f"recall@{k}"] = init_recall[recall_key]
				summary["initial"][f"precision@{k}"] = init_precision[precision_key]
	
	# Evaluate each weighting mode and model type if evaluation is possible
	for weighting_mode, model_results in all_results.items():
		logger.info(f"Processing results for weighting mode: {weighting_mode}")
		
		# Initialize summary for this weighting mode
		summary["weighting_modes"][weighting_mode] = {}
		
		for model_type, reranked_results in model_results.items():
			logger.info(f"Processing model: {model_type}")
			
			# Save run file if requested
			if save_runs:
				run_file = os.path.join(output_dir, f"{dataset_name}_{model_type}_{weighting_mode}.run")
				save_results_file(reranked_results, run_file)
			
			# Skip evaluation if we already know it will fail
			if not evaluation_successful:
				summary["weighting_modes"][weighting_mode][model_type] = {
					"error": "Evaluation not possible due to lack of relevance judgments"
				}
				continue
				
			# Evaluate
			try:
				ndcg, _map, recall, precision = evaluator.evaluate(
					valid_qrels, reranked_results, evaluator.k_values
				)
				
				# Store metrics for this model
				model_metrics = {
					"metrics": {}
				}
				
				# Initialize summary for this model
				summary["weighting_modes"][weighting_mode][model_type] = {}
				
				# Print results and store in metrics
				print(f"\n===== {model_type.upper()} Model with {weighting_mode} weighting =====")
				for k in evaluator.k_values:
					ndcg_key = f"NDCG@{k}"
					map_key = f"MAP@{k}"
					recall_key = f"Recall@{k}"
					precision_key = f"P@{k}"
					
					if ndcg_key in ndcg and ndcg_key in init_ndcg:
						ndcg_improve = ndcg[ndcg_key] - init_ndcg[ndcg_key]
						map_improve = _map[map_key] - init_map[map_key]
						recall_improve = recall[recall_key] - init_recall[recall_key]
						precision_improve = precision[precision_key] - init_precision[precision_key]
						
						# Print metrics
						print(f"NDCG@{k}: {ndcg[ndcg_key]:.4f} (Δ: {ndcg_improve:+.4f})")
						print(f"MAP@{k}: {_map[map_key]:.4f} (Δ: {map_improve:+.4f})")
						print(f"Recall@{k}: {recall[recall_key]:.4f} (Δ: {recall_improve:+.4f})")
						print(f"P@{k}: {precision[precision_key]:.4f} (Δ: {precision_improve:+.4f})")
						print()
						
						# Store metrics
						model_metrics["metrics"][f"k={k}"] = {
							"ndcg": {
								"reranked": ndcg[ndcg_key],
								"initial": init_ndcg[ndcg_key],
								"delta": ndcg_improve
							},
							"map": {
								"reranked": _map[map_key],
								"initial": init_map[map_key],
								"delta": map_improve
							},
							"recall": {
								"reranked": recall[recall_key],
								"initial": init_recall[recall_key],
								"delta": recall_improve
							},
							"precision": {
								"reranked": precision[precision_key],
								"initial": init_precision[precision_key],
								"delta": precision_improve
							}
						}
						
						# Add to summary
						summary["weighting_modes"][weighting_mode][model_type][f"ndcg@{k}"] = ndcg[ndcg_key]
						summary["weighting_modes"][weighting_mode][model_type][f"ndcg@{k}_delta"] = ndcg_improve
						summary["weighting_modes"][weighting_mode][model_type][f"map@{k}"] = _map[map_key]
						summary["weighting_modes"][weighting_mode][model_type][f"recall@{k}"] = recall[recall_key]
						summary["weighting_modes"][weighting_mode][model_type][f"precision@{k}"] = precision[precision_key]
				
				# Save metrics if requested
				if save_metrics:
					metrics_file = os.path.join(output_dir, f"{dataset_name}_{model_type}_{weighting_mode}_metrics.json")
					with open(metrics_file, 'w') as f:
						json.dump(model_metrics, f, indent=2)
			except Exception as e:
				logger.error(f"Error evaluating {model_type} with {weighting_mode} weighting: {e}")
				summary["weighting_modes"][weighting_mode][model_type] = {
					"error": str(e)
				}
	
	# Save summary metrics
	summary_file = os.path.join(output_dir, f"{dataset_name}_weighting_modes_summary.json")
	with open(summary_file, 'w') as f:
		json.dump(summary, f, indent=2)
	
	logger.info(f"Saved summary metrics to {summary_file}")
	
	# Print comparative summary (only if evaluation was successful)
	if evaluation_successful:
		print("\n===== COMPARATIVE SUMMARY =====")
		print(f"Dataset: {dataset_name}")
		print("\nNDCG@10 Comparison:")
		print(f"{'Model/Mode':<20} {'NDCG@10':<10} {'Δ':<10}")
		print("-" * 40)
		
		# Print initial results
		ndcg_key_10 = "NDCG@10"
		if ndcg_key_10 in init_ndcg:
			print(f"{'Initial':<20} {init_ndcg[ndcg_key_10]:.4f}     {0.0:+.4f}")
			
			# Build a list of all (model_type, weighting_mode) pairs
			all_pairs = []
			for weighting_mode in summary["weighting_modes"]:
				for model_type in summary["weighting_modes"][weighting_mode]:
					if f"ndcg@10" in summary["weighting_modes"][weighting_mode][model_type]:
						ndcg10 = summary["weighting_modes"][weighting_mode][model_type]["ndcg@10"]
						delta = summary["weighting_modes"][weighting_mode][model_type]["ndcg@10_delta"]
						all_pairs.append((model_type, weighting_mode, ndcg10, delta))
			
			# Sort by NDCG@10 performance
			all_pairs.sort(key=lambda x: x[2], reverse=True)
			
			# Print sorted results
			for model_type, weighting_mode, ndcg10, delta in all_pairs:
				print(f"{model_type}/{weighting_mode:<20} {ndcg10:.4f}     {delta:+.4f}")
		else:
			print("NDCG@10 not available in evaluation results")
	else:
		print("\n===== EVALUATION NOT POSSIBLE =====")
		print("No overlap between query IDs in results and relevance judgments")
		print(f"Queries in results: {len(initial_results)}")
		print(f"Queries in qrels: {len(qrels)}")
		print("Run files were saved successfully even though evaluation metrics could not be computed")

def main():
	parser = argparse.ArgumentParser(description="Evaluate reranking with multiple weighting strategies")
	
	# Dataset configuration
	parser.add_argument("--dataset", required=True, 
						help="BEIR dataset name (e.g., msmarco, trec-covid, nfcorpus, quora)")
	parser.add_argument("--split", default="dev", choices=["dev", "test"],
						help="Dataset split to use (dev or test)")
	parser.add_argument("--data_dir", default="./datasets",
						help="Directory to store/load datasets")
	
	# Retrieval configuration
	parser.add_argument("--initial_ranking_file", type=str, required=True,
						help="Path to pre-computed initial rankings file")
	
	# Query limiting for faster testing
	parser.add_argument("--max_queries", type=int, default=None,
						help="Maximum number of queries to process (for testing)")
	parser.add_argument("--query_ids", type=str, default=None,
						help="Comma-separated list of specific query IDs to process")
	
	# Model configuration
	parser.add_argument("--checkpoint_dir", type=str, required=True,
						help="Base directory containing model checkpoints")
	parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf",
						help="HuggingFace model identifier")
	parser.add_argument("--layers", default="0,3,6,9,12,15,18,21",
						help="Comma-separated layer indices")
	parser.add_argument("--dtw_layers", default="6,9,12,15",
						help="Comma-separated DTW layer indices")
	parser.add_argument("--weighter_step", type=int, default=None,
						help="Specific checkpoint step to load (default: latest)")
	parser.add_argument("--model_types", type=str, default=None,
						help="Comma-separated list of model types to evaluate (e.g., 'dtw,layer_12')")
	
	# Weighting configuration
	parser.add_argument("--weighting_modes", type=str, default="uniform,learned,combined",
						help="Comma-separated list of weighting modes to evaluate")
	parser.add_argument("--log_weights_file", type=str, default=None,
						help="Path to token log weights file (for log_weights mode)")
	
	# Reranking parameters
	parser.add_argument("--top_k", type=int, default=100,
						help="Number of documents to rerank per query")
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size for document encoding")
	
	# Output options
	parser.add_argument("--output_dir", type=str, default="./results/weighting_comparison",
						help="Directory to save output files")
	parser.add_argument("--save_runs", action="store_true",
						help="Save reranked results for each model in TREC format")
	parser.add_argument("--save_metrics", action="store_true",
						help="Save evaluation metrics for each model in JSON format")
	
	# Flag to skip evaluation if there are no qrels
	parser.add_argument("--skip_no_qrel_eval", action="store_true",
						help="Skip evaluation if there are no query relevance judgments")
	
	args = parser.parse_args()
	
	# Process query IDs if specified
	query_id_list = None
	if args.query_ids:
		query_id_list = args.query_ids.split(',')
		logger.info(f"Processing specific query IDs: {query_id_list}")
	
	# Convert string lists to integer lists
	layer_indices = [int(x) for x in args.layers.split(',')]
	dtw_layer_indices = [int(x) for x in args.dtw_layers.split(',')]
	
	# Process weighting modes
	weighting_modes = [mode.strip() for mode in args.weighting_modes.split(',')]
	logger.info(f"Evaluating weighting modes: {weighting_modes}")
	
	# Process model types if specified
	requested_model_types = None
	if args.model_types:
		requested_model_types = [mt.strip() for mt in args.model_types.split(',')]
		logger.info(f"Evaluating specific model types: {requested_model_types}")
	
	# Create output directory if needed
	os.makedirs(args.output_dir, exist_ok=True)
	
	# 1. Download and load dataset
	logger.info(f"Loading {args.dataset} dataset ({args.split} split)")
	data_path = os.path.join(args.data_dir, args.dataset)
	
	# Download the dataset if not already present
	if not os.path.exists(data_path):
		url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
		data_path = util.download_and_unzip(url, args.data_dir)
	
	# Load the corpus, queries, and relevance judgments
	corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
	logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} qrels")
	
	# Set up evaluator for metrics calculation
	evaluator = EvaluateRetrieval(k_values=[1, 3, 5, 10, 20, 100])
	
	# 2. Load initial rankings
	logger.info(f"Loading initial rankings from {args.initial_ranking_file}")
	initial_results = load_initial_rankings(args.initial_ranking_file)
	
	# Ensure we have qrels for all queries we're evaluating
	valid_query_ids = set(initial_results.keys()) & set(qrels.keys())
	logger.info(f"Found {len(valid_query_ids)} of {len(initial_results)} queries with relevance judgments")
	
	# Check if we have enough qrels to evaluate
	if len(valid_query_ids) == 0:
		logger.warning("No overlap between query IDs in results and relevance judgments!")
		if args.skip_no_qrel_eval:
			logger.info("Skipping evaluation as requested by --skip_no_qrel_eval")
			return
		else:
			logger.info("Proceeding with reranking, but evaluation will not be meaningful")
	
	# 3. Load the model
	model, available_model_types = load_model(
		checkpoint_dir=args.checkpoint_dir,
		model_name=args.model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		weighter_step=args.weighter_step
	)
	
	if model is None or not available_model_types:
		logger.error("No models could be loaded. Exiting.")
		return
	
	# Filter model types if requested
	if requested_model_types:
		available_model_types = [mt for mt in available_model_types if mt in requested_model_types]
		if not available_model_types:
			logger.error("None of the requested model types are available. Exiting.")
			return
		logger.info(f"Filtered to {len(available_model_types)} requested model types")
	
	# 4. Initialize with log weights if needed and file provided
	if "log_weights" in weighting_modes and args.log_weights_file:
		if not initialize_with_log_weights(model, args.log_weights_file):
			logger.warning("Failed to initialize with log weights")
			# Remove log_weights mode if initialization failed
			weighting_modes = [mode for mode in weighting_modes if mode != "log_weights"]
	
	# 5. Perform reranking with all weighting modes
	all_reranked_results = rerank_with_all_weighting_modes(
		model=model,
		available_model_types=available_model_types,
		corpus=corpus,
		queries=queries,
		results=initial_results,
		weighting_modes=weighting_modes,
		top_k=args.top_k,
		batch_size=args.batch_size,
		max_queries=args.max_queries,
		query_ids=query_id_list
	)
	
	# 6. Evaluate and save results
	evaluate_and_save_results(
		all_results=all_reranked_results,
		evaluator=evaluator,
		initial_results=initial_results,
		qrels=qrels,
		output_dir=args.output_dir,
		dataset_name=args.dataset,
		save_runs=args.save_runs,
		save_metrics=args.save_metrics
	)
	
	# Unload the model to free memory
	model.unload_model()
	logger.info("Evaluation completed")

if __name__ == "__main__":
	main()