#!/usr/bin/env python
# evaluate_all_models.py
"""
Usage Examples:

1. Evaluate all available models with parallel processing:
```
python evaluate_all_models.py \
  --dataset quora \
  --split dev \
  --initial_ranking_file ./quora_bm25_results.tsv \
  --checkpoint_dir ./model_checkpoints/checkpoints \
  --output_dir ./results \
  --save_metrics \
  --save_runs \
  --parallel \
  --workers 4
```

2. Evaluate a specific model (e.g., DTW) with parallel processing:
```
python evaluate_all_models.py \
  --dataset quora \
  --split dev \
  --initial_ranking_file ./quora_bm25_results.tsv \
  --checkpoint_dir ./model_checkpoints/checkpoints \
  --model_type dtw \
  --output_dir ./results \
  --save_metrics \
  --parallel \
  --workers 4
```

3. Evaluate all models with a limited number of queries:
```
python evaluate_all_models.py \
  --dataset quora \
  --split dev \
  --initial_ranking_file ./quora_bm25_results.tsv \
  --checkpoint_dir ./model_checkpoints/checkpoints \
  --max_queries 100 \
  --output_dir ./results \
  --save_metrics \
  --parallel \
  --workers 4
```
"""

import argparse
import logging
import os
import torch
import multiprocessing
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import math
import pickle
import time
from datetime import timedelta
import random
import json

# Import BEIR components
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

def get_device():
	"""Get the appropriate device for computation"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	elif torch.backends.mps.is_available():
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

		logging.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free (Total: {total_memory:.2f}GB)")
	else:
		logging.info("No GPU available")

def load_initial_rankings(file_path: str) -> Dict[str, Dict[str, float]]:
	"""
	Load pre-computed initial rankings from a file

	Args:
		file_path: Path to the rankings file (TREC format: qid docid rank)

	Returns:
		Dictionary mapping query_id to {doc_id: score}
	"""
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
		logging.error(f"Error loading initial rankings: {e}")
		raise

	logging.info(f"Loaded rankings for {len(results)} queries")
	return results

def save_results_file(results: Dict[str, Dict[str, float]], file_path: str):
	"""Save results in TREC format"""
	with open(file_path, 'w') as f:
		for query_id, doc_scores in results.items():
			# Sort documents by score
			sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
			for rank, (doc_id, score) in enumerate(sorted_docs, 1):
				f.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")
	logging.info(f"Saved results for {len(results)} queries to {file_path}")

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

	# Check for DTW model
	dtw_dir = os.path.join(checkpoint_dir, "dtw_model")
	if os.path.exists(dtw_dir):
		# Verify it has checkpoint files
		checkpoint_files = [f for f in os.listdir(dtw_dir) 
						   if f.startswith("weighter_step_") and f.endswith(".pt")]
		if checkpoint_files:
			available_models.append("dtw")

	# Check for layer models
	for layer_idx in layer_indices:
		layer_dir = os.path.join(checkpoint_dir, f"single_layer_{layer_idx}")
		if os.path.exists(layer_dir):
			# Verify it has checkpoint files
			checkpoint_files = [f for f in os.listdir(layer_dir) 
							   if f.startswith("weighter_step_") and f.endswith(".pt")]
			if checkpoint_files:
				available_models.append(f"layer_{layer_idx}")

	return available_models

def load_multi_layer_reranker_model(
	checkpoint_dir: str,
	model_name: str = "meta-llama/Llama-2-7b-hf",
	layer_indices: List[int] = [0, 3, 6, 9, 12, 15, 18, 21],
	dtw_layer_indices: List[int] = [6, 9, 12, 15],
	weighter_step: Optional[int] = None,
	single_model_type: Optional[str] = None
):
	"""
	Load the MultiLayerReranker model with weights for all models or a specific model

	Args:
		checkpoint_dir: Base directory containing model checkpoints
		model_name: Name of the base LLM model
		layer_indices: Which layers to use in the reranker
		dtw_layer_indices: Which layers to use for DTW
		weighter_step: Specific checkpoint step to load (None for latest)
		single_model_type: If specified, only load this model type (else load all)

	Returns:
		Tuple of (loaded model, available_model_types)
	"""
	# Import here to avoid import errors if user only wants to run BM25
	try:
		from multi_layer_reranker import MultiLayerReranker
	except ImportError:
		logging.error("Failed to import MultiLayerReranker. Make sure the model code is in your Python path.")
		return None, []

	# First check which models are available
	if single_model_type:
		available_model_types = [single_model_type] if single_model_type in check_available_models(checkpoint_dir, layer_indices) else []
		if not available_model_types:
			logging.error(f"Specified model '{single_model_type}' not found in checkpoint directory or has no checkpoints")
			return None, []
	else:
		available_model_types = check_available_models(checkpoint_dir, layer_indices)
		if not available_model_types:
			logging.error(f"No valid models with checkpoints found in {checkpoint_dir}")
			return None, []

	logging.info(f"Found {len(available_model_types)} available models: {', '.join(available_model_types)}")

	device = get_device()
	logging.info(f"Using device: {device}")

	# Initialize the model
	model = MultiLayerReranker(
		model_name=model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		device=device
	)

	# Load model into memory (loads the LLaMA model)
	model.load_model()
	print_gpu_utilization()

	# Prepare to load weights
	models_to_load = []

	# Check if DTW model should be loaded
	if "dtw" in available_model_types:
		models_to_load.append(("dtw", os.path.join(checkpoint_dir, "dtw_model")))

	# Check which layer models should be loaded
	for model_type in available_model_types:
		if model_type.startswith("layer_"):
			layer_idx = int(model_type.split("_")[1])
			models_to_load.append((model_type, os.path.join(checkpoint_dir, f"single_layer_{layer_idx}")))

	# Load weights for each model
	for model_type, model_dir in models_to_load:
		# Find checkpoint file - either specific step or latest
		if weighter_step is None:
			# Find the latest checkpoint
			checkpoint_files = [f for f in os.listdir(model_dir) 
								if f.startswith("weighter_step_") and f.endswith(".pt")]
			if not checkpoint_files:
				logging.warning(f"No checkpoint files found in {model_dir}, skipping {model_type}")
				continue

			# Sort by step number to find latest
			latest_checkpoint = sorted(
				checkpoint_files,
				key=lambda x: int(x.split("_")[2].split(".")[0])
			)[-1]

			checkpoint_file = os.path.join(model_dir, latest_checkpoint)
			step = int(latest_checkpoint.split("_")[2].split(".")[0])
		else:
			# Use specified step
			checkpoint_file = os.path.join(model_dir, f"weighter_step_{weighter_step}.pt")
			step = weighter_step

			if not os.path.exists(checkpoint_file):
				logging.warning(f"Checkpoint not found: {checkpoint_file}, skipping {model_type}")

				# Show available steps
				available_checkpoints = [f for f in os.listdir(model_dir) 
									if f.startswith("weighter_step_") and f.endswith(".pt")]
				if available_checkpoints:
					available_steps = sorted([int(f.split("_")[2].split(".")[0]) for f in available_checkpoints])
					logging.info(f"Available steps for {model_type}: {available_steps}")

				# Remove from available models
				if model_type in available_model_types:
					available_model_types.remove(model_type)
				continue

		logging.info(f"Loading weights for {model_type} from {checkpoint_file}")

		# Load the weights
		try:
			if model_type == "dtw":
				model.dtw_weighter.load_state_dict(
					torch.load(checkpoint_file, map_location=device)
				)
				logging.info(f"Loaded DTW weighter from step {step}")
			else:
				model.layer_weighters[model_type].load_state_dict(
					torch.load(checkpoint_file, map_location=device)
				)
				logging.info(f"Loaded {model_type} weighter from step {step}")
		except Exception as e:
			logging.error(f"Error loading weights for {model_type}: {e}")
			if model_type in available_model_types:
				available_model_types.remove(model_type)

	model.eval()  # Set to evaluation mode
	print_gpu_utilization()  # Check GPU usage after loading

	if not available_model_types:
		logging.error("No models could be loaded successfully")
		return None, []

	return model, available_model_types

def rerank_all_models_single_pass(
	model,
	available_model_types: List[str],
	corpus: Dict[str, Dict[str, str]],
	queries: Dict[str, str],
	results: Dict[str, Dict[str, float]],
	top_k: int = 100,
	batch_size: int = 32,
	max_queries: Optional[int] = None,
	query_ids: Optional[List[str]] = None,
	parallel: bool = False,
	num_workers: int = 1
) -> Dict[str, Dict[str, Dict[str, float]]]:
	"""
	Rerank the initial retrieval results using all available models in a single pass

	Args:
		model: Loaded MultiLayerReranker model
		available_model_types: List of model types to use for reranking
		corpus: Dictionary mapping doc_id to document data
		queries: Dictionary mapping query_id to query text
		results: Initial retrieval results {query_id: {doc_id: score}}
		top_k: How many documents to rerank per query
		batch_size: Batch size for document encoding
		max_queries: Maximum number of queries to process
		query_ids: List of specific query IDs to process
		parallel: Whether to use parallel processing
		num_workers: Number of parallel workers

	Returns:
		Dictionary mapping model_type to reranked results {query_id: {doc_id: score}}
	"""
	# Initialize results dictionary for each model
	all_model_results = {model_type: {} for model_type in available_model_types}

	# Select queries to process
	if query_ids is not None:
		# Filter for valid query IDs
		valid_qids = set(queries.keys()) & set(results.keys())
		query_ids = [qid for qid in query_ids if qid in valid_qids]
		if not query_ids:
			logging.warning("None of the specified query IDs were found in the dataset")
			return all_model_results
		limited_queries = {qid: queries[qid] for qid in query_ids}
		logging.info(f"Processing {len(limited_queries)} specified queries")
	elif max_queries is not None and max_queries < len(results):
		# Randomly select queries
		logging.info(f"Limiting to {max_queries} of {len(results)} queries")
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

	logging.info(f"Reranking top-{top_k} documents for {len(query_ids)} queries using {len(available_model_types)} models")
	logging.info(f"Models: {', '.join(available_model_types)}")

	# Start timing
	start_time = time.time()

	logging.info("Using sequential processing")

	for query_id in tqdm(query_ids, desc="Reranking queries"):
		if query_id not in queries:
			continue

		query_text = queries[query_id]

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
			for model_type in available_model_types:
				all_model_results[model_type][query_id] = {}
			continue

		# Get rankings for all models in a single pass by using "all" as model_type
		# This ensures that we only do a single forward pass through the LLaMA model
		rankings = model.rerank(
			query=query_text,
			documents=candidate_docs,
			model_type="all",  # This is the key change to get scores for all models in one pass
			batch_size=batch_size,
			return_scores=True
		)

		# Process results for each model
		for model_type in available_model_types:
			if model_type in rankings:
				sorted_indices, sorted_scores = rankings[model_type]

				# Map scores back to doc_ids
				doc_scores = {}
				for idx, score in zip(sorted_indices, sorted_scores):
					if idx < len(valid_doc_ids):  # Safety check
						doc_id = valid_doc_ids[idx]
						doc_scores[doc_id] = score

				all_model_results[model_type][query_id] = doc_scores
			else:
				logging.warning(f"Model type '{model_type}' not found in rankings output for query {query_id}")
				all_model_results[model_type][query_id] = {}

		# Print GPU usage periodically
		if len(all_model_results[available_model_types[0]]) % 10 == 0:
			print_gpu_utilization()

			# Calculate ETA
			processed = len(all_model_results[available_model_types[0]])
			elapsed = time.time() - start_time
			queries_per_sec = processed / elapsed
			remaining = len(query_ids) - processed
			eta_seconds = remaining / queries_per_sec if queries_per_sec > 0 else 0

			logging.info(f"Processed {processed}/{len(query_ids)} queries "
							f"({queries_per_sec:.2f} queries/sec, "
							f"ETA: {timedelta(seconds=int(eta_seconds))})")

	# Final timing
	total_time = time.time() - start_time
	logging.info(f"Reranking completed in {timedelta(seconds=int(total_time))}, "
				 f"average {len(query_ids)/total_time:.2f} queries/sec")

	return all_model_results

def main():
	parser = argparse.ArgumentParser(description="Evaluate all MultiLayerReranker models efficiently")

	# Dataset configuration
	parser.add_argument("--dataset", default="msmarco", 
						help="BEIR dataset name (e.g., msmarco, trec-covid, nfcorpus)")
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
	parser.add_argument("--model_type", default=None,
						help="Only evaluate this specific model (default: evaluate all available models)")

	# Reranking parameters
	parser.add_argument("--top_k", type=int, default=100,
						help="Number of documents to rerank per query")
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size for document encoding")

	# Parallelization
	parser.add_argument("--parallel", action="store_true",
						help="Use parallel processing for evaluation")
	parser.add_argument("--workers", type=int, default=4,
						help="Number of worker threads for parallel processing")

	# Output options
	parser.add_argument("--output_dir", type=str, default="./results",
						help="Directory to save output files")
	parser.add_argument("--save_runs", action="store_true",
						help="Save reranked results for each model in TREC format")
	parser.add_argument("--save_metrics", action="store_true",
						help="Save evaluation metrics for each model in JSON format")

	args = parser.parse_args()

	# Process query IDs if specified
	query_id_list = None
	if args.query_ids:
		query_id_list = args.query_ids.split(',')
		logging.info(f"Processing specific query IDs: {query_id_list}")

	# Convert string lists to integer lists
	layer_indices = [int(x) for x in args.layers.split(',')]
	dtw_layer_indices = [int(x) for x in args.dtw_layers.split(',')]

	# Create output directory if needed
	os.makedirs(args.output_dir, exist_ok=True)

	# 1. Download and load dataset
	logging.info(f"Loading {args.dataset} dataset ({args.split} split)")
	data_path = os.path.join(args.data_dir, args.dataset)

	# Download the dataset if not already present
	if not os.path.exists(data_path):
		url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
		data_path = util.download_and_unzip(url, args.data_dir)

	# Load the corpus, queries, and relevance judgments
	corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
	logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} qrels")

	# Set up evaluator for the metrics calculation
	evaluator = EvaluateRetrieval(k_values=[1, 3, 5, 10, 20, 100])

	# 2. Load initial rankings
	logging.info(f"Loading initial rankings from {args.initial_ranking_file}")
	initial_results = load_initial_rankings(args.initial_ranking_file)

	# Ensure we have qrels for all queries we're evaluating
	valid_query_ids = set(initial_results.keys()) & set(qrels.keys())
	if len(valid_query_ids) < len(initial_results):
		logging.warning(f"Only {len(valid_query_ids)} of {len(initial_results)} queries have relevance judgments")
		initial_results = {qid: initial_results[qid] for qid in valid_query_ids}

	# 3. Load the model(s)
	model, available_model_types = load_multi_layer_reranker_model(
		checkpoint_dir=args.checkpoint_dir,
		model_name=args.model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		weighter_step=args.weighter_step,
		single_model_type=args.model_type
	)

	if model is None or not available_model_types:
		logging.error("No models could be loaded. Exiting.")
		return

	# 4. Rerank with all models in a single pass per query
	all_reranked_results = rerank_all_models_single_pass(
		model=model,
		available_model_types=available_model_types,
		corpus=corpus,
		queries=queries,
		results=initial_results,
		top_k=args.top_k,
		batch_size=args.batch_size,
		max_queries=args.max_queries,
		query_ids=query_id_list,
		parallel=args.parallel,
		num_workers=args.workers
	)

	# 5. Evaluate initial retrieval results
	logging.info("Evaluating initial retrieval results")
	init_ndcg, init_map, init_recall, init_precision = evaluator.evaluate(
		qrels, initial_results, evaluator.k_values
	)

	# Create a summary of all results for comparison
	summary_metrics = {
		"dataset": args.dataset,
		"split": args.split,
		"weighter_step": args.weighter_step if args.weighter_step else "latest",
		"initial": {},
		"models": {}
	}

	# Add initial retrieval metrics to summary - FIXED: Use proper key format
	for k in evaluator.k_values:
		# Use properly formatted keys for each metric type
		ndcg_key = f"NDCG@{k}"
		map_key = f"MAP@{k}"
		recall_key = f"Recall@{k}"
		precision_key = f"P@{k}"

		# Check if keys exist and store with our own format
		if ndcg_key in init_ndcg:
			summary_metrics["initial"][f"ndcg@{k}"] = init_ndcg[ndcg_key]
			summary_metrics["initial"][f"map@{k}"] = init_map[map_key]
			summary_metrics["initial"][f"recall@{k}"] = init_recall[recall_key]
			summary_metrics["initial"][f"precision@{k}"] = init_precision[precision_key]
		else:
			logging.warning(f"Key {ndcg_key} not found in evaluation metrics")
			summary_metrics["initial"][f"ndcg@{k}"] = 0.0
			summary_metrics["initial"][f"map@{k}"] = 0.0
			summary_metrics["initial"][f"recall@{k}"] = 0.0
			summary_metrics["initial"][f"precision@{k}"] = 0.0

	# 6. Evaluate and save results for each model
	for model_type in available_model_types:
		reranked_results = all_reranked_results[model_type]

		# Evaluate
		logging.info(f"Evaluating {model_type} reranking")
		ndcg, _map, recall, precision = evaluator.evaluate(
			qrels, reranked_results, evaluator.k_values
		)

		# Save reranked results if requested
		if args.save_runs:
			run_file = os.path.join(args.output_dir, f"{args.dataset}_{model_type}_reranked.run")
			save_results_file(reranked_results, run_file)

		# Collect metrics for this model
		model_metrics = {
			"config": {
				"model_type": model_type,
				"weighter_step": args.weighter_step if args.weighter_step else "latest",
				"queries_evaluated": len(reranked_results)
			},
			"metrics": {}
		}

		# Add to summary metrics
		summary_metrics["models"][model_type] = {}

		# Print results for this model
		print(f"\n===== {model_type.upper()} Model Results =====")
		print(f"Dataset: {args.dataset} ({args.split} split)")
		print(f"Queries evaluated: {len(reranked_results)}")

		for k in evaluator.k_values:
			# Use properly formatted keys for each metric type
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

				# Store detailed metrics
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
				summary_metrics["models"][model_type][f"ndcg@{k}"] = ndcg[ndcg_key]
				summary_metrics["models"][model_type][f"ndcg@{k}_delta"] = ndcg_improve
				summary_metrics["models"][model_type][f"map@{k}"] = _map[map_key]
				summary_metrics["models"][model_type][f"recall@{k}"] = recall[recall_key]
				summary_metrics["models"][model_type][f"precision@{k}"] = precision[precision_key]
			else:
				logging.warning(f"Key {ndcg_key} not found in evaluation metrics for {model_type}")
				# Handle missing keys gracefully
				print(f"NDCG@{k}: N/A")
				print(f"MAP@{k}: N/A")
				print(f"Recall@{k}: N/A")
				print(f"P@{k}: N/A")
				print()

				# Add zeros to summary
				summary_metrics["models"][model_type][f"ndcg@{k}"] = 0.0
				summary_metrics["models"][model_type][f"ndcg@{k}_delta"] = 0.0
				summary_metrics["models"][model_type][f"map@{k}"] = 0.0
				summary_metrics["models"][model_type][f"recall@{k}"] = 0.0
				summary_metrics["models"][model_type][f"precision@{k}"] = 0.0

		# Save individual model metrics if requested
		if args.save_metrics:
			metrics_file = os.path.join(args.output_dir, f"{args.dataset}_{model_type}_metrics.json")
			with open(metrics_file, 'w') as f:
				json.dump(model_metrics, f, indent=2)
			logging.info(f"Saved {model_type} metrics to {metrics_file}")

	# Save summary metrics
	summary_file = os.path.join(args.output_dir, f"{args.dataset}_summary.json")
	with open(summary_file, 'w') as f:
		json.dump(summary_metrics, f, indent=2)
	logging.info(f"Saved summary metrics to {summary_file}")

	# Print comparative summary
	print("\n===== COMPARATIVE SUMMARY =====")
	print(f"Dataset: {args.dataset} ({args.split} split)")
	print(f"Step: {args.weighter_step if args.weighter_step else 'latest'}")
	print(f"Total queries evaluated: {len(all_reranked_results[available_model_types[0]])}")
	print("\nNDCG@10 Comparison:")
	print(f"{'Model':<12} {'NDCG@10':<10} {'Δ':<10}")
	print("-" * 32)

	# Make sure we have NDCG@10 before accessing - FIXED to use correct key
	ndcg_key_10 = "NDCG@10"
	if ndcg_key_10 in init_ndcg:
		print(f"{'Initial':<12} {init_ndcg[ndcg_key_10]:.4f}     {0.0:+.4f}")

		# Sort models by NDCG@10 performance - FIXED with safe access
		sorted_models = []
		for model_type in available_model_types:
			if f"ndcg@10" in summary_metrics["models"][model_type]:
				sorted_models.append(model_type)

		if sorted_models:
			sorted_models = sorted(
				sorted_models,
				key=lambda m: summary_metrics["models"][m]["ndcg@10"],
				reverse=True
			)

			for model_type in sorted_models:
				ndcg10 = summary_metrics["models"][model_type]["ndcg@10"]
				delta = summary_metrics["models"][model_type]["ndcg@10_delta"]
				print(f"{model_type:<12} {ndcg10:.4f}     {delta:+.4f}")
		else:
			print("No models have valid NDCG@10 metrics")
	else:
		print("NDCG@10 not available in evaluation results")

	# Unload the model to free memory
	model.unload_model()

	logging.info("Evaluation completed")

if __name__ == "__main__":
	main()