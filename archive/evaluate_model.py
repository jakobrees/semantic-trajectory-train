import os
import sys
import argparse
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
from tqdm import tqdm
from collections import defaultdict, Counter
import time

# Import our model modules
from multi_layer_reranker import MultiLayerReranker
from reranker_model import VocabLookupWeighter, get_device

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler("evaluation.log"),
		logging.StreamHandler(sys.stdout)
	]
)
logger = logging.getLogger('evaluation')

def load_queries(queries_file: str) -> Dict[int, str]:
	"""Load query IDs and text from file"""
	queries = {}
	logger.info(f"Loading queries from {queries_file}")

	try:
		with open(queries_file, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc="Loading queries"):
				parts = line.strip().split('\t')
				if len(parts) >= 2:
					try: 
						qid, text = int(parts[0]), parts[1]
						queries[qid] = text
					except ValueError:
						logger.warning(f"Skipping malformed line in queries: {line.strip()[:100]}...")
	except Exception as e:
		logger.error(f"Error loading queries: {e}")
		sys.exit(1)

	logger.info(f"Loaded {len(queries)} queries")
	return queries

def load_collection(collection_file: str) -> Dict[int, str]:
	"""Load passage IDs and text from file"""
	passages = {}
	logger.info(f"Loading collection from {collection_file}")

	try:
		with open(collection_file, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc="Loading collection"):
				parts = line.strip().split('\t')
				if len(parts) >= 2:
					try:
						pid, text = int(parts[0]), parts[1]
						passages[pid] = text
					except ValueError:
						logger.warning(f"Skipping malformed line in collection: {line.strip()[:100]}...")
	except Exception as e:
		logger.error(f"Error loading collection: {e}")
		sys.exit(1)

	logger.info(f"Loaded {len(passages)} passages")
	return passages

def load_top1000(run_file: str) -> Dict[int, List[int]]:
	"""Load initial top 1000 results for each query"""
	rankings = {}
	logger.info(f"Loading initial rankings from {run_file}")

	try:
		with open(run_file, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc="Loading top1000"):
				parts = line.strip().split('\t')
				if len(parts) >= 3:
					try:
						qid, pid, rank = int(parts[0]), int(parts[1]), int(parts[2])
						if qid not in rankings:
							rankings[qid] = []
						rankings[qid].append((pid, rank))
					except ValueError:
						logger.warning(f"Skipping malformed line in run file: {line.strip()[:100]}...")
	except Exception as e:
		logger.error(f"Error loading top1000: {e}")
		sys.exit(1)

	# Sort passage IDs by rank for each query
	for qid in rankings:
		rankings[qid] = [pid for pid, rank in sorted(rankings[qid], key=lambda x: x[1])]

	logger.info(f"Loaded rankings for {len(rankings)} queries")
	return rankings

def load_qrels(qrels_file: str) -> Dict[int, List[int]]:
	"""Load relevance judgments"""
	qrels = {}
	logger.info(f"Loading qrels from {qrels_file}")

	try:
		with open(qrels_file, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc="Loading qrels"):
				parts = line.strip().split('\t')
				if len(parts) >= 3:
					try:
						qid, pid = int(parts[0]), int(parts[2])
						if qid not in qrels:
							qrels[qid] = []
						qrels[qid].append(pid)
					except ValueError:
						logger.warning(f"Skipping malformed line in qrels: {line.strip()[:100]}...")
	except Exception as e:
		logger.error(f"Error loading qrels: {e}")
		sys.exit(1)

	logger.info(f"Loaded qrels for {len(qrels)} queries")
	return qrels

def load_model(
	checkpoint_path: str,
	model_name: str = "meta-llama/Llama-2-7b-hf",
	layer_indices: List[int] = [0, 3, 6, 9, 12, 15, 18, 21],
	dtw_layer_indices: List[int] = [6, 9, 12, 15],
	model_type: str = "dtw",
	weighter_step: int = None
) -> MultiLayerReranker:
	"""
	Load the trained reranker model from checkpoint

	Args:
		checkpoint_path: Path to model checkpoint directory
		model_name: HuggingFace model identifier
		layer_indices: Layer indices for the model
		dtw_layer_indices: Layer indices used for DTW
		model_type: Which sub-model to use ("dtw" or "layer_<idx>")
		weighter_step: Specific step to load (None for latest)

	Returns:
		Loaded MultiLayerReranker model
	"""
	logger.info(f"Loading model from {checkpoint_path}")

	# Initialize the model
	device = get_device()
	model = MultiLayerReranker(
		model_name=model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		device=device
	)

	# Load model into memory
	model.load_model()

	# Determine model to load (DTW or specific layer)
	if model_type == "dtw":
		model_dir = f"{checkpoint_path}/dtw_model"
	elif model_type.startswith("layer_"):
		layer_idx = int(model_type.split("_")[1])
		model_dir = f"{checkpoint_path}/single_layer_{layer_idx}"
	else:
		raise ValueError(f"Unknown model_type: {model_type}")

	if not os.path.exists(model_dir):
		raise FileNotFoundError(f"Model directory not found: {model_dir}")

	# Find checkpoint file
	if weighter_step is None:
		# Find the latest checkpoint
		checkpoint_files = [f for f in os.listdir(model_dir) 
							if f.startswith("weighter_step_") and f.endswith(".pt")]
		if not checkpoint_files:
			raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

		latest_checkpoint = sorted(
			checkpoint_files,
			key=lambda x: int(x.split("_")[2].split(".")[0])
		)[-1]

		checkpoint_file = f"{model_dir}/{latest_checkpoint}"
		step = int(latest_checkpoint.split("_")[2].split(".")[0])
	else:
		# Use specified step
		checkpoint_file = f"{model_dir}/weighter_step_{weighter_step}.pt"
		step = weighter_step

		if not os.path.exists(checkpoint_file):
			raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

	logger.info(f"Loading weights from {checkpoint_file}")

	# Load the weights
	if model_type == "dtw":
		model.dtw_weighter.load_state_dict(
			torch.load(checkpoint_file, map_location=device)
		)
	else:
		model.layer_weighters[model_type].load_state_dict(
			torch.load(checkpoint_file, map_location=device)
		)

	logger.info(f"Successfully loaded model (step {step})")
	return model

def compute_mrr(qrels: Dict[int, List[int]], rankings: Dict[int, List[int]], cutoff: int = 10) -> Dict[str, float]:
	"""
	Compute Mean Reciprocal Rank (MRR) at cutoff

	Args:
		qrels: Dictionary mapping query IDs to relevant passage IDs
		rankings: Dictionary mapping query IDs to ranked passage IDs
		cutoff: Maximum rank to consider (e.g., 10 for MRR@10)

	Returns:
		Dictionary of metric values
	"""
	mrr = 0.0
	relevant_query_count = 0
	all_scores = {}

	# Track rank positions for relevant documents found
	relevant_ranks = []

	for qid in rankings:
		if qid in qrels:
			relevant_pids = set(qrels[qid])
			ranked_pids = rankings[qid][:cutoff]  # Consider only up to cutoff

			# Check if we find a relevant document in the top ranks
			for rank, pid in enumerate(ranked_pids, 1):
				if pid in relevant_pids:
					mrr += 1.0 / rank
					relevant_ranks.append(rank)
					break

			relevant_query_count += 1

	# Compute MRR
	if relevant_query_count > 0:
		mrr = mrr / relevant_query_count
	else:
		logger.warning("No queries with relevant documents found.")
		mrr = 0.0

	# Prepare results
	all_scores['MRR @10'] = mrr
	all_scores['Queries Evaluated'] = relevant_query_count

	if relevant_ranks:
		all_scores['Queries with Relevant @10'] = len(relevant_ranks)
		all_scores['Mean Rank @10'] = sum(relevant_ranks) / len(relevant_ranks)
		all_scores['Median Rank @10'] = sorted(relevant_ranks)[len(relevant_ranks)//2]

	return all_scores

def rerank_queries(
	model: MultiLayerReranker,
	queries: Dict[int, str],
	passages: Dict[int, str],
	initial_rankings: Dict[int, List[int]],
	model_type: str = "dtw",
	batch_size: int = 32,
	max_queries: Optional[int] = None
) -> Dict[int, List[int]]:
	"""
	Rerank passages for each query using our model

	Args:
		model: MultiLayerReranker model
		queries: Dictionary mapping query IDs to query text
		passages: Dictionary mapping passage IDs to passage text
		initial_rankings: Dictionary mapping query IDs to ranked passage IDs
		model_type: Model type to use for reranking
		batch_size: Batch size for processing documents
		max_queries: Maximum number of queries to process (for testing)

	Returns:
		Dictionary mapping query IDs to reranked passage IDs
	"""
	reranked_results = {}

	# Limit queries for testing if specified
	query_ids = list(initial_rankings.keys())
	if max_queries is not None and max_queries < len(query_ids):
		query_ids = query_ids[:max_queries]

	logger.info(f"Reranking passages for {len(query_ids)} queries using {model_type} model")

	# Process each query
	for qid in tqdm(query_ids, desc="Reranking queries"):
		if qid not in queries:
			logger.warning(f"Query ID {qid} not found in queries. Skipping.")
			continue

		query_text = queries[qid]
		passage_ids = initial_rankings[qid]

		# Get passage texts
		passage_texts = []
		valid_passage_ids = []

		for pid in passage_ids:
			if pid in passages:
				passage_texts.append(passages[pid])
				valid_passage_ids.append(pid)
			else:
				logger.warning(f"Passage ID {pid} not found in collection.")

		if not passage_texts:
			logger.warning(f"No valid passages found for query {qid}. Skipping.")
			continue

		# Rerank passages
		try:
			reranked_indices = model.rerank(
				query=query_text,
				documents=passage_texts,
				model_type=model_type,
				batch_size=batch_size,
				return_scores=False
			)

			# Map indices back to passage IDs
			reranked_passage_ids = [valid_passage_ids[idx] for idx in reranked_indices]
			reranked_results[qid] = reranked_passage_ids

		except Exception as e:
			logger.error(f"Error reranking query {qid}: {e}")
			# Use original ranking as fallback
			reranked_results[qid] = valid_passage_ids

	return reranked_results

def save_run_file(rankings: Dict[int, List[int]], output_file: str):
	"""
	Save rankings to MSMARCO format run file

	Args:
		rankings: Dictionary mapping query IDs to ranked passage IDs
		output_file: Path to output file
	"""
	logger.info(f"Saving run file to {output_file}")

	with open(output_file, 'w') as f:
		for qid, passage_ids in rankings.items():
			for rank, pid in enumerate(passage_ids, 1):
				f.write(f"{qid}\t{pid}\t{rank}\n")

	logger.info(f"Saved rankings for {len(rankings)} queries")

def main():
	parser = argparse.ArgumentParser(description="Evaluate reranker models on MSMARCO")

	# Data paths
	parser.add_argument("--queries", required=True, help="Path to queries file")
	parser.add_argument("--collection", required=True, help="Path to collection file")
	parser.add_argument("--qrels", required=True, help="Path to qrels file")
	parser.add_argument("--run", required=True, help="Path to initial rankings file")

	# Model configuration
	parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint directory")
	parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf", help="HuggingFace model identifier")
	parser.add_argument("--layers", default="0,3,6,9,12,15,18,21", help="Comma-separated layer indices")
	parser.add_argument("--dtw_layers", default="6,9,12,15", help="Comma-separated DTW layer indices")
	parser.add_argument("--model_type", default="dtw", help="Model type (dtw or layer_<idx>)")
	parser.add_argument("--weighter_step", type=int, default=None, help="Specific step to load (default: latest)")

	# Evaluation parameters
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for passage encoding")
	parser.add_argument("--output_run", default="reranked.run", help="Path to output run file")
	parser.add_argument("--max_queries", type=int, default=None, help="Maximum queries to evaluate (for testing)")

	args = parser.parse_args()

	# Convert string lists to integer lists
	layer_indices = [int(x) for x in args.layers.split(',')]
	dtw_layer_indices = [int(x) for x in args.dtw_layers.split(',')]

	# Load queries, collection, and initial rankings
	start_time = time.time()
	queries = load_queries(args.queries)
	passages = load_collection(args.collection)
	initial_rankings = load_top1000(args.run)
	qrels = load_qrels(args.qrels)
	load_time = time.time() - start_time
	logger.info(f"Data loading completed in {load_time:.2f} seconds")

	# Load the model
	start_time = time.time()
	model = load_model(
		checkpoint_path=args.checkpoint_path,
		model_name=args.model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		model_type=args.model_type,
		weighter_step=args.weighter_step
	)
	model_load_time = time.time() - start_time
	logger.info(f"Model loading completed in {model_load_time:.2f} seconds")

	# Rerank passages
	start_time = time.time()
	reranked_results = rerank_queries(
		model=model,
		queries=queries,
		passages=passages,
		initial_rankings=initial_rankings,
		model_type=args.model_type,
		batch_size=args.batch_size,
		max_queries=args.max_queries
	)
	rerank_time = time.time() - start_time
	logger.info(f"Reranking completed in {rerank_time:.2f} seconds")

	# Save the run file
	save_run_file(reranked_results, args.output_run)

	# Compute evaluation metrics
	metrics = compute_mrr(qrels, reranked_results)

	# Print results
	print("\n===== Evaluation Results =====")
	print(f"Model: {args.model_type} from {args.checkpoint_path}")
	print(f"Step: {args.weighter_step if args.weighter_step else 'latest'}")
	print("===== Metrics =====")
	for metric, value in metrics.items():
		print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

	# Compare with initial ranking
	initial_metrics = compute_mrr(qrels, initial_rankings)
	print("\n===== Initial Ranking Metrics =====")
	for metric, value in initial_metrics.items():
		print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

	# Calculate and show improvement
	if 'MRR @10' in metrics and 'MRR @10' in initial_metrics:
		improvement = metrics['MRR @10'] - initial_metrics['MRR @10']
		improvement_percent = (improvement / initial_metrics['MRR @10']) * 100 if initial_metrics['MRR @10'] > 0 else 0
		print(f"\nMRR@10 Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")

	# Unload the model to free memory
	model.unload_model()

	logger.info("Evaluation completed")

if __name__ == "__main__":
	main()