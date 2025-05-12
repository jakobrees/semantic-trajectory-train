#!/usr/bin/env python
# quora_evaluation.py - with modifications for new weighting methods
"""
Evaluation script for Quora dataset with built-in BM25 retrieval
"""
import argparse
import logging
import os
import torch
import json
import pickle
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random
import numpy as np
import math
import concurrent.futures
import multiprocessing
from functools import partial
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

class SimpleBM25:
	"""A simplified BM25 implementation that doesn't require Elasticsearch."""
	def __init__(self, k1=1.5, b=0.75):
		self.k1 = k1
		self.b = b
		self.index = None
		self.doc_lengths = None
		self.avg_doc_length = None
		self.doc_freqs = None
		self.doc_count = None
		self.docid_to_idx = None
		self.idx_to_docid = None

	def index_corpus(self, corpus):
		"""Index the corpus for BM25 retrieval."""
		logger.info("Building BM25 index...")
		doc_count = len(corpus)
		# Map doc_ids to internal indices
		docid_to_idx = {}
		idx_to_docid = []
		for i, (doc_id, _) in enumerate(corpus.items()):
			docid_to_idx[doc_id] = i
			idx_to_docid.append(doc_id)

		# Process corpus in batches
		doc_tokens = []
		doc_lengths = np.zeros(doc_count, dtype=np.float32)
		for i, (doc_id, doc) in enumerate(tqdm(corpus.items(), desc="Tokenizing documents")):
			# Combine title and text if available
			if "title" in doc and doc["title"]:
				text = f"{doc['title']} {doc['text']}"
			else:
				text = doc["text"]

			# Simple tokenization
			tokens = [t.lower() for t in text.split()]
			doc_tokens.append(tokens)
			doc_lengths[docid_to_idx[doc_id]] = len(tokens)

		avg_doc_length = np.mean(doc_lengths)

		# Build document frequencies dictionary
		logger.info("Computing document frequencies...")
		doc_freqs = {}
		term_doc_freqs = {}
		for i, tokens in enumerate(tqdm(doc_tokens, desc="Computing document frequencies")):
			# Count terms only once per document
			terms = set(tokens)
			for term in terms:
				if term not in doc_freqs:
					doc_freqs[term] = 1
					term_doc_freqs[term] = {i: 1}
				else:
					doc_freqs[term] += 1
					term_doc_freqs[term][i] = 1

		# Store index
		self.index = term_doc_freqs
		self.doc_lengths = doc_lengths
		self.avg_doc_length = avg_doc_length
		self.doc_freqs = doc_freqs
		self.doc_count = doc_count
		self.docid_to_idx = docid_to_idx
		self.idx_to_docid = idx_to_docid

		logger.info(f"Indexed {doc_count} documents with {len(doc_freqs)} unique terms")

	def save_index(self, path):
		"""Save the index to disk."""
		os.makedirs(os.path.dirname(path), exist_ok=True)
		index_data = {
			'index': self.index,
			'doc_lengths': self.doc_lengths,
			'avg_doc_length': self.avg_doc_length,
			'doc_freqs': self.doc_freqs,
			'doc_count': self.doc_count,
			'docid_to_idx': self.docid_to_idx,
			'idx_to_docid': self.idx_to_docid,
			'k1': self.k1,
			'b': self.b
		}
		with open(path, 'wb') as f:
			pickle.dump(index_data, f)
		logger.info(f"Saved BM25 index to {path}")

	def load_index(self, path):
		"""Load the index from disk."""
		with open(path, 'rb') as f:
			index_data = pickle.load(f)
		self.index = index_data['index']
		self.doc_lengths = index_data['doc_lengths']
		self.avg_doc_length = index_data['avg_doc_length']
		self.doc_freqs = index_data['doc_freqs']
		self.doc_count = index_data['doc_count']
		self.docid_to_idx = index_data['docid_to_idx']
		self.idx_to_docid = index_data['idx_to_docid']
		self.k1 = index_data['k1']
		self.b = index_data['b']
		logger.info(f"Loaded BM25 index with {self.doc_count} documents and {len(self.doc_freqs)} terms")

	def search(self, query, top_k=100):
		"""Search the index for the query and return top_k documents."""
		if self.index is None:
			raise ValueError("Index not built. Call index_corpus first.")

		# Simple tokenization for query
		query_terms = [t.lower() for t in query.split()]

		# Calculate BM25 scores
		scores = np.zeros(self.doc_count)
		for term in query_terms:
			if term not in self.doc_freqs:
				continue

			df = self.doc_freqs[term]
			idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

			for doc_idx, freq in self.index[term].items():
				doc_len = self.doc_lengths[doc_idx]
				numerator = freq * (self.k1 + 1)
				denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
				scores[doc_idx] += idf * numerator / denominator

		# Get top_k document indices
		top_indices = np.argsort(-scores)[:top_k]

		# Map to document ids and scores
		results = {}
		for idx in top_indices:
			if scores[idx] > 0:  # Only include documents with non-zero scores
				doc_id = self.idx_to_docid[idx]
				results[doc_id] = float(scores[idx])

		return results

def _process_bm25_batch(batch_queries, bm25_index, top_k):
	"""
	Worker function to process a batch of queries with BM25

	Args:
		batch_queries: Dictionary mapping query_id to query_text
		bm25_index: Dictionary with BM25 index data
		top_k: Number of top documents to retrieve

	Returns:
		Dictionary mapping query_id to {doc_id: score}
	"""
	# Need to create a new SimpleBM25 instance in the worker
	bm25 = SimpleBM25()
	bm25.index = bm25_index['index']
	bm25.doc_lengths = bm25_index['doc_lengths']
	bm25.avg_doc_length = bm25_index['avg_doc_length']
	bm25.doc_freqs = bm25_index['doc_freqs']
	bm25.doc_count = bm25_index['doc_count']
	bm25.docid_to_idx = bm25_index['docid_to_idx']
	bm25.idx_to_docid = bm25_index['idx_to_docid']
	bm25.k1 = bm25_index['k1']
	bm25.b = bm25_index['b']

	batch_results = {}
	for qid, qtext in batch_queries.items():
		batch_results[qid] = bm25.search(qtext, top_k=top_k)

	return batch_results

def parallel_bm25_retrieval(queries, corpus, top_k=1000, index_path=None, max_queries=None, num_workers=None):
	"""
	Retrieve documents using parallel BM25 processing

	Args:
		queries: Dictionary mapping query_id to query text
		corpus: Dictionary mapping doc_id to document data
		top_k: Number of documents to retrieve per query
		index_path: Path to save/load BM25 index
		max_queries: Maximum number of queries to process (None for all)
		num_workers: Number of parallel workers (None for auto-detection)

	Returns:
		Dictionary mapping query_id to {doc_id: score}
	"""
	# Create BM25 indexer
	bm25 = SimpleBM25()

	# Try to load existing index
	if index_path and os.path.exists(index_path):
		logger.info(f"Loading BM25 index from {index_path}")
		bm25.load_index(index_path)
	else:
		# Build new index
		bm25.index_corpus(corpus)
		if index_path:
			os.makedirs(os.path.dirname(index_path), exist_ok=True)
			bm25.save_index(index_path)

	# Select queries to process
	if max_queries is not None and max_queries < len(queries):
		# Randomly select queries
		logger.info(f"Limiting to {max_queries} of {len(queries)} queries")
		query_ids = list(queries.keys())
		if max_queries < len(query_ids):
			query_ids = random.sample(query_ids, max_queries)
		limited_queries = {qid: queries[qid] for qid in query_ids}
	else:
		# Process all queries
		limited_queries = queries

	# Determine number of workers
	if num_workers is None:
		num_workers = min(multiprocessing.cpu_count(), 30)  # Use up to 30 cores
		# Leave a few cores free on larger systems
		if num_workers > 8:
			num_workers = 24  # Use 24 cores as requested

	logger.info(f"Using {num_workers} CPU cores for BM25 retrieval")

	# For tiny query sets, don't bother with multiprocessing
	if len(limited_queries) < 10:
		logger.info(f"Processing {len(limited_queries)} queries sequentially")
		results = {}
		for query_id, query_text in tqdm(limited_queries.items(), desc="BM25 retrieval"):
			results[query_id] = bm25.search(query_text, top_k=top_k)
		return results

	# Prepare index data for workers
	bm25_index_data = {
		'index': bm25.index,
		'doc_lengths': bm25.doc_lengths,
		'avg_doc_length': bm25.avg_doc_length,
		'doc_freqs': bm25.doc_freqs,
		'doc_count': bm25.doc_count,
		'docid_to_idx': bm25.docid_to_idx,
		'idx_to_docid': bm25.idx_to_docid,
		'k1': bm25.k1,
		'b': bm25.b
	}

	# Split queries into batches for each worker
	all_query_ids = list(limited_queries.keys())
	batch_size = max(1, len(all_query_ids) // num_workers)
	batches = []
	for i in range(0, len(all_query_ids), batch_size):
		batch_query_ids = all_query_ids[i:i+batch_size]
		batch = {qid: limited_queries[qid] for qid in batch_query_ids}
		batches.append(batch)

	logger.info(f"Distributed {len(limited_queries)} queries into {len(batches)} batches")

	# Initialize progress bar outside of parallelism
	pbar = tqdm(total=len(limited_queries), desc="BM25 retrieval")

	# Process batches in parallel
	results = {}
	with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
		# Create a partial function with the fixed arguments
		worker_fn = partial(_process_bm25_batch, bm25_index=bm25_index_data, top_k=top_k)

		# Submit all batches
		future_to_batch = {executor.submit(worker_fn, batch): i for i, batch in enumerate(batches)}

		# Process results as they complete
		for future in concurrent.futures.as_completed(future_to_batch):
			try:
				batch_results = future.result()
				results.update(batch_results)
				pbar.update(len(batch_results))
			except Exception as e:
				logger.error(f"Error in worker process: {e}")
				logger.error(f"Error details: {str(e)}")
				# Keep going with other batches

	pbar.close()
	logger.info(f"Completed BM25 retrieval for {len(results)} queries")
	return results

def load_or_create_bm25_results(queries, corpus, top_k=1000, output_file="./quora_bm25_results.tsv", 
							  index_file="./quora_bm25_index.pkl", force_new=False, max_queries=None, num_workers=24):
	"""
	Load existing BM25 results or create new ones if file doesn't exist

	Args:
		queries: Dictionary mapping query_id to query text
		corpus: Dictionary mapping doc_id to document data
		top_k: Number of documents to retrieve per query
		output_file: Path to save/load results
		index_file: Path to save/load BM25 index
		force_new: Whether to force creation of new results even if file exists
		max_queries: Maximum number of queries to process
		num_workers: Number of parallel workers

	Returns:
		Dictionary mapping query_id to {doc_id: score}
	"""
	# Check if results file already exists
	if os.path.exists(output_file) and not force_new:
		logger.info(f"Loading existing BM25 results from {output_file}")
		results = {}
		with open(output_file, 'r') as f:
			for line in tqdm(f, desc="Loading BM25 results"):
				parts = line.strip().split()
				if len(parts) >= 3:
					qid, pid = parts[0], parts[1]
					score = float(parts[3]) if len(parts) >= 4 else 1000.0 - float(parts[2])
					if qid not in results:
						results[qid] = {}
					results[qid][pid] = score
		logger.info(f"Loaded {len(results)} queries from BM25 results file")
		return results

	# Need to create new results
	logger.info("Running parallel BM25 retrieval with multiple processes...")

	results = parallel_bm25_retrieval(
		queries=queries,
		corpus=corpus,
		top_k=top_k,
		index_path=index_file,
		max_queries=max_queries,
		num_workers=num_workers
	)

	# Save results file
	logger.info(f"Saving BM25 results to {output_file}")
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	with open(output_file, 'w') as f:
		for qid, docs in results.items():
			sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
			for rank, (docid, score) in enumerate(sorted_docs, 1):
				f.write(f"{qid}\t{docid}\t{rank}\t{score:.6f}\n")

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

def load_model(checkpoint_dir, model_name, layer_indices, dtw_layer_indices, 
			   weighter_step=None, calc_surprise=True, pos_b=0.9, pos_c=1.0):
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

	# Initialize the model with new parameters
	model = MultiLayerReranker(
		model_name=model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		device=device,
		calc_surprise=calc_surprise,  # Enable surprise calculation
		pos_weighting_b=pos_b,  # Parameter for positional weighting
		pos_weighting_c=pos_c,  # Parameter for positional weighting
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

	# Initialize the query-only log frequency weighter
	model.initialize_token_weighter_with_frequencies(
		model.query_log_weighter,
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
	max_queries: Optional[int] = None
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

	Returns:
		Nested dictionary: {weighting_mode: {model_type: {query_id: {doc_id: score}}}}
	"""
	# Ensure queries in BM25 results exist in the queries dictionary
	# This handles possible mismatch between query IDs
	valid_query_ids = [qid for qid in results.keys() if qid in queries]

	if not valid_query_ids:
		logger.error("No queries in BM25 results match the loaded queries dictionary")
		# Return empty results
		empty_results = {}
		for mode in weighting_modes:
			empty_results[mode] = {}
			for model_type in available_model_types:
				empty_results[mode][model_type] = {}
		return empty_results

	logger.info(f"Found {len(valid_query_ids)} valid queries that match between BM25 results and queries dictionary")

	# Limit to max_queries if specified
	if max_queries is not None and max_queries < len(valid_query_ids):
		logger.info(f"Limiting to {max_queries} of {len(valid_query_ids)} queries")
		valid_query_ids = random.sample(valid_query_ids, max_queries)

	# Initialize nested results dictionaries for all weighting modes
	all_results = {}
	for mode in weighting_modes:
		all_results[mode] = {}
		for model_type in available_model_types:
			all_results[mode][model_type] = {}

	logger.info(f"Reranking top-{top_k} documents for {len(valid_query_ids)} queries")
	logger.info(f"Models: {', '.join(available_model_types)}")
	logger.info(f"Weighting modes: {', '.join(weighting_modes)}")

	# Start timing
	start_time = time.time()

	# Process each query
	for idx, query_id in enumerate(tqdm(valid_query_ids, desc="Reranking queries")):
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
			remaining_queries = len(valid_query_ids) - (idx + 1)
			eta_seconds = remaining_queries / queries_per_sec if queries_per_sec > 0 else 0

			logger.info(f"Processed {idx+1}/{len(valid_query_ids)} queries "
						 f"({queries_per_sec:.2f} queries/sec, "
						 f"ETA: {timedelta(seconds=int(eta_seconds))})")

	# Final timing
	total_time = time.time() - start_time
	if len(valid_query_ids) > 0:
		avg_queries_per_sec = len(valid_query_ids)/total_time
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
	parser = argparse.ArgumentParser(description="Evaluate Quora dataset with multiple weighting strategies")

	# Dataset configuration
	parser.add_argument("--data_dir", default="./datasets",
						help="Directory to store/load datasets")

	# BM25 configuration
	parser.add_argument("--bm25_top_k", type=int, default=1000,
						help="Number of documents to retrieve with BM25")
	parser.add_argument("--bm25_results", type=str, default="./results/quora_bm25_results.tsv",
						help="Path to save/load BM25 results")
	parser.add_argument("--bm25_index", type=str, default="./results/quora_bm25_index.pkl",
						help="Path to save/load BM25 index")
	parser.add_argument("--force_bm25", action="store_true",
						help="Force recomputation of BM25 results even if file exists")
	parser.add_argument("--bm25_workers", type=int, default=24,
						help="Number of worker processes for BM25 retrieval")

	# Query limiting for faster testing
	parser.add_argument("--max_queries", type=int, default=100,
						help="Maximum number of queries to process (for testing)")

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
	parser.add_argument("--weighting_modes", type=str, default="query_log_only,positional,surprise",
						help="Comma-separated list of weighting modes to evaluate")
	parser.add_argument("--log_weights_file", type=str, default=None,
						help="Path to token log weights file (for query_log_only mode)")

	# Parameters for position-based weighting
	parser.add_argument("--pos_b", type=float, default=0.9,
						help="Base parameter (b) for positional weighting formula")
	parser.add_argument("--pos_c", type=float, default=1.0,
						help="Coefficient (c) for positional weighting formula")

	# Reranking parameters
	parser.add_argument("--top_k", type=int, default=100,
						help="Number of documents to rerank per query")
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size for document encoding")

	# Surprise calculation
	parser.add_argument("--calc_surprise", action="store_true",
						help="Calculate token surprise values (for surprise weighting)")
	parser.add_argument("--surprise_scaling", type=float, default=1.0,
						help="Scaling factor for surprise values")

	# Output options
	parser.add_argument("--output_dir", type=str, default="./results/quora_weighting_comparison",
						help="Directory to save output files")
	parser.add_argument("--save_runs", action="store_true",
						help="Save reranked results for each model in TREC format")
	parser.add_argument("--save_metrics", action="store_true",
						help="Save evaluation metrics for each model in JSON format")

	args = parser.parse_args()

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

	# Ensure we have surprise calculation if surprise weighting is requested
	if "surprise" in weighting_modes and not args.calc_surprise:
		logger.warning("Surprise weighting requested but surprise calculation is disabled.")
		logger.warning("Enabling surprise calculation.")
		args.calc_surprise = True

	# 1. Load Quora dataset
	logger.info("Loading quora dataset (test split)")
	data_path = os.path.join(args.data_dir, "quora")

	# Download the dataset if not already present
	if not os.path.exists(data_path):
		url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip"
		data_path = util.download_and_unzip(url, args.data_dir)

	# Load the corpus, queries, and relevance judgments
	corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
	logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} qrels")

	# Set up evaluator for metrics calculation
	evaluator = EvaluateRetrieval(k_values=[1, 3, 5, 10, 20, 100])

	# 2. Load or compute BM25 results
	os.makedirs(os.path.dirname(args.bm25_results), exist_ok=True)
	os.makedirs(os.path.dirname(args.bm25_index), exist_ok=True)

	initial_results = load_or_create_bm25_results(
		queries=queries,
		corpus=corpus,
		top_k=args.bm25_top_k,
		output_file=args.bm25_results,
		index_file=args.bm25_index,
		force_new=args.force_bm25,
		max_queries=None,  # BM25 indexing uses all queries
		num_workers=args.bm25_workers
	)

	# Check if we have any queries with relevance judgments
	valid_query_ids = set(initial_results.keys()) & set(qrels.keys())
	logger.info(f"Found {len(valid_query_ids)} of {len(initial_results)} queries with relevance judgments")

	# 3. Load the model
	model, available_model_types = load_model(
		checkpoint_dir=args.checkpoint_dir,
		model_name=args.model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		weighter_step=args.weighter_step,
		calc_surprise=args.calc_surprise,
		pos_b=args.pos_b,
		pos_c=args.pos_c
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
	if "query_log_only" in weighting_modes and args.log_weights_file:
		if not initialize_with_log_weights(model, args.log_weights_file):
			logger.warning("Failed to initialize with log weights")
			# Remove query_log_only mode if initialization failed
			weighting_modes = [mode for mode in weighting_modes if mode != "query_log_only"]

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
		max_queries=args.max_queries
	)

	# 6. Evaluate and save results
	evaluate_and_save_results(
		all_results=all_reranked_results,
		evaluator=evaluator,
		initial_results=initial_results,
		qrels=qrels,
		output_dir=args.output_dir,
		dataset_name="quora",
		save_runs=args.save_runs,
		save_metrics=args.save_metrics
	)

	# Unload the model to free memory
	model.unload_model()
	logger.info("Evaluation completed")

if __name__ == "__main__":
	main()