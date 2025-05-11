"""
Evaluation Script for MultiLayerReranker with BEIR

Example Commands:

# 1. Index Building (one-time preparation)
# Build and save BM25 index for a dataset
python evaluate_with_beir.py --dataset msmarco --split dev --index_only --bm25_index ./datasets/msmarco_bm25_index.pkl

# 2. Initial Retrieval with BM25 (one-time preparation)
# Retrieve documents with BM25 and save results
python evaluate_with_beir.py --dataset nfcorpus --split test --retrieve_only --bm25_index ./datasets/nfcorpus_bm25_index.pkl --bm25_results ./results/nfcorpus_bm25_results.tsv

# 3. Evaluate DTW Model
# Using pre-computed BM25 results for initial ranking
python evaluate_with_beir.py --dataset quora --split dev --initial_ranking_file ./results/quora_bm25_results.tsv --checkpoint_dir ./model_checkpoints/checkpoints --model_type dtw --output_run ./results/quora_dtw_reranked.run --output_metrics ./results/quora_dtw_metrics.json

# 4. Evaluate Specific Layer Model
# Evaluate layer 15 which often performs well
python evaluate_with_beir.py --dataset trec-covid --split test --initial_ranking_file ./results/trec-covid_bm25_results.tsv --checkpoint_dir ./model_checkpoints/checkpoints --model_type layer_15 --weighter_step 1000 --output_run ./results/trec-covid_layer15_reranked.run

# 5. Limited Evaluation for Testing
# Process only 10 queries for quick testing
python evaluate_with_beir.py --dataset scifact --split test --initial_ranking_file ./results/scifact_bm25_results.tsv --checkpoint_dir ./model_checkpoints/checkpoints --model_type dtw --max_queries 10

# 6. BM25 Retrieval and Immediate Reranking
# One-step process (compute BM25 and then rerank)
python evaluate_with_beir.py --dataset nq --split test --use_bm25 --bm25_index ./datasets/nq_bm25_index.pkl --checkpoint_dir ./model_checkpoints/checkpoints --model_type layer_12 --output_run ./results/nq_layer12_reranked.run

# 7. Evaluate Using Specific Checkpoint
# Use weights from a specific training step
python evaluate_with_beir.py --dataset arguana --split test --initial_ranking_file ./results/arguana_bm25_results.tsv --checkpoint_dir ./model_checkpoints/checkpoints --model_type dtw --weighter_step 2000 --output_metrics ./results/arguana_dtw_step2000_metrics.json
"""


import argparse
import logging
import os
import multiprocessing
import concurrent.futures
import threading
from functools import partial
import torch
import pathlib
from typing import Dict, List, Optional, Union
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

def check_model_paths(checkpoint_dir, model_type, list_available=True):
	"""
	Check if the model path exists and list available models if requested

	Args:
		checkpoint_dir: Base directory containing model checkpoints
		model_type: Which model to load - 'dtw' or 'layer_X' where X is a layer index
		list_available: Whether to list available models if the requested one isn't found

	Returns:
		Tuple of (model_exists, model_dir)
	"""
	if model_type == "dtw":
		model_dir = os.path.join(checkpoint_dir, "dtw_model")
	elif model_type.startswith("layer_"):
		layer_idx = int(model_type.split("_")[1])
		model_dir = os.path.join(checkpoint_dir, f"single_layer_{layer_idx}")
	else:
		logging.error(f"Unknown model_type: {model_type}. Use 'dtw' or 'layer_X' where X is a layer index.")
		return False, None

	if not os.path.exists(model_dir):
		logging.error(f"Model directory not found: {model_dir}")

		if list_available and os.path.exists(checkpoint_dir):
			available_models = []
			# Check for dtw model
			if os.path.exists(os.path.join(checkpoint_dir, "dtw_model")):
				available_models.append("dtw")

			# Check for layer models
			for item in os.listdir(checkpoint_dir):
				if item.startswith("single_layer_"):
					try:
						layer_num = int(item.split("_")[-1])
						available_models.append(f"layer_{layer_num}")
					except ValueError:
						continue

			if available_models:
				logging.info(f"Available models in {checkpoint_dir}:")
				for model in sorted(available_models):
					logging.info(f"  - {model}")
			else:
				logging.info(f"No models found in {checkpoint_dir}")

		return False, model_dir

	return True, model_dir

def load_model(
	checkpoint_dir: str,
	model_name: str = "meta-llama/Llama-2-7b-hf",
	layer_indices: List[int] = [0, 3, 6, 9, 12, 15, 18, 21],
	dtw_layer_indices: List[int] = [6, 9, 12, 15],
	model_type: str = "dtw",
	weighter_step: Optional[int] = None
):
	"""
	Load the MultiLayerReranker model with trained weights

	Args:
		checkpoint_dir: Base directory containing model checkpoints
					   (should contain subdirectories like 'dtw_model', 'single_layer_X', etc.)
		model_name: Name of the base LLM model
		layer_indices: Which layers to use in the reranker
		dtw_layer_indices: Which layers to use for DTW
		model_type: Which model to load - 'dtw' or 'layer_X' where X is a layer index
		weighter_step: Specific checkpoint step to load (None for latest)

	Returns:
		Loaded MultiLayerReranker model or None if loading fails
	"""
	# First check if the model directory exists
	model_exists, model_dir = check_model_paths(checkpoint_dir, model_type)
	if not model_exists:
		return None

	# Import here to avoid import errors if user only wants to run BM25
	try:
		from multi_layer_reranker import MultiLayerReranker
	except ImportError:
		logging.error("Failed to import MultiLayerReranker. Make sure the model code is in your Python path.")
		return None

	device = get_device()
	logging.info(f"Using device: {device}")

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

	# Find checkpoint file - either specific step or latest
	if weighter_step is None:
		# Find the latest checkpoint
		checkpoint_files = [f for f in os.listdir(model_dir) 
							if f.startswith("weighter_step_") and f.endswith(".pt")]
		if not checkpoint_files:
			logging.error(f"No checkpoint files found in {model_dir}")
			return None

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
			logging.error(f"Checkpoint not found: {checkpoint_file}")
			available_checkpoints = [f for f in os.listdir(model_dir) 
								if f.startswith("weighter_step_") and f.endswith(".pt")]
			if available_checkpoints:
				available_steps = sorted([int(f.split("_")[2].split(".")[0]) for f in available_checkpoints])
				logging.info(f"Available checkpoint steps: {available_steps}")
			return None

	logging.info(f"Loading weights from {checkpoint_file}")

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
		logging.error(f"Error loading model weights: {e}")
		return None

	# Additional log with model info
	logging.info(f"Model configuration: {model_type} model using {model_name}")
	if model_type == "dtw":
		logging.info(f"DTW using layers: {dtw_layer_indices}")
	else:
		logging.info(f"Using layer {model_type.split('_')[1]}")

	model.eval()  # Set to evaluation mode
	print_gpu_utilization()  # Check GPU usage after loading
	return model

def rerank_with_multi_layer_reranker(
	model,
	corpus: Dict[str, Dict[str, str]],
	queries: Dict[str, str],
	results: Dict[str, Dict[str, float]],
	model_type: str = "dtw",
	top_k: int = 100,
	batch_size: int = 32,
	show_progress: bool = True
) -> Dict[str, Dict[str, float]]:
	"""
	Rerank the initial retrieval results using the MultiLayerReranker

	Args:
		model: Loaded MultiLayerReranker model
		corpus: Dictionary mapping doc_id to document data
		queries: Dictionary mapping query_id to query text
		results: Initial retrieval results {query_id: {doc_id: score}}
		model_type: Which model type to use for scoring ("dtw" or "layer_X")
		top_k: How many documents to rerank per query
		batch_size: Batch size for document encoding
		show_progress: Whether to show progress bar

	Returns:
		Dictionary mapping query_id to reranked {doc_id: score}
	"""
	rerank_results = {}
	logging.info(f"Reranking top-{top_k} documents using {model_type} model")

	# Start timing
	start_time = time.time()
	total_queries = len(results)
	processed_queries = 0

	# Setup progress bar if requested
	if show_progress:
		pbar = tqdm(results.items(), desc="Reranking queries")
	else:
		pbar = results.items()

	for query_id, doc_scores in pbar:
		query_text = queries[query_id]

		# Get top-k doc_ids from initial retrieval
		sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
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
			rerank_results[query_id] = {}
			continue

		# Use the model's rerank method to get scores
		rankings = model.rerank(
			query=query_text,
			documents=candidate_docs,
			model_type=model_type,
			batch_size=batch_size,
			return_scores=True
		)

		# Extract rankings for the specified model_type
		if model_type in rankings:
			sorted_indices, sorted_scores = rankings[model_type]

			# Map scores back to doc_ids
			doc_scores = {}
			for idx, score in zip(sorted_indices, sorted_scores):
				doc_id = valid_doc_ids[idx]
				doc_scores[doc_id] = score

			rerank_results[query_id] = doc_scores
		else:
			logging.warning(f"Model type '{model_type}' not found in rankings output")
			rerank_results[query_id] = {}

		# Update processed queries counter
		processed_queries += 1

		# Print GPU usage every 10 queries
		if processed_queries % 10 == 0:
			print_gpu_utilization()

			# Calculate ETA
			elapsed = time.time() - start_time
			queries_per_sec = processed_queries / elapsed
			remaining_queries = total_queries - processed_queries
			eta_seconds = remaining_queries / queries_per_sec if queries_per_sec > 0 else 0

			logging.info(f"Processed {processed_queries}/{total_queries} queries "
						 f"({queries_per_sec:.2f} queries/sec, "
						 f"ETA: {timedelta(seconds=int(eta_seconds))})")

	# Final timing
	total_time = time.time() - start_time
	logging.info(f"Reranking completed in {timedelta(seconds=int(total_time))}, "
				 f"average {processed_queries/total_time:.2f} queries/sec")

	return rerank_results

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
		logging.info("Building BM25 index...")
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
		logging.info("Computing document frequencies...")
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

		logging.info(f"Indexed {doc_count} documents with {len(doc_freqs)} unique terms")

	def save_index(self, path):
		"""Save the index to disk."""
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
		logging.info(f"Saved BM25 index to {path}")

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

		logging.info(f"Loaded BM25 index with {self.doc_count} documents and {len(self.doc_freqs)} terms")

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

	def batch_search(self, queries, top_k=100):
		"""
		Search multiple queries in a batch

		Args:
			queries: Dictionary {query_id: query_text}
			top_k: Number of top documents to retrieve

		Returns:
			Dictionary {query_id: {doc_id: score}}
		"""
		results = {}
		for query_id, query_text in queries.items():
			results[query_id] = self.search(query_text, top_k=top_k)
		return results

def _process_bm25_batch(batch_queries, bm25_index, top_k):
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

def retrieve_with_bm25(corpus, queries, index_path=None, top_k=100, max_queries=None, 
					  query_ids=None, num_workers=None):
	"""
	Retrieve documents using Simple BM25 with parallel processing

	Args:
		corpus: Dictionary mapping doc_id to document data
		queries: Dictionary mapping query_id to query text
		index_path: Path to save/load the BM25 index
		top_k: Number of top documents to retrieve
		max_queries: Maximum number of queries to process (None for all)
		query_ids: Specific query IDs to process (None for random selection if max_queries is set)
		num_workers: Number of parallel workers (None for auto-detection)

	Returns:
		Dictionary mapping query_id to {doc_id: score}
	"""
	# Create BM25 indexer
	bm25 = SimpleBM25()

	# Try to load existing index
	if index_path and os.path.exists(index_path):
		logging.info(f"Loading BM25 index from {index_path}")
		bm25.load_index(index_path)
	else:
		# Build new index
		bm25.index_corpus(corpus)
		if index_path:
			bm25.save_index(index_path)

	# Select queries to process
	if query_ids is not None:
		# Use specific query IDs
		valid_qids = set(queries.keys())
		query_ids = [qid for qid in query_ids if qid in valid_qids]
		if not query_ids:
			logging.warning("None of the specified query IDs were found in the dataset")
			return {}
		limited_queries = {qid: queries[qid] for qid in query_ids}
		logging.info(f"Processing {len(limited_queries)} specified queries")
	elif max_queries is not None and max_queries < len(queries):
		# Randomly select queries
		logging.info(f"Limiting to {max_queries} of {len(queries)} queries")
		query_ids = list(queries.keys())
		if max_queries < len(query_ids):
			query_ids = random.sample(query_ids, max_queries)
		limited_queries = {qid: queries[qid] for qid in query_ids}
	else:
		# Process all queries
		limited_queries = queries

	# Determine number of workers
	if num_workers is None:
		num_workers = min(multiprocessing.cpu_count(), 32)  # Cap at 32 processes
		# Leave some cores free on larger systems
		if num_workers > 8:
			num_workers -= 2

	# For tiny query sets, don't bother with multiprocessing
	if len(limited_queries) < 10:
		logging.info(f"Processing {len(limited_queries)} queries sequentially")
		results = {}
		for query_id, query_text in tqdm(limited_queries.items(), desc="BM25 retrieval"):
			results[query_id] = bm25.search(query_text, top_k=top_k)
		return results

	logging.info(f"BM25 retrieval using {num_workers} CPU cores in parallel")

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
				logging.error(f"Error in worker process: {e}")
				# Keep going with other batches

	pbar.close()
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

def main():
	parser = argparse.ArgumentParser(description="Evaluate MultiLayerReranker with BEIR")

	# Parallelization
	parser.add_argument("--workers", type=int, default=None,
						help="Number of CPU cores to use for BM25 retrieval (default: auto)")

	# Task mode
	mode_group = parser.add_mutually_exclusive_group()
	mode_group.add_argument("--index_only", action="store_true",
						help="Only build and save BM25 index, don't rerank")
	mode_group.add_argument("--retrieve_only", action="store_true",
						help="Only retrieve with BM25, don't rerank")
	mode_group.add_argument("--rerank_only", action="store_true",
						help="Only rerank using pre-computed results")

	# Dataset configuration
	parser.add_argument("--dataset", default="msmarco", 
						help="BEIR dataset name (e.g., msmarco, trec-covid, nfcorpus)")
	parser.add_argument("--split", default="dev", choices=["dev", "test"],
						help="Dataset split to use (dev or test)")
	parser.add_argument("--data_dir", default="./datasets",
						help="Directory to store/load datasets")

	# Retrieval configuration
	retrieval_group = parser.add_mutually_exclusive_group()
	retrieval_group.add_argument("--use_bm25", action="store_true",
					  help="Use BM25 for initial retrieval")
	retrieval_group.add_argument("--initial_ranking_file", type=str,
					  help="Path to pre-computed initial rankings file")

	# BM25 settings
	parser.add_argument("--bm25_index", type=str, default=None,
						help="Path to save/load BM25 index")
	parser.add_argument("--bm25_results", type=str, default=None,
						help="Path to save/load BM25 results")

	# Query limiting for faster testing
	parser.add_argument("--max_queries", type=int, default=None,
						help="Maximum number of queries to process (for testing)")
	parser.add_argument("--query_ids", type=str, default=None,
						help="Comma-separated list of specific query IDs to process")

	# Model configuration
	parser.add_argument("--checkpoint_dir", type=str, default=None,
						help="Base directory containing model checkpoints")
	parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf",
						help="HuggingFace model identifier")
	parser.add_argument("--layers", default="0,3,6,9,12,15,18,21",
						help="Comma-separated layer indices")
	parser.add_argument("--dtw_layers", default="6,9,12,15",
						help="Comma-separated DTW layer indices")
	parser.add_argument("--model_type", default="dtw", 
						help="Model type to evaluate ('dtw' or 'layer_X' where X is layer index)")
	parser.add_argument("--weighter_step", type=int, default=None,
						help="Specific checkpoint step to load (default: latest)")

	# Reranking parameters
	parser.add_argument("--top_k", type=int, default=100,
						help="Number of documents to rerank per query")
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size for document encoding")

	# Output options
	parser.add_argument("--output_run", type=str, default=None,
						help="Path to save reranked results in TREC format")
	parser.add_argument("--output_metrics", type=str, default=None,
						help="Path to save evaluation metrics in JSON format")

	args = parser.parse_args()

	# Validate arguments based on mode
	if args.index_only or args.retrieve_only:
		# For index-only or retrieve-only modes, BM25 must be used
		args.use_bm25 = True

		# For index-only mode, ensure we have a path to save the index
		if args.index_only and not args.bm25_index:
			args.bm25_index = os.path.join(args.data_dir, f"{args.dataset}_bm25_index.pkl")
			logging.info(f"Setting BM25 index path to {args.bm25_index}")

		# For retrieve-only mode, ensure we have paths for the index and results
		if args.retrieve_only:
			if not args.bm25_index:
				args.bm25_index = os.path.join(args.data_dir, f"{args.dataset}_bm25_index.pkl")
				logging.info(f"Setting BM25 index path to {args.bm25_index}")
			if not args.bm25_results:
				args.bm25_results = os.path.join(args.data_dir, f"{args.dataset}_bm25_results.tsv")
				logging.info(f"Setting BM25 results path to {args.bm25_results}")

	if args.rerank_only:
		# For rerank-only mode, need initial rankings
		if not args.initial_ranking_file and not args.bm25_results:
			logging.error("For --rerank_only mode, either --initial_ranking_file or --bm25_results must be specified")
			return

		# If no initial_ranking_file is specified, use bm25_results
		if not args.initial_ranking_file:
			args.initial_ranking_file = args.bm25_results

		# Ensure checkpoint_dir is provided
		if not args.checkpoint_dir:
			logging.error("For --rerank_only mode, --checkpoint_dir must be specified")
			return

	# If running normal mode (not index/retrieve/rerank only), validate required args
	if not (args.index_only or args.retrieve_only or args.rerank_only):
		if not (args.use_bm25 or args.initial_ranking_file):
			logging.error("Either --use_bm25 or --initial_ranking_file must be specified")
			return

		if not args.checkpoint_dir:
			logging.error("--checkpoint_dir must be specified for reranking")
			return

	# Process query IDs if specified
	query_id_list = None
	if args.query_ids:
		query_id_list = args.query_ids.split(',')
		logging.info(f"Processing specific query IDs: {query_id_list}")

	# Convert string lists to integer lists
	layer_indices = [int(x) for x in args.layers.split(',')]
	dtw_layer_indices = [int(x) for x in args.dtw_layers.split(',')]

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

	# 2. Index or retrieve with BM25 if requested
	if args.index_only:
		# Only build and save the BM25 index
		bm25 = SimpleBM25()
		bm25.index_corpus(corpus)
		bm25.save_index(args.bm25_index)
		logging.info(f"Built and saved BM25 index to {args.bm25_index}")
		return

	# 3. Execute BM25 retrieval if needed
	results = None

	if args.retrieve_only or args.use_bm25:
		# Use our simple BM25 implementation
		if args.bm25_index is None:
			args.bm25_index = os.path.join(args.data_dir, f"{args.dataset}_bm25_index.pkl")

		logging.info("Performing retrieval with Simple BM25")
		results = retrieve_with_bm25(corpus, queries, args.bm25_index, 
								   top_k=args.top_k, max_queries=args.max_queries,
								   query_ids=query_id_list, num_workers=args.workers)

		# Save BM25 results if requested or in retrieve-only mode
		if args.bm25_results or args.retrieve_only:
			if args.retrieve_only and not args.bm25_results:
				args.bm25_results = os.path.join(args.data_dir, f"{args.dataset}_bm25_results.tsv")
			save_results_file(results, args.bm25_results)

		# If we're only retrieving, we're done
		if args.retrieve_only:
			# Optionally evaluate BM25 results
			logging.info("Evaluating BM25 results")
			# Ensure we have qrels for all queries we're evaluating
			valid_query_ids = set(results.keys()) & set(qrels.keys())
			if len(valid_query_ids) < len(results):
				logging.warning(f"Only {len(valid_query_ids)} of {len(results)} queries have relevance judgments")
				filtered_results = {qid: results[qid] for qid in valid_query_ids}
			else:
				filtered_results = results

			ndcg, _map, recall, precision = evaluator.evaluate(qrels, filtered_results, evaluator.k_values)

			print("\n===== BM25 Retrieval Results =====")
			for k in evaluator.k_values:
				ndcg_key = f"NDCG@{k}"
				map_key = f"MAP@{k}"
				recall_key = f"Recall@{k}"
				precision_key = f"P@{k}"

				if ndcg_key in ndcg:
					print(f"NDCG@{k}: {ndcg[ndcg_key]:.4f}")
					print(f"MAP@{k}: {_map[map_key]:.4f}")
					print(f"Recall@{k}: {recall[recall_key]:.4f}")
					print(f"P@{k}: {precision[precision_key]:.4f}")
					print()

			return

	# 4. Load initial rankings from file if not using BM25
	if args.initial_ranking_file:
		logging.info(f"Loading initial rankings from {args.initial_ranking_file}")
		results = load_initial_rankings(args.initial_ranking_file)

		# Limit queries if specified
		if args.max_queries is not None and args.max_queries < len(results):
			logging.info(f"Limiting to {args.max_queries} of {len(results)} queries")
			if query_id_list:
				# Use specific query IDs
				valid_ids = set(results.keys()) & set(query_id_list)
				if not valid_ids:
					logging.error("None of the specified query IDs were found in the initial rankings")
					return
				results = {qid: results[qid] for qid in valid_ids}
			else:
				# Random selection
				query_ids = list(results.keys())
				query_ids = random.sample(query_ids, args.max_queries)
				results = {qid: results[qid] for qid in query_ids}

	# Ensure we have qrels for all queries we're evaluating
	valid_query_ids = set(results.keys()) & set(qrels.keys())
	if len(valid_query_ids) < len(results):
		logging.warning(f"Only {len(valid_query_ids)} of {len(results)} queries have relevance judgments")
		results = {qid: results[qid] for qid in valid_query_ids}

	# 5. Load the reranker model if needed
	if not args.index_only and not args.retrieve_only:
		model = load_model(
			checkpoint_dir=args.checkpoint_dir,
			model_name=args.model_name,
			layer_indices=layer_indices,
			dtw_layer_indices=dtw_layer_indices,
			model_type=args.model_type,
			weighter_step=args.weighter_step
		)

		if model is None:
			logging.error("Failed to load model, cannot continue with reranking")
			return

		# 6. Rerank the results
		rerank_results = rerank_with_multi_layer_reranker(
			model=model,
			corpus=corpus,
			queries=queries,
			results=results,
			model_type=args.model_type,
			top_k=args.top_k,
			batch_size=args.batch_size
		)

		# 7. Evaluate the results
		logging.info("Evaluating reranked results")
		ndcg, _map, recall, precision = evaluator.evaluate(qrels, rerank_results, evaluator.k_values)

		# Also calculate the metrics for the initial results to show improvement
		logging.info("Evaluating initial retrieval results")
		init_ndcg, init_map, init_recall, init_precision = evaluator.evaluate(qrels, results, evaluator.k_values)

		# Print improvement
		print("\n===== Evaluation Results =====")
		print(f"Model: {args.model_type} from {args.checkpoint_dir}")
		print(f"Step: {args.weighter_step if args.weighter_step else 'latest'}")
		print(f"Dataset: {args.dataset} ({args.split} split)")
		print(f"Queries evaluated: {len(rerank_results)}")
		print("\n===== Reranking Performance =====")

		# Collect metrics for potential JSON output
		all_metrics = {
			"config": {
				"dataset": args.dataset,
				"split": args.split,
				"model_type": args.model_type,
				"weighter_step": args.weighter_step,
				"queries_evaluated": len(rerank_results)
			},
			"metrics": {}
		}

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
				print(f"NDCG@{k}: {ndcg[ndcg_key]:.4f} (Initial: {init_ndcg[ndcg_key]:.4f}, Δ: {ndcg_improve:+.4f})")
				print(f"MAP@{k}: {_map[map_key]:.4f} (Initial: {init_map[map_key]:.4f}, Δ: {map_improve:+.4f})")
				print(f"Recall@{k}: {recall[recall_key]:.4f} (Initial: {init_recall[recall_key]:.4f}, Δ: {recall_improve:+.4f})")
				print(f"P@{k}: {precision[precision_key]:.4f} (Initial: {init_precision[precision_key]:.4f}, Δ: {precision_improve:+.4f})")
				print()

				# Store metrics
				all_metrics["metrics"][f"k={k}"] = {
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
			else:
				logging.warning(f"Key {ndcg_key} not found in evaluation metrics")
				print(f"NDCG@{k}: N/A")
				print(f"MAP@{k}: N/A")
				print(f"Recall@{k}: N/A")
				print(f"P@{k}: N/A")
				print()

		# 8. Save reranked results if requested
		if args.output_run:
			logging.info(f"Saving reranked results to {args.output_run}")
			save_results_file(rerank_results, args.output_run)

		# 9. Save metrics if requested
		if args.output_metrics:
			logging.info(f"Saving metrics to {args.output_metrics}")
			with open(args.output_metrics, 'w') as f:
				json.dump(all_metrics, f, indent=2)

		# Unload the model to free memory
		model.unload_model()

	logging.info("Evaluation completed")

if __name__ == "__main__":
	main()