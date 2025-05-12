import os
import random
import logging
import sys
from typing import Iterator, Tuple, List
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('msmarco_loader')

class MSMARCOTripleLoader:
	"""Efficient loader for MS MARCO triples with additional negative examples."""

	def __init__(
		self, 
		triples_path: str,
		additional_negatives_per_query: int = 4,
		batch_size: int = 32,
		cache_size: int = 10000,
		seed: int = 42
	):
		"""
		Initialize the MS MARCO triples loader.

		Args:
			triples_path: Path to the triples.train.small.tsv file
			additional_negatives_per_query: Number of additional negative examples to add per query
			batch_size: Number of examples per batch
			cache_size: Size of the query-passage pair cache for generating additional negatives
			seed: Random seed for reproducibility
		"""
		self.triples_path = triples_path
		self.additional_negatives = additional_negatives_per_query
		self.batch_size = batch_size
		self.cache_size = cache_size

		# Set random seed
		random.seed(seed)

		# Cache for query-specific data
		self.query_cache = {}  # {query_text: {'positives': set(), 'negatives': set(), 'all_passages': list()}}
		self.passage_pool = []  # Pool of passages for sampling additional negatives

		# Verify the triples file exists
		if not os.path.exists(self.triples_path):
			raise FileNotFoundError(f"Triples file not found: {self.triples_path}")

		# Initialize cached data for efficient negative sampling
		self._initialize_caches()

	def _initialize_caches(self):
		"""Initialize caches by scanning a portion of the triples file."""
		logger.info(f"Initializing caches from {self.triples_path}")

		# We'll scan a portion of the file to build initial caches
		lines_to_scan = min(self.cache_size * 10, 1000000)  # Scan up to 1M lines or 10x cache_size

		queries_seen = set()
		unique_passages = set()

		with open(self.triples_path, 'r', encoding='utf-8', errors='ignore') as f:
			for i, line in enumerate(tqdm(f, total=lines_to_scan, desc="Initializing cache")):
				if i >= lines_to_scan:
					break

				try:
					parts = line.strip().split('\t')
					if len(parts) >= 3:
						query, pos_passage, neg_passage = parts[0], parts[1], parts[2]

						# Add to query cache if this is a new query
						if query not in self.query_cache and len(self.query_cache) < self.cache_size:
							self.query_cache[query] = {
								'positives': set([pos_passage]),
								'negatives': set([neg_passage]),
								'all_passages': [pos_passage, neg_passage]
							}
						elif query in self.query_cache:
							self.query_cache[query]['positives'].add(pos_passage)
							self.query_cache[query]['negatives'].add(neg_passage)
							self.query_cache[query]['all_passages'].extend([pos_passage, neg_passage])

						queries_seen.add(query)

						# Add to passage pool for random sampling
						if len(self.passage_pool) < self.cache_size and pos_passage not in unique_passages:
							self.passage_pool.append(pos_passage)
							unique_passages.add(pos_passage)

						if len(self.passage_pool) < self.cache_size and neg_passage not in unique_passages:
							self.passage_pool.append(neg_passage)
							unique_passages.add(neg_passage)

				except Exception as e:
					logger.warning(f"Error processing line {i}: {e}")
					continue

		logger.info(f"Cached {len(self.query_cache)} queries and {len(self.passage_pool)} unique passages")
		logger.info(f"Total unique queries seen: {len(queries_seen)}")

	def _get_additional_negatives(self, query: str, num_negatives: int) -> List[str]:
		"""
		Get additional negative passages for a query.

		Args:
			query: Query text
			num_negatives: Number of negative passages to retrieve

		Returns:
			List of negative passage texts
		"""
		negatives = []

		# If query is in cache, avoid passages already associated with this query
		avoid_passages = set()
		if query in self.query_cache:
			avoid_passages = set(self.query_cache[query]['all_passages'])

		# Try to sample from passage pool
		available_passages = [p for p in self.passage_pool if p not in avoid_passages]

		# If we don't have enough passages in the pool, use what we have
		if len(available_passages) < num_negatives:
			samples = available_passages
		else:
			samples = random.sample(available_passages, num_negatives)

		negatives.extend(samples)

		# If we still need more negatives, generate some from other queries' positives
		if len(negatives) < num_negatives and len(self.query_cache) > 1:
			other_queries = [q for q in self.query_cache if q != query]
			if other_queries:
				for _ in range(num_negatives - len(negatives)):
					other_query = random.choice(other_queries)
					if self.query_cache[other_query]['positives']:
						random_passage = random.choice(list(self.query_cache[other_query]['positives']))
						negatives.append(random_passage)

					# Break if we've reached enough negatives
					if len(negatives) >= num_negatives:
						break

		return negatives[:num_negatives]  # Ensure we return exactly num_negatives

	def stream_training_data(self) -> Iterator[List[Tuple[str, str, int]]]:
		"""
		Stream training data from the triples file with additional negatives.

		Yields:
			Batches of (query_text, passage_text, label) tuples
		"""
		logger.info(f"Streaming data from {self.triples_path} with {self.additional_negatives} additional negatives per query")

		current_batch = []

		with open(self.triples_path, 'r', encoding='utf-8', errors='ignore') as f:
			for line_num, line in enumerate(f):
				try:
					parts = line.strip().split('\t')
					if len(parts) < 3:
						continue

					query, pos_passage, neg_passage = parts[0], parts[1], parts[2]

					# 1. Add the original positive example
					current_batch.append((query, pos_passage, 1))

					# 2. Add the original negative example
					current_batch.append((query, neg_passage, 0))

					# 3. Add additional negative examples
					if self.additional_negatives > 0:
						additional_negs = self._get_additional_negatives(query, self.additional_negatives)
						for neg in additional_negs:
							current_batch.append((query, neg, 0))

					# Update the cache with this query and passages if not already there
					if query not in self.query_cache and len(self.query_cache) < self.cache_size:
						self.query_cache[query] = {
							'positives': set([pos_passage]),
							'negatives': set([neg_passage]),
							'all_passages': [pos_passage, neg_passage]
						}
					elif query in self.query_cache:
						self.query_cache[query]['positives'].add(pos_passage)
						self.query_cache[query]['negatives'].add(neg_passage)

						# Only add to all_passages if it's not already too large
						if len(self.query_cache[query]['all_passages']) < 100:  # Limit per query
							self.query_cache[query]['all_passages'].extend([pos_passage, neg_passage])

					# Check if we have a full batch
					if len(current_batch) >= self.batch_size:
						yield current_batch
						current_batch = []

					# Occasionally update the passage pool with new passages
					if line_num % 10000 == 0 and line_num > 0:
						# Randomly replace some passages in the pool
						if self.passage_pool:
							replace_count = min(10, len(self.passage_pool))
							for _ in range(replace_count):
								idx = random.randrange(len(self.passage_pool))
								self.passage_pool[idx] = pos_passage if random.random() < 0.5 else neg_passage

						# Log progress
						logger.info(f"Processed {line_num} lines")

				except Exception as e:
					logger.warning(f"Error processing line {line_num}: {e}")
					continue

		# Yield any remaining examples
		if current_batch:
			yield current_batch

	# Ignore this for acutal use it's part of the demo only
	def get_sample_batch(self, sample_size=3) -> List[Tuple[str, str, int]]:
		"""Get a sample batch for inspection."""
		with open(self.triples_path, 'r', encoding='utf-8', errors='ignore') as f:
			batch = []
			for _ in range(sample_size):
				line = f.readline().strip()
				parts = line.split('\t')
				if len(parts) >= 3:
					query, pos_passage, neg_passage = parts[0], parts[1], parts[2]

					# Add positive example
					batch.append((query, pos_passage, 1))

					# Add original negative example
					batch.append((query, neg_passage, 0))

					# Add additional negative examples
					additional_negs = self._get_additional_negatives(query, self.additional_negatives)
					for neg in additional_negs:
						batch.append((query, neg, 0))

			return batch

# Example usage
if __name__ == "__main__":
	triples_path = "./triples.train.small.tsv"  # Update to your path

	# Initialize the loader with 3 additional negatives per query
	loader = MSMARCOTripleLoader(
		triples_path=triples_path,
		additional_negatives_per_query=3,
		batch_size=32,
		cache_size=10000
	)

	# Print a sample batch for inspection
	sample_batch = loader.get_sample_batch(sample_size=3)

	print(f"\nSample batch with additional negatives:")
	for i, (query, passage, label) in enumerate(sample_batch):
		print(f"Example {i+1}:")
		print(f"  Query: '{query[:50]}...' ({len(query)} chars)")
		print(f"  Passage: '{passage[:50]}...' ({len(passage)} chars)")
		print(f"  Label: {label}")
		print()

	# Demonstrate streaming (just process a few batches)
	print("\nDemonstrating streaming with additional negatives:")

	start_time = time.time()
	batch_count = 0
	example_count = 0
	pos_count = 0
	neg_count = 0

	for batch in loader.stream_training_data():
		batch_count += 1
		example_count += len(batch)

		# Count positive and negative examples
		batch_pos = sum(1 for _, _, label in batch if label == 1)
		batch_neg = len(batch) - batch_pos

		pos_count += batch_pos
		neg_count += batch_neg

		# Print batch composition
		print(f"Batch {batch_count}: {len(batch)} examples, {batch_pos} positive, {batch_neg} negative")

		# Print a sample from the first batch
		if batch_count == 1:
			print("\nSample from first batch:")
			for i, (query, passage, label) in enumerate(batch[:5]):
				print(f"  {i+1}. Query: '{query[:40]}...', Label: {label}")

		# Just check a few batches
		if batch_count >= 5:
			break

	elapsed = time.time() - start_time
	print(f"\nProcessed {batch_count} batches with {example_count} total examples in {elapsed:.2f} seconds")
	print(f"Positive ratio: {pos_count/example_count:.3f} ({pos_count}/{example_count})")
	print(f"Approximate stream rate: {example_count/elapsed:.1f} examples/second")