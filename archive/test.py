import os
import logging
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import json

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Use a small dataset
dataset = "nfcorpus"
data_path = f"./datasets/{dataset}"

# Download if needed
if not os.path.exists(data_path):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, "./datasets")

# Load data
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")

# Create sample results (first 3 queries, top 10 docs)
sample_results = {}
sample_qids = list(qrels.keys())[:3]
for qid in sample_qids:
    sample_results[qid] = {}
    # Add 10 documents with descending scores
    for i, doc_id in enumerate(list(corpus.keys())[:10]):
        sample_results[qid][doc_id] = 10 - i

# Initialize evaluator
evaluator = EvaluateRetrieval(k_values=[1, 3, 5, 10, 20, 100])

# Evaluate sample results
logging.info("Running evaluation...")
ndcg, _map, recall, precision = evaluator.evaluate(qrels, sample_results, evaluator.k_values)

# Debug: Print the raw structure of result dictionaries
logging.info("\n===== RAW DATA STRUCTURES =====")
logging.info(f"NDCG type: {type(ndcg)}")
logging.info(f"NDCG keys: {list(ndcg.keys())}")
for k in ndcg.keys():
    logging.info(f"  Key type: {type(k)}, Value: {ndcg[k]}")

logging.info(f"\nMAP type: {type(_map)}")
logging.info(f"MAP keys: {list(_map.keys())}")

logging.info(f"\nRecall type: {type(recall)}")
logging.info(f"Recall keys: {list(recall.keys())}")

logging.info(f"\nPrecision type: {type(precision)}")
logging.info(f"Precision keys: {list(precision.keys())}")

# Debug: Try to serialize the dictionaries to understand what happens with JSON
try:
    serialized = json.dumps({
        'ndcg': {str(k): v for k, v in ndcg.items()},
        'map': {str(k): v for k, v in _map.items()},
        'recall': {str(k): v for k, v in recall.items()},
        'precision': {str(k): v for k, v in precision.items()}
    })
    logging.info("\nJSON serialization worked with explicit string conversion of keys")
except Exception as e:
    logging.error(f"JSON serialization failed: {e}")

# Try direct access with string keys to see if that's the issue
logging.info("\n===== TRYING DIFFERENT KEY FORMATS =====")
for k in evaluator.k_values:
    int_key_exists = k in ndcg
    str_key_exists = str(k) in ndcg
    logging.info(f"Key {k}: As int: {int_key_exists}, As string: {str_key_exists}")
    
    # Try accessing with both formats
    if int_key_exists:
        logging.info(f"  Value with int key: {ndcg[k]}")
    if str_key_exists:
        logging.info(f"  Value with string key: {ndcg[str(k)]}")