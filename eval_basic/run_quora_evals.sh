#!/bin/bash
# run_quora_evaluation.sh

# Set default parameters
CHECKPOINT_DIR="/home/ubuntu/a100-storage/projects/semantic-trajectory/model_checkpoints/model_checkpoints_run_1/checkpoints"
OUTPUT_DIR="./results/quora_weighting_comparison"
MAX_QUERIES=100
BATCH_SIZE=128
WORKERS=24

# Process arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		--checkpoint_dir)
			CHECKPOINT_DIR="$2"
			shift 2
			;;
		--output_dir)
			OUTPUT_DIR="$2"
			shift 2
			;;
		--max_queries)
			MAX_QUERIES="$2"
			shift 2
			;;
		--batch_size)
			BATCH_SIZE="$2"
			shift 2
			;;
		--workers)
			WORKERS="$2"
			shift 2
			;;
		--force_bm25)
			FORCE_BM25="--force_bm25"
			shift
			;;
		--full)
			MAX_QUERIES=5000  # Use a large number to process all queries
			shift
			;;
		--help)
			echo "Usage: $0 [options]"
			echo "Options:"
			echo "  --checkpoint_dir DIR  Directory with model checkpoints (default: /home/ubuntu/a100-storage/projects/semantic-trajectory/model_checkpoints/model_checkpoints_run_1/checkpoints)"
			echo "  --output_dir DIR      Output directory for results (default: ./results/quora_weighting_comparison)"
			echo "  --max_queries N       Maximum queries to process (default: 100)"
			echo "  --batch_size N        Batch size for document encoding (default: 32)"
			echo "  --workers N           Number of worker processes for BM25 (default: 24)"
			echo "  --force_bm25          Force recomputation of BM25 results"
			echo "  --full                Process all available queries"
			echo "  --help                Show this help message"
			exit 0
			;;
		*)
			echo "Unknown parameter: $1"
			echo "Use --help for usage information."
			exit 1
			;;
	esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Quora evaluation with parallel BM25 retrieval..."
echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo "Maximum queries: $MAX_QUERIES"
echo "BM25 worker processes: $WORKERS"

# Run the evaluation
python quora_evaluation.py \
	--checkpoint_dir "$CHECKPOINT_DIR" \
	--weighting_modes "uniform,learned,combined" \
	--output_dir "$OUTPUT_DIR" \
	--max_queries "$MAX_QUERIES" \
	--batch_size "$BATCH_SIZE" \
	--bm25_workers "$WORKERS" \
	--bm25_results "$OUTPUT_DIR/quora_bm25_results.tsv" \
	--bm25_index "$OUTPUT_DIR/quora_bm25_index.pkl" \
	$FORCE_BM25 \
	--save_runs \
	--save_metrics

echo "Evaluation complete!"
echo "Results saved to $OUTPUT_DIR"