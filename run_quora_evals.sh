#!/bin/bash
# run_quora_evaluation.sh - modified for new weighting methods

# Set default parameters
CHECKPOINT_DIR="/home/ubuntu/a100-storage/projects/semantic-trajectory/model_checkpoints/model_checkpoints_run_1/checkpoints"
OUTPUT_DIR="./results/quora_new_weighting_comparison"
MAX_QUERIES=100
BATCH_SIZE=128
WORKERS=24
WEIGHTING_MODES="query_log_only,positional,surprise"
CALC_SURPRISE=true
POS_B=3.0
POS_C=3.0
SURPRISE_SCALING=1.0

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
	--weighting_modes)
	  WEIGHTING_MODES="$2"
	  shift 2
	  ;;
	--pos_b)
	  POS_B="$2"
	  shift 2
	  ;;
	--pos_c)
	  POS_C="$2"
	  shift 2
	  ;;
	--surprise_scaling)
	  SURPRISE_SCALING="$2"
	  shift 2
	  ;;
	--no_surprise)
	  CALC_SURPRISE=false
	  shift
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
	  echo "  --checkpoint_dir DIR     Directory with model checkpoints"
	  echo "  --output_dir DIR         Output directory for results"
	  echo "  --max_queries N          Maximum queries to process"
	  echo "  --batch_size N           Batch size for document encoding"
	  echo "  --workers N              Number of worker processes for BM25"
	  echo "  --weighting_modes LIST   Comma-separated list of weighting modes to evaluate"
	  echo "  --pos_b FLOAT            Base parameter for positional weighting (default: 0.9)"
	  echo "  --pos_c FLOAT            Coefficient for positional weighting (default: 1.0)"
	  echo "  --surprise_scaling FLOAT Scaling factor for surprise values (default: 1.0)"
	  echo "  --no_surprise            Disable surprise calculation (not recommended)"
	  echo "  --force_bm25             Force recomputation of BM25 results"
	  echo "  --full                   Process all available queries"
	  echo "  --help                   Show this help message"
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

echo "Starting Quora evaluation with new weighting methods..."
echo "Using checkpoint directory: $CHECKPOINT_DIR"
echo "Maximum queries: $MAX_QUERIES"
echo "BM25 worker processes: $WORKERS"
echo "Weighting modes: $WEIGHTING_MODES"
echo "Positional weighting parameters: b=$POS_B, c=$POS_C"
echo "Surprise calculation: $CALC_SURPRISE"

# Build the surprise calculation flag
if [ "$CALC_SURPRISE" = true ]; then
  SURPRISE_FLAG="--calc_surprise"
  echo "Surprise calculation enabled with scaling factor: $SURPRISE_SCALING"
else
  SURPRISE_FLAG=""
  echo "Surprise calculation disabled"
fi

# Run the evaluation
python quora_evaluation.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --weighting_modes "$WEIGHTING_MODES" \
  --output_dir "$OUTPUT_DIR" \
  --max_queries "$MAX_QUERIES" \
  --batch_size "$BATCH_SIZE" \
  --bm25_workers "$WORKERS" \
  --bm25_results "$OUTPUT_DIR/quora_bm25_results.tsv" \
  --bm25_index "$OUTPUT_DIR/quora_bm25_index.pkl" \
  --pos_b "$POS_B" \
  --pos_c "$POS_C" \
  --surprise_scaling "$SURPRISE_SCALING" \
  $SURPRISE_FLAG \
  $FORCE_BM25 \
  --save_runs \
  --save_metrics

echo "Evaluation complete!"
echo "Results saved to $OUTPUT_DIR"