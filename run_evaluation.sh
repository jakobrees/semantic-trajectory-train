#!/bin/bash
# run_evaluation.sh

# Check for required parameters
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [model_type] [batch_size] [max_queries]"
    echo "Example: $0 /home/ubuntu/a100-storage/llama_reranker_project/model_checkpoints/checkpoints dtw 64"
    exit 1
fi

# Set parameters
CHECKPOINT_PATH=$1
MODEL_TYPE=${2:-"dtw"}  # Default to DTW model
BATCH_SIZE=${3:-64}     # Default batch size
MAX_QUERIES=${4:-6800}  # Default size

# Data paths - adjust these to your data locations
QUERIES_PATH="./queries.dev.tsv"
COLLECTION_PATH="./collection.tsv"
QRELS_PATH="./qrels.dev.tsv"
RUN_PATH="./top1000.dev"
OUTPUT_RUN="./reranked_${MODEL_TYPE}.run"

# Run evaluation
python evaluate_model.py \
  --queries ${QUERIES_PATH} \
  --collection ${COLLECTION_PATH} \
  --qrels ${QRELS_PATH} \
  --run ${RUN_PATH} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --model_name meta-llama/Llama-2-7b-hf \
  --layers 0,3,6,9,12,15,18,21 \
  --dtw_layers 6,9,12,15 \
  --model_type ${MODEL_TYPE} \
  --batch_size ${BATCH_SIZE} \
  --output_run ${OUTPUT_RUN} \
  ${MAX_QUERIES:+--max_queries ${MAX_QUERIES}}

# Verify the output exists
if [ -f ${OUTPUT_RUN} ]; then
    echo "Reranked run file saved to ${OUTPUT_RUN}"
else
    echo "Error: Run file not created"
    exit 1
fi

# Additional evaluation with ms_marco_eval.py if available
if [ -f "ms_marco_eval.py" ]; then
    echo "Running official MSMARCO evaluation script..."
    python ms_marco_eval.py ${QRELS_PATH} ${OUTPUT_RUN}
fi

echo "Evaluation complete!"