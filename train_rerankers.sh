#!/bin/bash
# train_rerankers.sh

# Default parameters
MODEL_NAME="meta-llama/Llama-2-7b-hf"
LAYER_INDICES="0 3 6 9 12 15 18 21"
WEIGHTER_TYPES="token positional surprise"
WEIGHTING_MODES="full query_only none"
WEIGHT_NORMALIZATION="linear"  # or "softmax"
DATA_PATH="./datasets/msmarco"
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_STEPS=500
SAVE_STEPS=25
EVAL_STEPS=2
MAX_LENGTH=512
MAX_LAYERS_PER_BATCH=4  # Process this many layers at once to avoid OOM
OUTPUT_DIR="./model_checkpoints/msmarco_rerankers"
TOKEN_WEIGHTS_FILE="token_frequency_data/llama2_token_freq_weights.pkl"
TOKEN_WEIGHT_TYPE="log_weights"
MARGIN=0.05
LAMBDA_FACTOR=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --layers)
            LAYER_INDICES="$2"
            shift 2
            ;;
        --weighter_types)
            WEIGHTER_TYPES="$2"
            shift 2
            ;;
        --weighting_modes)
            WEIGHTING_MODES="$2"
            shift 2
            ;;
        --weight_normalization)
            WEIGHT_NORMALIZATION="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --max_layers_per_batch)
            MAX_LAYERS_PER_BATCH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --token_weights_file)
            TOKEN_WEIGHTS_FILE="$2"
            shift 2
            ;;
        --token_weight_type)
            TOKEN_WEIGHT_TYPE="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --lambda_factor)
            LAMBDA_FACTOR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./train_rerankers.sh [options]"
            echo "Options:"
            echo "  --model_name MODEL               HuggingFace model identifier (default: meta-llama/Llama-2-7b-hf)"
            echo "  --layers \"INDICES\"               Space-separated layer indices (default: \"0 3 6 9 12 15 18 21\")"
            echo "  --weighter_types \"TYPES\"         Space-separated weighter types (default: \"token positional surprise\")"
            echo "  --weighting_modes \"MODES\"        Space-separated weighting modes (default: \"full query_only none\")"
            echo "  --weight_normalization TYPE      Weight normalization type: linear or softmax (default: linear)"
            echo "  --data_path PATH                 Path to MSMARCO dataset (default: ./datasets/msmarco)"
            echo "  --batch_size SIZE                Batch size (default: 16)"
            echo "  --gradient_accumulation_steps N  Gradient accumulation steps (default: 4)"
            echo "  --learning_rate RATE             Learning rate (default: 1e-4)"
            echo "  --weight_decay VALUE             Weight decay (default: 0.01)"
            echo "  --max_steps STEPS                Maximum training steps (default: 500)"
            echo "  --save_steps STEPS               Save checkpoint every N steps (default: 25)"
            echo "  --eval_steps STEPS               Evaluate every N steps (default: 2)"
            echo "  --max_length LENGTH              Maximum sequence length (default: 512)"
            echo "  --max_layers_per_batch N         Maximum layers to process at once (default: 4)"
            echo "  --output_dir DIR                 Output directory (default: ./model_checkpoints/msmarco_rerankers)"
            echo "  --token_weights_file FILE        Path to token weights file (default: token_frequency_data/llama2_token_freq_weights.pkl)"
            echo "  --token_weight_type TYPE         Token weight type: log_weights or reciprocal_weights (default: log_weights)"
            echo "  --margin VALUE                   Margin for ranking loss (default: 0.05)"
            echo "  --lambda_factor VALUE            Weight for MSE loss component (default: 0.1)"
            echo "  --help                           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=== Training Configuration ==="
echo "Model: $MODEL_NAME"
echo "Layer indices: $LAYER_INDICES"
echo "Weighter types: $WEIGHTER_TYPES"
echo "Weighting modes: $WEIGHTING_MODES"
echo "Weight normalization: $WEIGHT_NORMALIZATION"
echo "Data path: $DATA_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Max steps: $MAX_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Max length: $MAX_LENGTH"
echo "Max layers per batch: $MAX_LAYERS_PER_BATCH"
echo "Output directory: $OUTPUT_DIR"
echo "Token weights file: $TOKEN_WEIGHTS_FILE"
echo "Token weight type: $TOKEN_WEIGHT_TYPE"
echo "Margin: $MARGIN"
echo "Lambda factor: $LAMBDA_FACTOR"
echo "==========================="

# Convert space-separated layer indices to command argument format
LAYER_ARGS=""
for layer in $LAYER_INDICES; do
    LAYER_ARGS="$LAYER_ARGS $layer"
done

# Convert space-separated weighter types to command argument format
WEIGHTER_ARGS=""
for weighter in $WEIGHTER_TYPES; do
    WEIGHTER_ARGS="$WEIGHTER_ARGS $weighter"
done

# Convert space-separated weighting modes to command argument format
WEIGHTING_MODE_ARGS=""
for mode in $WEIGHTING_MODES; do
    WEIGHTING_MODE_ARGS="$WEIGHTING_MODE_ARGS $mode"
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Log the command
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/logs/training_$TIMESTAMP.log"
echo "Logging to: $LOG_FILE"

# Run the training command
{
    echo "Starting training at $(date)"
    echo "Command:"
    echo "python shared_model_trainer.py \
      --model_name $MODEL_NAME \
      --layer_indices $LAYER_ARGS \
      --weighter_types $WEIGHTER_ARGS \
      --weighting_modes $WEIGHTING_MODE_ARGS \
      --data_path $DATA_PATH \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
      --learning_rate $LEARNING_RATE \
      --weight_decay $WEIGHT_DECAY \
      --max_steps $MAX_STEPS \
      --save_steps $SAVE_STEPS \
      --evaluation_steps $EVAL_STEPS \
      --max_length $MAX_LENGTH \
      --max_layers_per_batch $MAX_LAYERS_PER_BATCH \
      --output_dir $OUTPUT_DIR \
      --token_weights_filepath $TOKEN_WEIGHTS_FILE \
      --token_weight_type $TOKEN_WEIGHT_TYPE \
      --margin $MARGIN \
      --lambda_factor $LAMBDA_FACTOR \
      --weight_normalization $WEIGHT_NORMALIZATION"

    # Execute the command
    python shared_model_trainer.py \
      --model_name "$MODEL_NAME" \
      --layer_indices $LAYER_ARGS \
      --weighter_types $WEIGHTER_ARGS \
      --weighting_modes $WEIGHTING_MODE_ARGS \
      --data_path "$DATA_PATH" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay "$WEIGHT_DECAY" \
      --max_steps "$MAX_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --evaluation_steps "$EVAL_STEPS" \
      --max_length "$MAX_LENGTH" \
      --max_layers_per_batch "$MAX_LAYERS_PER_BATCH" \
      --output_dir "$OUTPUT_DIR" \
      --token_weights_filepath "$TOKEN_WEIGHTS_FILE" \
      --token_weight_type "$TOKEN_WEIGHT_TYPE" \
      --margin "$MARGIN" \
      --lambda_factor "$LAMBDA_FACTOR" \
      --weight_normalization "$WEIGHT_NORMALIZATION"

    echo "Training completed at $(date)"
} 2>&1 | tee "$LOG_FILE"