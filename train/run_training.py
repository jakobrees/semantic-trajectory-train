import argparse
import logging
import torch
import os
import sys
from multi_model_trainer import train_multi_layer_reranker

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler("training.log")
	]
)
logger = logging.getLogger('run_training')

def parse_args():
	parser = argparse.ArgumentParser(description="Train multi-layer reranker models")

	# Model configuration
	parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
						help="HuggingFace model identifier")
	parser.add_argument("--layers", type=str, default="0,3,6,9,12,15,18,21",
						help="Comma-separated list of layer indices to use")
	parser.add_argument("--dtw_layers", type=str, default="6,9,12,15",
						help="Comma-separated list of layer indices to use for DTW")

	# Training configuration
	parser.add_argument("--triples_path", type=str, default="./triples.train.small.tsv",
						help="Path to MS MARCO triples file")
	parser.add_argument("--batch_size", type=int, default=16,
						help="Batch size for training")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
						help="Number of batches to accumulate before optimizer step")
	parser.add_argument("--learning_rate", type=float, default=1e-4,
						help="Learning rate for optimizers")
	parser.add_argument("--weight_decay", type=float, default=1e-5,
                      help="L2 regularization coefficient")
	parser.add_argument("--steps", type=int, default=1000,
						help="Number of training steps")
	parser.add_argument("--eval_steps", type=int, default=25,
						help="Evaluate and log metrics every N steps")
	parser.add_argument("--save_steps", type=int, default=100,
						help="Save checkpoints every N steps")

	# Output configuration
	parser.add_argument("--output_dir", type=str, default="./model_checkpoints",
						help="Directory to save outputs")

	# Resume training
	parser.add_argument("--resume_from", type=str, default=None,
						help="Directory to resume checkpoints from")
	parser.add_argument("--resume_step", type=int, default=None,
						help="Step to resume from")

	# Memory management
	parser.add_argument("--memory_cleanup_steps", type=int, default=10,
						help="Run explicit memory cleanup every N steps")

	return parser.parse_args()

def main():
	args = parse_args()

	# Convert string lists to integer lists
	layer_indices = [int(x) for x in args.layers.split(',')]
	dtw_layer_indices = [int(x) for x in args.dtw_layers.split(',')]

	# Print configuration
	logger.info("=== Training Configuration ===")
	logger.info(f"Model: {args.model_name}")
	logger.info(f"Layer indices: {layer_indices}")
	logger.info(f"DTW layer indices: {dtw_layer_indices}")
	logger.info(f"Batch size: {args.batch_size}")
	logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
	logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
	logger.info(f"Learning rate: {args.learning_rate}")
	logger.info(f"Weight decay: {args.weight_decay}")
	logger.info(f"Steps: {args.steps}")
	logger.info(f"Output directory: {args.output_dir}")

	if torch.cuda.is_available():
		logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
		logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

	# Run training
	train_multi_layer_reranker(
		model_name=args.model_name,
		layer_indices=layer_indices,
		dtw_layer_indices=dtw_layer_indices,
		triples_path=args.triples_path,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		batch_size=args.batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		steps=args.steps,
		eval_steps=args.eval_steps,
		save_steps=args.save_steps,
		output_dir=args.output_dir,
		resume_from=args.resume_from,
		resume_step=args.resume_step,
		memory_cleanup_steps=args.memory_cleanup_steps
	)

if __name__ == "__main__":
	main()