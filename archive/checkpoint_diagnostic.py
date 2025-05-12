#!/usr/bin/env python
# checkpoint_diagnostic.py
"""
Diagnostic script to check checkpoint structure and help debug loading issues
"""

import os
import sys
import argparse
import torch
import glob
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
	format="%(asctime)s - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

logger = logging.getLogger(__name__)

def explore_checkpoint_directory(checkpoint_dir):
	"""Explore a checkpoint directory structure and provide detailed information"""
	if not os.path.exists(checkpoint_dir):
		logger.error(f"Directory does not exist: {checkpoint_dir}")
		return False

	logger.info(f"Exploring checkpoint directory: {checkpoint_dir}")

	# Check for expected subdirectories
	expected_dirs = ["dtw_model"]
	for i in range(0, 24, 3):  # Assuming layers 0,3,6,9,12,15,18,21
		expected_dirs.append(f"single_layer_{i}")

	found_dirs = []
	for dir_name in os.listdir(checkpoint_dir):
		dir_path = os.path.join(checkpoint_dir, dir_name)
		if os.path.isdir(dir_path):
			found_dirs.append(dir_name)

	logger.info(f"Found subdirectories: {found_dirs}")

	# Check which expected directories exist
	missing_dirs = [d for d in expected_dirs if d not in found_dirs]
	if missing_dirs:
		logger.warning(f"Missing expected directories: {missing_dirs}")

	# Explore each existing directory
	model_checkpoints = defaultdict(list)
	for dir_name in found_dirs:
		dir_path = os.path.join(checkpoint_dir, dir_name)

		# Check naming pattern of checkpoint files
		pt_files = glob.glob(os.path.join(dir_path, "*.pt"))
		weighter_step_files = [f for f in pt_files if "weighter_step_" in os.path.basename(f)]
		other_pt_files = [f for f in pt_files if f not in weighter_step_files]

		# Show file info
		logger.info(f"Directory: {dir_name}")
		if weighter_step_files:
			logger.info(f"  Found {len(weighter_step_files)} weighter_step files:")
			for f in sorted(weighter_step_files):
				file_size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
				logger.info(f"    {os.path.basename(f)} ({file_size:.2f} MB)")

				# Try to extract step number
				try:
					step = int(os.path.basename(f).split("_")[2].split(".")[0])
					model_checkpoints[dir_name].append(step)
				except (IndexError, ValueError):
					logger.warning(f"    Couldn't parse step number from {os.path.basename(f)}")

				# Try to load the checkpoint
				try:
					state_dict = torch.load(f, map_location="cpu")
					if isinstance(state_dict, dict):
						keys = list(state_dict.keys())
						logger.info(f"    Contains {len(keys)} keys, first few: {keys[:3]}...")
					else:
						logger.warning(f"    Not a state_dict, type: {type(state_dict)}")
				except Exception as e:
					logger.error(f"    Error loading file: {e}")

		if other_pt_files:
			logger.info(f"  Found {len(other_pt_files)} other .pt files:")
			for f in sorted(other_pt_files):
				logger.info(f"    {os.path.basename(f)}")

		if not pt_files:
			# Check for other files
			other_files = os.listdir(dir_path)
			if other_files:
				logger.info(f"  No .pt files, but found other files: {other_files[:5]}...")
			else:
				logger.info(f"  Directory is empty")

	# Summary of available models
	logger.info("\nSummary of available checkpoints:")
	available_models = []

	if model_checkpoints["dtw_model"]:
		available_models.append("dtw")
		logger.info(f"DTW model: steps {sorted(model_checkpoints['dtw_model'])}")

	for dir_name in found_dirs:
		if dir_name.startswith("single_layer_"):
			layer_idx = dir_name.split("_")[-1]
			if model_checkpoints[dir_name]:
				available_models.append(f"layer_{layer_idx}")
				logger.info(f"Layer {layer_idx} model: steps {sorted(model_checkpoints[dir_name])}")

	if available_models:
		logger.info(f"\nAvailable models for loading: {available_models}")
		return True
	else:
		logger.error("No loadable model checkpoints found!")
		return False

def main():
	parser = argparse.ArgumentParser(description="Checkpoint directory diagnostic tool")
	parser.add_argument("--checkpoint_dir", type=str, required=True,
						help="Path to the checkpoint directory to analyze")
	args = parser.parse_args()

	explore_checkpoint_directory(args.checkpoint_dir)

if __name__ == "__main__":
	main()