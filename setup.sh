#!/bin/bash
set -e  # Exit on any error

echo "Starting setup process..."

# Step 1: Download the dataset
echo "Downloading dataset..."
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz
echo "Download completed."

# Step 2: Decompress files (tar handles gunzipping automatically with -z flag)
echo "Decompressing files..."
tar -xzf triples.train.small.tar.gz
rm triples.train.small.tar.gz
echo "Decompression completed."

# Step 3: Create and setup Python environment
echo "Setting up Python environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Exiting."
    exit 1
fi

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created."

# Activate virtual environment
source venv/bin/activate

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "Warning: requirements.txt not found. No packages installed."
fi

echo "Setup complete! Virtual environment is now active."
echo "To deactivate the virtual environment, run: deactivate"