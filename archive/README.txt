## Usage Instructions For Taining:

1. Save this script as `setup.sh` in your project root directory
2. Make sure you have a `requirements.txt` file in the same directory
3. Make the script executable:
   ```bash
   chmod +x setup.sh
   ```
4. Run the script:
   ```bash
   ./setup.sh
   ```
5. Run the training script, configure as you like, pay attention to layer data:
   ```
   python run_training.py   --model_name meta-llama/Llama-2-7b-hf   --layers 0,3,6,9,12,15,18,21   --dtw_layers 6,9,12,15   --batch_size 128   --gradient_accumulation_steps 4   --learning_rate 1e-2   --weight_decay 1e-6   --steps 1000   --eval_steps 50   --save_steps 50   --output_dir /home/ubuntu/a100-storage/projects/semantic-trajectory/model_checkpoints
   ```

The script includes error checking to exit if any command fails, creates necessary directories, and provides status messages throughout the process.

## Usage Instructions For Evaluation:

1. Download evaluation data via
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```
2. 