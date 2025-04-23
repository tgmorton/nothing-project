#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=gpt2_p6000_test1_venv  # Job name for identification
#SBATCH --partition=general_gpu_p6000    # Target the P6000 queue
#SBATCH --nodes=1                        # Request one node
#SBATCH --ntasks-per-node=1              # Run one task (the python script)
#SBATCH --cpus-per-task=8                # Request CPUs for the task (dataloader workers, etc.) - P6000 node has 24
#SBATCH --mem=48G                        # Request RAM for the task (Node has 64GB total) - Excludes GPU VRAM
#SBATCH --gres=gpu:1                     # Request 1 GPU (Can potentially be more specific like gpu:P6000:1 if needed by your cluster)
#SBATCH --time=24:00:00                  # Time limit (HH:MM:SS) - Max is 7 days, 24h is a good test duration
#SBATCH --output=logs/%x_%j.out          # Standard output log file (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err           # Standard error log file

# Exit on error
set -e

# === Environment Setup ===
echo "=== Job Started: $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE" # Might be total node memory, check --mem specification
echo "GPUs: $CUDA_VISIBLE_DEVICES" # Slurm usually sets this

# --- Load necessary modules ---
# >> Replace with your specific module load commands <<
echo "Loading modules..."
# Example: module load cuda/11.8 python/3.10 # Adjust versions/names! Must match python used for venv
module load cuda/11.8 # <<< MODIFY THESE LINES FOR YOUR CLUSTER

# --- Securely Load Neptune Credentials ---
# IMPORTANT: Assumes you created ~/.neptune_creds with 'export NEPTUNE_API_TOKEN=...'
#            and set permissions with 'chmod 600 ~/.neptune_creds'
NEPTUNE_CRED_FILE="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE"
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE"
    # Decide if you want the job to fail or continue without Neptune
    # exit 1 # Uncomment this line to make the job fail if creds are missing
fi

# --- Activate your Python virtual environment ---
# >> Replace with the absolute path to YOUR venv activation script <<
VENV_PATH="/path/to/your/project/root/venv/bin/activate" # <<< MODIFY THIS LINE
if [ -f "$VENV_PATH" ]; then
    echo "Activating venv: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment activation script not found at $VENV_PATH"
    exit 1
fi

# --- Optional: Ensure requirements are installed (good for reproducibility) ---
# echo "Checking/Installing requirements..."
# pip install -r /path/to/your/project/root/requirements.txt # <<< MODIFY PATH & Uncomment if needed

# --- Set project directory ---
# >> Optional: cd to your project root if your script relies on relative paths <<
# PROJECT_DIR="/path/to/your/project/root" # <<< MODIFY PATH & Uncomment if needed
# cd "$PROJECT_DIR"

# === Training Script Execution ===
echo "Starting Python training script..."

# Ensure the logs directory exists
mkdir -p logs

# Define paths (MODIFY THESE!)
TRAIN_DATA_PATH="/path/to/your/train_arrow_dataset"
VALID_DATA_PATH="/path/to/your/validation_arrow_dataset"
OUTPUT_DIR="/path/to/your/output_directory/gpt2_p6000_test1_venv"
PRIMING_DIR="/path/to/your/priming_csv_directory" # Needed if using --run_priming_eval

# Define Neptune args (Project might be set via cred file now, Run Name/Tags are still useful)
#NEPTUNE_PROJECT_ARG=""
# If NEPTUNE_PROJECT is NOT exported in your cred file, uncomment and set it here:
NEPTUNE_PROJECT_ARG="--neptune_project thmorton/NothingProject"
NEPTUNE_RUN_NAME="gpt2_p6000_test1_venv_$(date +%Y%m%d_%H%M)" # Example name
NEPTUNE_TAGS="p6000" "test1" "baseline" "venv"

# First iteration parameters
# Note: No explicit Neptune project/token args needed if env vars are set
python src/train_with_priming.py \
    --model "gpt2" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --validation_dataset_path "${VALID_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 32 \
    \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --num_warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --num_workers ${SLURM_CPUS_PER_TASK:-4} \
    --seed 42 \
    \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 500 \
    \
    ${NEPTUNE_PROJECT_ARG} \
    --neptune_run_name "${NEPTUNE_RUN_NAME}" \
    --neptune_tags ${NEPTUNE_TAGS} \
    # --run_priming_eval \ # Uncomment if you want to run priming eval
    # --priming_eval_dir_path "${PRIMING_DIR}" # Uncomment if using priming eval
    # --priming_eval_steps 500 # Optional: default is eval_steps

# === Job Completion ===
echo "=== Job Finished: $(date) ==="