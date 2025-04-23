#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=gpt2_p6000_sif_test1    # Job name for identification
#SBATCH --partition=general_gpu_p6000    # Target the P6000 queue
#SBATCH --nodes=1                        # Request one node
#SBATCH --ntasks-per-node=1              # Run one task (the python script)
#SBATCH --cpus-per-task=8                # Request CPUs for the task (dataloader workers, etc.)
#SBATCH --mem=48G                        # Request RAM for the task (Node has 64GB total)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --time=24:00:00                  # Time limit (HH:MM:SS) - e.g., 24 hours for first test
#SBATCH --output=logs/%x_%j.out          # Standard output log file (%x=job name, %j=job ID) - Relative to submission dir
#SBATCH --error=logs/%x_%j.err           # Standard error log file - Relative to submission dir

# Exit on error
set -e

# === Environment Setup ===
echo "=== Job Started: $(date) ==="
echo "Current Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR" # Directory where sbatch was run
echo "Node: $SLURMD_NODENAME"
echo "Username: $USER"
echo "Home Directory: $HOME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules ---
# IMPORTANT: Verify these module names and versions using 'module avail' on SSRDE
echo "Loading system modules..."
module load singularity/4.1.1  cuda/11.8  # <<< MODIFY versions if needed based on 'module avail'

# --- Securely Load Neptune Credentials ---
# Assumes file created at $HOME/.neptune_creds with 'export NEPTUNE_API_TOKEN=...'
# and permissions set with 'chmod 600 $HOME/.neptune_creds'
NEPTUNE_CRED_FILE="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE"
    # Make Neptune token available inside Singularity container
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}" # Use :- to avoid error if unset
    # Export project if it's set in the cred file
    if [ -n "${NEPTUNE_PROJECT:-}" ]; then
      export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"
    fi
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE. Continuing without Neptune export."
fi

# --- Define Paths on Host (SSRDE Cluster) ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif"  # Location of your transferred .sif file

# !! MODIFY THESE BASE PATHS !!
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"           # <<< EXAMPLE/MODIFY: Assumes data is in project/data
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"      # <<< EXAMPLE/MODIFY: Assumes output goes to project/results
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval" # <<< EXAMPLE/MODIFY: Assumes priming is in project/priming_data

# Define corresponding paths *inside* the container
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/.output"
CONTAINER_PRIMING_DIR="/eval"

# Define specific run output directory name
RUN_OUTPUT_NAME="gpt2_p6000_sif_test1"
HOST_RUN_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${RUN_OUTPUT_NAME}"
CONTAINER_RUN_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}/${RUN_OUTPUT_NAME}"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Data Base Directory (Host): ${HOST_DATA_BASE_DIR}"
echo "Output Base Directory (Host): ${HOST_OUTPUT_BASE_DIR}"

# Verify Singularity image exists
if [ ! -f "$HOST_SIF_PATH" ]; then
    echo "ERROR: Singularity image not found at $HOST_SIF_PATH"
    exit 1
fi

# Ensure the base output directory exists on the host system
# This directory will be created relative to where sbatch is run if paths are relative,
# or at the absolute path if specified. Using absolute is safer.
mkdir -p "${HOST_RUN_OUTPUT_DIR}"
echo "Ensured host output directory exists: ${HOST_RUN_OUTPUT_DIR}"

# Ensure the logs directory exists (for Slurm output) relative to submission dir
# Assumes script is run from HOST_PROJECT_DIR
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Training Script Execution (Inside Container) ===
echo "Starting Python training script inside Singularity container..."

# Define paths relative to container mount points for the python script args
# These paths MUST match the targets in the -B arguments below AND your data structure
CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m"         # Example: Assumes HOST_DATA_BASE_DIR/processed/train_arrow exists
CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_10m"    # Example: Assumes HOST_DATA_BASE_DIR/processed/validation_arrow exists
CONTAINER_PRIMING_PATH="${CONTAINER_PRIMING_DIR}/priming_corpuses" # Example: Assumes HOST_PRIMING_BASE_DIR/priming_eval_files exists

# Define Neptune args (Project might be set via cred file/env var)
NEPTUNE_PROJECT_ARG=""
# If NEPTUNE_PROJECT is NOT set via env var SINGULARITYENV_NEPTUNE_PROJECT, uncomment and set it here:
# NEPTUNE_PROJECT_ARG="--neptune_project YOUR_ORG/YOUR_PROJECT"
NEPTUNE_RUN_NAME="${RUN_OUTPUT_NAME}_$(date +%Y%m%d_%H%M)" # Link run name to output dir name
NEPTUNE_TAGS="p6000 test1 baseline singularity py39"       # Added space separation for tags

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_OUTPUT_BASE_DIR}":"${CONTAINER_OUTPUT_DIR}" \
    -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/src/train.py" \
        --model "gpt2" \
        --train_dataset_path "${CONTAINER_TRAIN_DATA_PATH}" \
        --validation_dataset_path "${CONTAINER_VALID_DATA_PATH}" \
        --output_dir "${CONTAINER_RUN_OUTPUT_DIR}" \
        \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 32 \
        --per_device_eval_batch_size 32 \
        \
        --learning_rate 5e-5 \
        --lr_scheduler_type "cosine" \
        --num_warmup_steps 1000 \
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
        --neptune_tags ${NEPTUNE_TAGS}
        --run_priming_eval \ # Uncomment if you want to run priming eval
        --priming_eval_dir_path "${CONTAINER_PRIMING_PATH}" # Uncomment if using priming eval
        --priming_eval_steps 100 # Optional: default is eval_steps

# === Job Completion ===
echo "=== Job Finished: $(date) ==="