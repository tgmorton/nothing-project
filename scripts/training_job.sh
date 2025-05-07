#!/bin/bash
# === SBATCH Directives for A5000 Training ===
#SBATCH --job-name=a5k_training_job      # Will be overridden by orchestrator
#SBATCH --partition=general_gpu_a5000    # <<< Target A5000 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                # <<< Increased CPUs
#SBATCH --mem=64G                        # <<< Increased RAM
#SBATCH --time=24:00:00
#SBATCH --output=logs/default_train_%j.out # Will be overridden by orchestrator
#SBATCH --error=logs/default_train_%j.err  # Will be overridden by orchestrator
#SBATCH --mail-type=END # Orchestrator handles BEGIN/ALL
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error
set -e

# === Environment Setup from Orchestrator ===
echo "=== Training Job Started: $(date) ==="
echo "Training Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Host Project Dir (from Orchestrator): ${HOST_PROJECT_DIR}"
echo "Shared Output Directory (from Orchestrator): ${SHARED_OUTPUT_DIR}"
echo "Shared Run ID (from Orchestrator): ${SHARED_RUN_ID}"

# Validate variables from orchestrator
if [ -z "$HOST_PROJECT_DIR" ] || [ -z "$SHARED_OUTPUT_DIR" ] || [ -z "$SHARED_RUN_ID" ]; then
    echo "ERROR: Critical environment variables (HOST_PROJECT_DIR, SHARED_OUTPUT_DIR, SHARED_RUN_ID) not set by orchestrator!"
    exit 1
fi

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Securely Load Neptune Credentials ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds" # Assumes $HOME is correctly set for the Slurm user
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    if [ -n "${NEPTUNE_PROJECT:-}" ]; then
      export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"
    fi
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE. Neptune logging might fail."
fi

# --- Define Paths on Host (using orchestrator-provided HOST_PROJECT_DIR) ---
HOST_SIF_PATH="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE if SIF name/location differs
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval" # If train.py needs access for non-eval purposes

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/output_train" # Mount point for SHARED_OUTPUT_DIR inside container
CONTAINER_PRIMING_DIR="/eval_data_train"

# --- Preparations ---
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
# SHARED_OUTPUT_DIR is created by the orchestrator.

# === Training Script Execution ===
echo "Starting Python training script (Training Only) inside Singularity container..."

# --- Define paths relative to container mount points ---
CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_100m"
CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m"
# CONTAINER_PRIMING_PATH_FOR_TRAIN="${CONTAINER_PRIMING_DIR}/priming-corpuses" # If needed by train.py

# --- Define Neptune args ---
NEPTUNE_PROJECT_ARG_VAL="" # Default to empty
# Use project from env var set by cred file or SINGULARITYENV_ if already set
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG_VAL="--neptune_project ${SINGULARITYENV_NEPTUNE_PROJECT}"
elif [ -n "${NEPTUNE_PROJECT:-}" ]; then # Fallback to shell variable if not passed via SINGULARITYENV_
    NEPTUNE_PROJECT_ARG_VAL="--neptune_project ${NEPTUNE_PROJECT}"
else
    NEPTUNE_PROJECT_ARG_VAL="--neptune_project thmorton/NothingProject" # Hardcoded fallback <<< UPDATE
fi

# Use the SHARED_RUN_ID for a unique Neptune run name
NEPTUNE_RUN_NAME_VAL="train_${SHARED_RUN_ID}"
NEPTUNE_TAGS_VAL="a5000 ${SHARED_RUN_ID} training_phase gpt2_100m" # Customize tags

# --- Set PyTorch CUDA Allocator Config ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Execute Singularity Command ---
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${SHARED_OUTPUT_DIR}":"${CONTAINER_OUTPUT_DIR}" \
    -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/src/train.py" \
        --model "gpt2" \
        --train_dataset_path "${CONTAINER_TRAIN_DATA_PATH}" \
        --validation_dataset_path "${CONTAINER_VALID_DATA_PATH}" \
        # --priming_eval_dir_path "${CONTAINER_PRIMING_PATH_FOR_TRAIN}" # Only if train.py needs it for non-eval
        --output_dir "${CONTAINER_OUTPUT_DIR}" \
        \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        \
        --learning_rate 3e-4 \
        --lr_scheduler_type "cosine" \
        --num_warmup_steps 100 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --model_size "100m" \
        \
        --use_amp \
        --num_workers ${SLURM_CPUS_PER_TASK:-4} \
        --seed 42 \
        \
        --logging_steps 200 \
        --save_steps 400 \
        # --eval_steps 400                     # REMOVE: Evaluation handled by monitor script
        # --priming_eval_steps 400             # REMOVE: Evaluation handled by monitor script
        \
        ${NEPTUNE_PROJECT_ARG_VAL} \
        --neptune_run_name "${NEPTUNE_RUN_NAME_VAL}" \
        --neptune_tags ${NEPTUNE_TAGS_VAL} \
        # --local_eval                        # REMOVE
        # --evaluate_script_path ...          # REMOVE
        # --trigger_standard_eval             # REMOVE
        # --trigger_priming_eval              # REMOVE

# Create a sentinel file in the SHARED_OUTPUT_DIR to signal completion
touch "${SHARED_OUTPUT_DIR}/TRAINING_COMPLETED.txt"
echo "Training script finished. Sentinel file TRAINING_COMPLETED.txt created in ${SHARED_OUTPUT_DIR}."

# === Job Completion ===
echo "=== Training Job Finished: $(date) ==="