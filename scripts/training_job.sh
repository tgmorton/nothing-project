#!/bin/bash
# training_job.sbatch
# Job name, output/error paths will be overridden by main_orchestrator.
#SBATCH --job-name=a5k_training_job
#SBATCH --partition=general_gpu_a5000   # Target A5000 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1                    # Request 1 A5000 GPU for training
#SBATCH --time=7-00:00:00                  # Max time for a single training run
#SBATCH --mail-type=END,FAIL            # Notify on end/fail of this training job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error for this training script
set -e
# set -o pipefail

echo "=== Training Job Script Started: $(date) ==="
echo "Training Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Host Project Dir (received): ${HOST_PROJECT_DIR}"
# SHARED_OUTPUT_DIR_HOST is the actual host path passed by main_orchestrator
echo "Shared Output Directory on Host (received): ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID (received): ${SHARED_RUN_ID}"
echo "Seed for this Training Run (received): ${SEED_FOR_TRAINING}"

# --- Validate variables received ---
if [ -z "$HOST_PROJECT_DIR" ] || [ -z "$SHARED_OUTPUT_DIR_HOST" ] || [ -z "$SHARED_RUN_ID" ] || [ -z "$SEED_FOR_TRAINING" ]; then
    echo "CRITICAL ERROR (Training Job): Essential environment variables not set by main_orchestrator!"
    exit 1
fi

# --- Load necessary system modules ---
echo "Loading system modules: singularity, cuda..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Securely Load Neptune Credentials ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds" # Assumes $HOME is correctly set for the Slurm user
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE" # Exports NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    if [ -n "${NEPTUNE_PROJECT:-}" ]; then
      export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"
    fi
else
    echo "WARNING (Training Job): Neptune credentials file not found at $NEPTUNE_CRED_FILE. Neptune logging might fail."
fi

# --- Define Paths on Host (using orchestrator-provided HOST_PROJECT_DIR) ---
HOST_SIF_PATH="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE if SIF name/location differs
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
# HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval" # Only if train.py needs priming files for *non-evaluation* purposes during training

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace"  # Standard mount point for project code
CONTAINER_DATA_DIR="/data"        # Mount point for HOST_DATA_BASE_DIR
# This is where SHARED_OUTPUT_DIR_HOST will be mounted inside the container for train.py
CONTAINER_OUTPUT_TARGET_DIR="/output_train"
# CONTAINER_PRIMING_TARGET_DIR="/eval_data_train" # Only if train.py uses priming files for non-eval

# --- Preparations ---
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "CRITICAL ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
# SHARED_OUTPUT_DIR_HOST is already created by main_orchestrator.

# === Training Script Execution ===
echo "Starting Python training script (train.py - Training Only Mode) for Seed ${SEED_FOR_TRAINING}..."

# --- Define paths relative to container mount points for train.py ---
# Example paths, adjust to your actual dataset structure
CONTAINER_TRAIN_SET_PATH="${CONTAINER_DATA_DIR}/processed/training_set_10m" # For 10M model
CONTAINER_VALID_SET_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m"     # For 10M model

# --- Define Neptune args for train.py ---
NEPTUNE_PROJECT_ARG_FOR_TRAINPY=""
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then # Prioritize env var set by cred file
    NEPTUNE_PROJECT_ARG_FOR_TRAINPY="--neptune_project ${SINGULARITYENV_NEPTUNE_PROJECT}"
elif [ -n "${NEPTUNE_PROJECT:-}" ]; then # Fallback to shell variable if not passed via SINGULARITYENV_
    NEPTUNE_PROJECT_ARG_FOR_TRAINPY="--neptune_project ${NEPTUNE_PROJECT}"
else # Hardcoded fallback if no project info found
    NEPTUNE_PROJECT_ARG_FOR_TRAINPY="--neptune_project thmorton/NothingProject" # <<< UPDATE Hardcoded fallback
fi

# SHARED_RUN_ID is already seed-specific (e.g., "s42_j12345_ts...")
NEPTUNE_RUN_NAME_FOR_TRAINPY="train_${SHARED_RUN_ID}"
# Construct tags, including the seed and SHARED_RUN_ID for traceability
NEPTUNE_TAGS_ARRAY=("training_phase" "gpt2_10m" "seed_${SEED_FOR_TRAINING}" "${SHARED_RUN_ID}")
# Add more tags as needed: "a5000" "baseline" etc.
# NEPTUNE_TAGS_ARRAY+=("a5000_gpu")
NEPTUNE_TAGS_STRING=$(IFS=" "; echo "${NEPTUNE_TAGS_ARRAY[*]}") # Convert to space-separated string for CLI

# --- Set PyTorch CUDA Allocator Config (if needed) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Execute Singularity Command ---
# Note: --local_eval and related args are *omitted* from the python3 train.py call.
# train.py will default local_eval to False, and evaluation will be handled by evaluation_monitor.sbatch.
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_OUTPUT_TARGET_DIR}" \
    # -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_TARGET_DIR}" # Only if train.py uses priming files non-eval
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/src/train.py" \
        --model "gpt2" \
        --model_size "10m" \
        --train_dataset_path "${CONTAINER_TRAIN_SET_PATH}" \
        --validation_dataset_path "${CONTAINER_VALID_SET_PATH}" \
        --output_dir "${CONTAINER_OUTPUT_TARGET_DIR}" \
        --seed "${SEED_FOR_TRAINING}" \
        \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        \
        --learning_rate 3e-4 \
        --lr_scheduler_type "cosine" \
        --num_warmup_steps 1000 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        \
        --use_amp \
        --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
        \
        --logging_steps 50 \
        --save_steps 100 \
        # --eval_steps 100 # train.py will ignore this if local_eval=False (default)
        # --priming_eval_steps 100 # train.py will ignore this if local_eval=False
        \
        ${NEPTUNE_PROJECT_ARG_FOR_TRAINPY} \
        --neptune_run_name "${NEPTUNE_RUN_NAME_FOR_TRAINPY}" \
        --neptune_tags ${NEPTUNE_TAGS_STRING}
        # DO NOT ADD --local_eval or related flags here for the orchestrated sweep.

# Create the sentinel file in the SHARED_OUTPUT_DIR_HOST to signal completion of train.py
touch "${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"
echo "Training script (train.py) finished for Seed ${SEED_FOR_TRAINING}. Sentinel file TRAINING_COMPLETED.txt created in ${SHARED_OUTPUT_DIR_HOST}."

# === Job Completion ===
echo "=== Training Job Script Finished for Seed ${SEED_FOR_TRAINING}: $(date) ==="