#!/bin/bash

# === SBATCH Directives (Unchanged) ===
#SBATCH --job-name=gpt2_p6000_sif_100m_LOCALEVAL # Modified job name slightly
#SBATCH --partition=general_gpu_p6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit on error
set -e

# === Environment Setup (Unchanged) ===
echo "=== Job Started (Local Eval Mode): $(date) ==="
echo "Current Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "Username: $USER"
echo "Home Directory: $HOME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules (Unchanged) ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Securely Load Neptune Credentials (Unchanged) ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    if [ -n "${NEPTUNE_PROJECT:-}" ]; then
      export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"
    fi
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE."
fi

# --- Define Paths on Host (Unchanged) ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif"
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

# --- Define Container Paths (Unchanged) ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/.output" # Output base mount inside container
CONTAINER_PRIMING_DIR="/eval"   # Priming base mount inside container

# --- Define Run Output Directory Name (Modified for clarity) ---
RUN_OUTPUT_NAME="gpt2_p6000_sif_local_eval_run" # Modified name
HOST_RUN_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${RUN_OUTPUT_NAME}"
CONTAINER_RUN_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}/${RUN_OUTPUT_NAME}" # Specific output dir for this run inside container

# --- Preparations (Unchanged) ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Data Base Directory (Host): ${HOST_DATA_BASE_DIR}"
echo "Output Base Directory (Host): ${HOST_OUTPUT_BASE_DIR}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_RUN_OUTPUT_DIR}"
echo "Ensured host output directory exists: ${HOST_RUN_OUTPUT_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Training Script Execution (Modified Arguments for Local Eval) ===
echo "Starting Python training script (with Local Eval) inside Singularity container..."

# --- Define paths relative to container mount points (Unchanged) ---
CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m"
CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_10m"
CONTAINER_PRIMING_PATH="${CONTAINER_PRIMING_DIR}/priming-corpuses"
CONTAINER_EVAL_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/evaluate.py" # Path to evaluate.py inside container

# --- Define Neptune args (Modified Tags) ---
NEPTUNE_PROJECT_ARG="" # Assume project set via SINGULARITYENV_ or leave empty if not using
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG="--neptune_project ${SINGULARITYENV_NEPTUNE_PROJECT}"
elif [ -n "${NEPTUNE_PROJECT:-}" ]; then # Fallback to shell variable if not passed via SINGULARITYENV_
    NEPTUNE_PROJECT_ARG="--neptune_project ${NEPTUNE_PROJECT}"
else
    NEPTUNE_PROJECT_ARG="--neptune_project thmorton/NothingProject" # Hardcoded fallback
fi

NEPTUNE_RUN_NAME="${RUN_OUTPUT_NAME}_$(date +%Y%m%d_%H%M)"
NEPTUNE_TAGS="p6000 test1 baseline singularity py39 local_eval" # Added local_eval tag

# --- Set PyTorch CUDA Allocator Config (Unchanged) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Execute Singularity Command (Modified Python Args) ---
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_OUTPUT_BASE_DIR}":"${CONTAINER_OUTPUT_DIR}" \
    -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/src/train.py" \
        --model "gpt2" \
        --train_dataset_path "${CONTAINER_DATA_DIR}/processed/training_set_100m" \
        --validation_dataset_path "${CONTAINER_DATA_DIR}/processed/test_set_10m" \
        --priming_eval_dir_path "${CONTAINER_PRIMING_PATH}" \
        --output_dir "${CONTAINER_RUN_OUTPUT_DIR}" \
        \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \

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
        --eval_steps 400 \
        --save_steps 400 \
        --priming_eval_steps 400 \
        \
        ${NEPTUNE_PROJECT_ARG} \
        --neptune_run_name "${NEPTUNE_RUN_NAME}" \
        --neptune_tags ${NEPTUNE_TAGS} \
        \
        --local_eval \
        --evaluate_script_path "${CONTAINER_EVAL_SCRIPT_PATH}" \
        --trigger_standard_eval \
        --trigger_priming_eval

# === Job Completion ===
echo "=== Job Finished: $(date) ==="