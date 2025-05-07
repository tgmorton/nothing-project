#!/bin/bash

# === SBATCH Directives for 9x 10M Seed Sweep ===
#SBATCH --job-name=train_9x10M_seeds  # <<< Updated Job Name
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00          # <<< Increased time significantly for 9 runs (9 days)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL # Added FAIL notification
#SBATCH --mail-user=thmorton@ucsd.edu

# Exit on error (important for sequential runs)
set -e

# === Environment Setup ===
echo "=== Job Started (9x 10M Seed Sweep): $(date) ==="
echo "Current Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "Username: $USER"
echo "Home Directory: $HOME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Securely Load Neptune Credentials ---
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

# --- Define Paths on Host ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif"
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/.output" # Output base mount inside container
CONTAINER_PRIMING_DIR="/eval"   # Priming base mount inside container

# --- Define Base Output Directory Name for the Job ---
JOB_BASE_OUTPUT_NAME="gpt2_10m_seeds_1-9_$(date +%Y%m%d_%H%M)" # Base name for this job's outputs
HOST_JOB_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${JOB_BASE_OUTPUT_NAME}"
CONTAINER_JOB_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}/${JOB_BASE_OUTPUT_NAME}" # Relative path inside container

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Data Base Directory (Host): ${HOST_DATA_BASE_DIR}"
echo "Output Base Directory (Host): ${HOST_OUTPUT_BASE_DIR}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi

# Create the base output directory for the entire job
mkdir -p "${HOST_JOB_OUTPUT_DIR}"
echo "Ensured base host output directory exists: ${HOST_JOB_OUTPUT_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs" # Ensure logs directory exists

# --- Define paths relative to container mount points ---
CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m" # Check if train/valid are swapped in original? Usually train_set is larger.
CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_10m" # Assuming original script's naming convention was intentional.
CONTAINER_PRIMING_PATH="${CONTAINER_PRIMING_DIR}/priming-corpuses"
CONTAINER_EVAL_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/evaluate.py"

# --- Define Base Neptune args ---
BASE_NEPTUNE_TAGS="gpt2 10m baseline singularity py39 local_eval" # Base tags for all runs
NEPTUNE_PROJECT_ARG=""
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG="--neptune_project ${SINGULARITYENV_NEPTUNE_PROJECT}"
elif [ -n "${NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG="--neptune_project ${NEPTUNE_PROJECT}"
else
    NEPTUNE_PROJECT_ARG="--neptune_project thmorton/NothingProject" # Hardcoded fallback
fi

# --- Set PyTorch CUDA Allocator Config ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Training Loop ===
echo "Starting sequential training for Seeds 1 to 9..."

for SEED in {1..9}
do
    echo "---------------------------------------------"
    echo "--- Starting Training for Seed ${SEED} ---"
    echo "---------------------------------------------"

    # --- Define Seed-Specific Output Directory ---
    SEED_RUN_OUTPUT_NAME="seed_${SEED}"
    HOST_SEED_OUTPUT_DIR="${HOST_JOB_OUTPUT_DIR}/${SEED_RUN_OUTPUT_NAME}"
    CONTAINER_SEED_OUTPUT_DIR="${CONTAINER_JOB_OUTPUT_DIR}/${SEED_RUN_OUTPUT_NAME}" # Relative output dir for this seed run inside container
    mkdir -p "${HOST_SEED_OUTPUT_DIR}"
    echo "Ensured host output directory for seed ${SEED} exists: ${HOST_SEED_OUTPUT_DIR}"

    # --- Define Seed-Specific Neptune Args ---
    NEPTUNE_RUN_NAME="${JOB_BASE_OUTPUT_NAME}_seed${SEED}" # Unique Neptune run name per seed
    SEED_NEPTUNE_TAGS="${BASE_NEPTUNE_TAGS} seed_${SEED}" # Add seed tag

    # --- Execute Singularity Command for the current seed ---
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
            --priming_eval_dir_path "${CONTAINER_PRIMING_PATH}" \
            --output_dir "${CONTAINER_SEED_OUTPUT_DIR}" \
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
            --model_size "10m" \
            \
            --use_amp \
            --num_workers ${SLURM_CPUS_PER_TASK:-4} \
            --seed ${SEED} \
            \
            --logging_steps 50 \
            --eval_steps 100 \
            --priming_eval_steps 100 \
            \
            ${NEPTUNE_PROJECT_ARG} \
            --neptune_run_name "${NEPTUNE_RUN_NAME}" \
            --neptune_tags ${SEED_NEPTUNE_TAGS} \
            \
            --local_eval \
            --evaluate_script_path "${CONTAINER_EVAL_SCRIPT_PATH}" \
            --trigger_standard_eval \
            --trigger_priming_eval

    echo "--- Finished Training for Seed ${SEED} ---"

done

# === Job Completion ===
echo "---------------------------------------------"
echo "=== All Seed Runs Completed: $(date) ==="
echo "---------------------------------------------"
