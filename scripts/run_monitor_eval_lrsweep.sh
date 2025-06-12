#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=gpt2_eval_monitor_lrsweep
#SBATCH --partition=general_gpu_p6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=7-0:00:00
#SBATCH --output=../logs/%x_%j_lr%a.out
#SBATCH --error=../logs/%x_%j_lr%a.err
#SBATCH --array=1-5

set -e

# --- Define Learning Rates ---
LEARNING_RATES=(1e-5 3e-5 1e-4 3e-4 6e-4)
LR_NAMES=(1em5 3em5 1em4 3em4 6em4)
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
LR_INDEX=$((TASK_ID - 1))
CURRENT_LR=${LEARNING_RATES[$LR_INDEX]}
CURRENT_LR_NAME=${LR_NAMES[$LR_INDEX]}

# === Environment Setup ===
echo "=== Job Started (Evaluation Monitor Mode for LR Sweep): $(date) ==="
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID, Selected LR: $CURRENT_LR (Name: $CURRENT_LR_NAME)"
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

# THE ROBUST FIX: Wait for the network filesystem to be ready before sourcing.
# This loop actively checks if the module system's core TCL script is readable,
# preventing race conditions on job startup.
MODULE_INIT_SCRIPT="/etc/profile.d/modules.sh"
MODULE_CORE_FILE="/sscf/ssrde-storage/apps/modules/libexec/modulecmd.tcl"
WAIT_SECONDS=10
echo "Waiting up to ${WAIT_SECONDS}s for module system at ${MODULE_CORE_FILE} to be available..."
for ((i=0; i<WAIT_SECONDS; i++)); do
    if [ -r "${MODULE_CORE_FILE}" ]; then
        echo "Module system is ready."
        break
    fi
    sleep 1
done

if ! [ -r "${MODULE_CORE_FILE}" ]; then
    echo "FATAL: Module system core file was not readable after ${WAIT_SECONDS} seconds."
    exit 1
fi

# Now that we've confirmed the filesystem is ready, we can safely source.
source "${MODULE_INIT_SCRIPT}"

# This command will now work correctly
module load singularity/4.1.1 cuda/11.8

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
HOST_SIF_PATH="/home/AD/thmorton/nothing-project/python39_llm_env.sif"
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
BASE_RUN_OUTPUT_NAME="May20_s42_lr_sweep"
RUN_OUTPUT_NAME="${BASE_RUN_OUTPUT_NAME}_lr-${CURRENT_LR_NAME}"
HOST_TRAINED_MODEL_PARENT_DIR="${HOST_PROJECT_DIR}/models/${BASE_RUN_OUTPUT_NAME}/${RUN_OUTPUT_NAME}"
HOST_EVAL_MONITOR_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/models/${BASE_RUN_OUTPUT_NAME}/${RUN_OUTPUT_NAME}/eval"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_TRAINED_MODEL_PARENT_DIR="/trained_model_checkpoints"
CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR="/eval_monitor_outputs"
CONTAINER_PRIMING_DIR="/eval"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Data Base Directory (Host): ${HOST_DATA_BASE_DIR}"
echo "Trained Model Parent Directory (Host - Checkpoints Location): ${HOST_TRAINED_MODEL_PARENT_DIR}"
echo "Evaluation Monitor Output Base Directory (Host): ${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}"

if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "$HOST_TRAINED_MODEL_PARENT_DIR" ]; then echo "ERROR: Trained model parent directory not found at $HOST_TRAINED_MODEL_PARENT_DIR"; exit 1; fi

mkdir -p "${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}"
echo "Ensured host evaluation monitor output directory exists: ${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Evaluation Monitor Script Execution ===
echo "Starting Python evaluation_monitor.py script inside Singularity container for LR ${CURRENT_LR} (Name: ${CURRENT_LR_NAME})..."

# --- Define paths relative to container mount points ---
CONTAINER_MONITOR_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/evaluation_monitor.py"
CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m"
CONTAINER_PRIMING_PATH="${CONTAINER_PRIMING_DIR}/just_shota"

# --- Define Neptune args ---
NEPTUNE_PROJECT_ARG=""
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG="--neptune_project ${SINGULARITYENV_NEPTUNE_PROJECT}"
elif [ -n "${NEPTUNE_PROJECT:-}" ]; then
    NEPTUNE_PROJECT_ARG="--neptune_project ${NEPTUNE_PROJECT}"
else
    NEPTUNE_PROJECT_ARG="--neptune_project thmorton/NothingProject" # Hardcoded fallback
fi

CONCEPTUAL_TRAINING_RUN_NAME="${RUN_OUTPUT_NAME}"
NEPTUNE_TAGS_FOR_MONITOR="p6000 eval_monitor singularity py39 lr_sweep lr_${CURRENT_LR_NAME}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WATCH_MODE_ARG="" # Default: Run once and exit

# --- Execute Singularity Command ---
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_TRAINED_MODEL_PARENT_DIR}":"${CONTAINER_TRAINED_MODEL_PARENT_DIR}" \
    -B "${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}":"${CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR}" \
    -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_MONITOR_SCRIPT_PATH}" \
        --model_parent_dir "${CONTAINER_TRAINED_MODEL_PARENT_DIR}" \
        --output_base_dir "${CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR}" \
        ${WATCH_MODE_ARG} \
        \
        --run_priming_eval \
        --priming_eval_dir_path "${CONTAINER_PRIMING_PATH}" \
        \
        --base_model_name "gpt2" \
        --model_class_name "GPT2LMHeadModel" \
        --per_device_eval_batch_size 8 \
        --priming_per_device_eval_batch_size 8 \
        --eval_max_samples 5000 \
        --priming_eval_max_samples_per_file 1000 \
        \
        --use_amp \
        --num_workers ${SLURM_CPUS_PER_TASK:-4} \
        --seed 42 \
        --no_skip_processed_checkpoints \
        \
        ${NEPTUNE_PROJECT_ARG} \
        --neptune_tags ${NEPTUNE_TAGS_FOR_MONITOR} \
        --neptune_training_run_name "${CONCEPTUAL_TRAINING_RUN_NAME}" \
        --run_standard_eval \
        --validation_dataset_path "${CONTAINER_VALID_DATA_PATH}"

# === Job Completion ===
echo "=== Job Finished for LR ${CURRENT_LR} (Name: ${CURRENT_LR_NAME}): $(date) ==="
