#!/bin/bash
# training_job.sh

# --- Source Project Configuration ---
# CONFIG_FILE_PATH_ABS is passed from main_orchestrator.sh
if [ -z "$CONFIG_FILE_PATH_ABS" ]; then
    echo "CRITICAL ERROR: CONFIG_FILE_PATH_ABS not provided to training_job.sh"
    exit 1
fi
if [ -f "$CONFIG_FILE_PATH_ABS" ]; then
    echo "Sourcing project configuration from $CONFIG_FILE_PATH_ABS"
    source "$CONFIG_FILE_PATH_ABS"
else
    echo "CRITICAL ERROR: Project configuration file not found at $CONFIG_FILE_PATH_ABS"
    exit 1
fi
# --- End Source Project Configuration ---

#SBATCH --job-name=${TRAIN_JOB_NAME_PREFIX:-train}_${SHARED_RUN_ID} # Job name includes SHARED_RUN_ID
#SBATCH --partition=${TRAIN_JOB_PARTITION:-general_gpu_p6000}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${TRAIN_JOB_CPUS:-8}
#SBATCH --mem=${TRAIN_JOB_MEM:-64G}
#SBATCH --time=${TRAIN_JOB_TIME:-7-00:00:00}
#SBATCH --mail-type=${TRAIN_JOB_MAIL_TYPE:-END,FAIL}
#SBATCH --mail-user=${TRAIN_JOB_MAIL_USER:-your_email@example.com}

set -e
echo "=== Training Job Script Started: $(date) ==="
echo "Training Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Host Project Dir (from env): ${HOST_PROJECT_DIR}"
echo "Shared Output Directory on Host (from env): ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID (from env): ${SHARED_RUN_ID}"
echo "Seed for this Training Run (from env): ${SEED_FOR_TRAINING}"
echo "Checkpoint Ready Sentinel Filename (from env): ${CHECKPOINT_READY_SENTINEL_FILENAME}"

# Validate essential env vars (already done by main_orchestrator before passing, but good check)
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_HOST" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_TRAINING" ] || \
   [ -z "$CHECKPOINT_READY_SENTINEL_FILENAME" ]; then
    echo "CRITICAL ERROR (Training Job): Essential environment variables from orchestrator missing!"
    exit 1
fi

echo "Loading system modules: ${SYSTEM_MODULES_TRAIN:-singularity/4.1.1 cuda/11.8}"
module load ${SYSTEM_MODULES_TRAIN:-singularity/4.1.1 cuda/11.8}

NEPTUNE_CRED_FILE_PATH="${NEPTUNE_CRED_FILE_PATH:-$HOME/.neptune_creds}"
if [ -f "$NEPTUNE_CRED_FILE_PATH" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE_PATH"
    source "$NEPTUNE_CRED_FILE_PATH" # Should export NEPTUNE_API_TOKEN, NEPTUNE_PROJECT
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-${DEFAULT_NEPTUNE_PROJECT}}"
else
    echo "WARNING: Neptune credentials file '$NEPTUNE_CRED_FILE_PATH' not found."
    export SINGULARITYENV_NEPTUNE_PROJECT="${DEFAULT_NEPTUNE_PROJECT}"
fi
export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}" # Set by main_orchestrator

HOST_SIF_PATH_ABS="${HOST_PROJECT_DIR}/${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT}"
HOST_DATA_BASE_DIR_ABS="${HOST_PROJECT_DIR}/${DATA_DIR_RELATIVE_TO_PROJECT_ROOT}"

CONTAINER_WORKSPACE_PATH="${CONTAINER_WORKSPACE_PATH:-/workspace}"
CONTAINER_DATA_DIR_PATH="${CONTAINER_DATA_DIR_PATH:-/data}"
CONTAINER_OUTPUT_TARGET_DIR_PATH="${CONTAINER_OUTPUT_DIR_PATH:-/output_train}"

if [ ! -f "$HOST_SIF_PATH_ABS" ]; then echo "CRITICAL ERROR: Singularity image not found at ${HOST_SIF_PATH_ABS}"; exit 1; fi

echo "Starting Python training script (train.py) for Seed ${SEED_FOR_TRAINING}..."

# Construct train.py arguments from config variables
TRAIN_PY_ARGS=()
TRAIN_PY_ARGS+=( "--model" "${TRAIN_PY_MODEL_NAME:-gpt2}" )
TRAIN_PY_ARGS+=( "--model_size" "${TRAIN_PY_MODEL_SIZE:-100m}" )
TRAIN_PY_ARGS+=( "--train_dataset_path" "${CONTAINER_DATA_DIR_PATH}/${TRAIN_PY_TRAIN_DATASET_RELATIVE_PATH}" )
TRAIN_PY_ARGS+=( "--validation_dataset_path" "${CONTAINER_DATA_DIR_PATH}/${TRAIN_PY_VALID_DATASET_RELATIVE_PATH}" )
TRAIN_PY_ARGS+=( "--output_dir" "${CONTAINER_OUTPUT_TARGET_DIR_PATH}" )
TRAIN_PY_ARGS+=( "--seed" "${SEED_FOR_TRAINING}" )
TRAIN_PY_ARGS+=( "--checkpoint_ready_sentinel" "${CHECKPOINT_READY_SENTINEL_FILENAME}" )

TRAIN_PY_ARGS+=( "--num_train_epochs" "${TRAIN_PY_NUM_EPOCHS:-20}" )
TRAIN_PY_ARGS+=( "--per_device_train_batch_size" "${TRAIN_PY_BATCH_SIZE:-8}" )
TRAIN_PY_ARGS+=( "--gradient_accumulation_steps" "${TRAIN_PY_GRAD_ACCUM_STEPS:-16}" )
TRAIN_PY_ARGS+=( "--learning_rate" "${TRAIN_PY_LR:-3e-4}" )
TRAIN_PY_ARGS+=( "--lr_scheduler_type" "${TRAIN_PY_LR_SCHEDULER:-cosine}" )
TRAIN_PY_ARGS+=( "--num_warmup_steps" "${TRAIN_PY_WARMUP_STEPS:-1000}" )
TRAIN_PY_ARGS+=( "--weight_decay" "${TRAIN_PY_WEIGHT_DECAY:-0.01}" )
TRAIN_PY_ARGS+=( "--max_grad_norm" "${TRAIN_PY_MAX_GRAD_NORM:-1.0}" )

if [ "${TRAIN_PY_USE_AMP:-true}" == "true" ]; then TRAIN_PY_ARGS+=( "--use_amp" ); fi
TRAIN_PY_ARGS+=( "--num_workers" "${TRAIN_JOB_CPUS:-4}" ) # Use Slurm allocated CPUs for workers
TRAIN_PY_ARGS+=( "--logging_steps" "${TRAIN_PY_LOGGING_STEPS:-100}" )
TRAIN_PY_ARGS+=( "--save_steps" "${TRAIN_PY_SAVE_STEPS:-200}" )

# Neptune args for train.py
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    TRAIN_PY_ARGS+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
fi
TRAIN_PY_ARGS+=( "--neptune_run_name" "${SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME}" )
NEPTUNE_TAGS_STRING_FOR_TRAIN_PY="${NEPTUNE_TRAINING_TAGS_DEFAULT:-training_phase gpt2_10m} seed_${SEED_FOR_TRAINING} ${SHARED_RUN_ID}"
TRAIN_PY_ARGS+=( "--neptune_tags" ${NEPTUNE_TAGS_STRING_FOR_TRAIN_PY} ) # Note: tags are multiple args

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_TRAIN:-expandable_segments:True}

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE_PATH}" \
    -B "${HOST_DATA_BASE_DIR_ABS}":"${CONTAINER_DATA_DIR_PATH}" \
    -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_OUTPUT_TARGET_DIR_PATH}" \
    "${HOST_SIF_PATH_ABS}" \
    python3 "${CONTAINER_WORKSPACE_PATH}/${TRAIN_PY_SCRIPT_RELATIVE_PATH:-src/train.py}" \
    "${TRAIN_PY_ARGS[@]}"

TRAINING_COMPLETED_SENTINEL_PATH="${SHARED_OUTPUT_DIR_HOST}/${TRAINING_COMPLETION_SENTINEL_FILENAME:-TRAINING_COMPLETED.txt}"
touch "${TRAINING_COMPLETED_SENTINEL_PATH}"
echo "Training script (train.py) finished for Seed ${SEED_FOR_TRAINING}."
echo "Sentinel file ${TRAINING_COMPLETED_SENTINEL_PATH} created in ${SHARED_OUTPUT_DIR_HOST}."

echo "=== Training Job Script Finished for Seed ${SEED_FOR_TRAINING}: $(date) ==="