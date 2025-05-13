#!/bin/bash
# training_job.sh
# Can be run via sbatch (respects #SBATCH) or directly with bash for local testing.

# --- Handle Configuration File Path ---
# The first command-line argument to this script can be the path to the config file.
# If no argument is provided, it defaults to ../project_config.sh (relative to this script's location)

CONFIG_FILE_ARG="$1" # Get the first argument

if [ -n "$CONFIG_FILE_ARG" ]; then
    # Argument provided, use it
    CONFIG_FILE="$CONFIG_FILE_ARG"
    echo "Configuration file path provided as argument: $CONFIG_FILE"
else
    # No argument provided, use default relative path
    # Assuming this script is in a 'scripts' subdir and config is in parent project dir
    DEFAULT_CONFIG_PATH="$(dirname "$0")/../project_config.sh"
    CONFIG_FILE="$DEFAULT_CONFIG_PATH"
    echo "No configuration file path provided as argument. Defaulting to: $CONFIG_FILE"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "CRITICAL ERROR: Project configuration file not found at resolved path: $CONFIG_FILE"
    echo "Please provide the path as an argument or ensure it exists at the default location."
    exit 1
fi

# Make the config file path absolute and export it for child scripts
export CONFIG_FILE_PATH_ABS="$(readlink -f "$CONFIG_FILE")"
echo "Sourcing project configuration from absolute path: $CONFIG_FILE_PATH_ABS"
source "$CONFIG_FILE_PATH_ABS"
# --- End Handle Configuration File Path ---

# These might be passed by main_orchestrator.sh. If not, provide defaults for local run.
SHARED_RUN_ID="${SHARED_RUN_ID:-local_run_$(date +%Y%m%d_%H%M%S)}"
# SHARED_OUTPUT_DIR_HOST defaults to a subdirectory in HOST_PROJECT_DIR/src/.output (from config's OUTPUT_BASE_DIR_RELATIVE_TO_PROJECT_ROOT)
_output_base_abs="${HOST_PROJECT_DIR}/${OUTPUT_BASE_DIR_RELATIVE_TO_PROJECT_ROOT:-src/.output}"
SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST:-${_output_base_abs}/${SHARED_RUN_ID}}"
SEED_FOR_TRAINING="${SEED_FOR_TRAINING:-${SEED_FOR_STANDALONE:-42}}"
CHECKPOINT_READY_SENTINEL_FILENAME="${CHECKPOINT_READY_SENTINEL_FILENAME:-EVAL_READY.txt}" # Default from train.py

# Ensure the output directory for this run exists
mkdir -p "$SHARED_OUTPUT_DIR_HOST"
# --- End Set Defaults ---


#SBATCH --job-name=${TRAIN_JOB_NAME_PREFIX:-train}_${SHARED_RUN_ID}
#SBATCH --partition=${TRAIN_JOB_PARTITION:-general_gpu_p6000} # Ensure this requests 1 GPU if for single GPU run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # CRITICAL for single GPU: ensure only 1 task
#SBATCH --cpus-per-task=${TRAIN_JOB_CPUS:-8}
#SBATCH --mem=${TRAIN_JOB_MEM:-64G}
#SBATCH --time=${TRAIN_JOB_TIME:-7-00:00:00}
#SBATCH --gres=gpu:${TRAIN_JOB_GPU_REQUEST:-1} # CRITICAL for single GPU: ensure "1" or "type:1"
#SBATCH --mail-type=${TRAIN_JOB_MAIL_TYPE:-END,FAIL}
#SBATCH --mail-user=${TRAIN_JOB_MAIL_USER:-your_email@example.com}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
echo "=== Training Job Script Started: $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}" # Display LOCAL if not a Slurm job
echo "Job Name: ${SLURM_JOB_NAME:-train_local}"
echo "Host Project Dir: ${HOST_PROJECT_DIR}"
echo "Shared Output Directory on Host: ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID: ${SHARED_RUN_ID}"
echo "Seed for this Training Run: ${SEED_FOR_TRAINING}"
echo "Checkpoint Ready Sentinel Filename: ${CHECKPOINT_READY_SENTINEL_FILENAME}"

echo "Loading system modules: ${SYSTEM_MODULES_TRAIN:-singularity/4.1.1 cuda/11.8}"
module load ${SYSTEM_MODULES_TRAIN:-singularity/4.1.1 cuda/11.8}

NEPTUNE_CRED_FILE_PATH="${NEPTUNE_CRED_FILE_PATH:-$HOME/.neptune_creds}"
if [ -f "$NEPTUNE_CRED_FILE_PATH" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE_PATH"
    source "$NEPTUNE_CRED_FILE_PATH"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-${DEFAULT_NEPTUNE_PROJECT}}"
else
    echo "WARNING: Neptune credentials file '$NEPTUNE_CRED_FILE_PATH' not found."
    export SINGULARITYENV_NEPTUNE_PROJECT="${DEFAULT_NEPTUNE_PROJECT}"
fi
export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}"

HOST_SIF_PATH_ABS="${HOST_PROJECT_DIR}/${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT}"
HOST_DATA_BASE_DIR_ABS="${HOST_PROJECT_DIR}/${DATA_DIR_RELATIVE_TO_PROJECT_ROOT}"

CONTAINER_WORKSPACE_PATH="${CONTAINER_WORKSPACE_PATH:-/workspace}"
CONTAINER_DATA_DIR_PATH="${CONTAINER_DATA_DIR_PATH:-/data}"
CONTAINER_OUTPUT_TARGET_DIR_PATH="${CONTAINER_OUTPUT_DIR_PATH:-/output_train}" # train.py output dir inside container

if [ ! -f "$HOST_SIF_PATH_ABS" ]; then echo "CRITICAL ERROR: Singularity image not found at ${HOST_SIF_PATH_ABS}"; exit 1; fi

echo "Starting Python training script (train.py) for Seed ${SEED_FOR_TRAINING}..."

# Construct train.py arguments from config variables
TRAIN_PY_ARGS=()
TRAIN_PY_ARGS+=( "--model" "${TRAIN_PY_MODEL_NAME:-gpt2}" )
TRAIN_PY_ARGS+=( "--model_size" "${TRAIN_PY_MODEL_SIZE:-10m}" ) # Example default
TRAIN_PY_ARGS+=( "--train_dataset_path" "${CONTAINER_DATA_DIR_PATH}/${TRAIN_PY_TRAIN_DATASET_RELATIVE_PATH}" )
# Pass validation dataset path for train.py; local_eval can also use this.
TRAIN_PY_ARGS+=( "--validation_dataset_path" "${CONTAINER_DATA_DIR_PATH}/${TRAIN_PY_VALID_DATASET_RELATIVE_PATH}" )
TRAIN_PY_ARGS+=( "--output_dir" "${CONTAINER_OUTPUT_TARGET_DIR_PATH}" ) # train.py writes checkpoints here
TRAIN_PY_ARGS+=( "--seed" "${SEED_FOR_TRAINING}" )
TRAIN_PY_ARGS+=( "--checkpoint_ready_sentinel" "${CHECKPOINT_READY_SENTINEL_FILENAME}" )

TRAIN_PY_ARGS+=( "--num_train_epochs" "${TRAIN_PY_NUM_EPOCHS:-2}" ) # Example default
TRAIN_PY_ARGS+=( "--per_device_train_batch_size" "${TRAIN_PY_BATCH_SIZE:-8}" )
TRAIN_PY_ARGS+=( "--gradient_accumulation_steps" "${TRAIN_PY_GRAD_ACCUM_STEPS:-16}" )
TRAIN_PY_ARGS+=( "--learning_rate" "${TRAIN_PY_LR:-3e-4}" )
TRAIN_PY_ARGS+=( "--lr_scheduler_type" "${TRAIN_PY_LR_SCHEDULER:-cosine}" )
TRAIN_PY_ARGS+=( "--num_warmup_steps" "${TRAIN_PY_WARMUP_STEPS:-1000}" )
TRAIN_PY_ARGS+=( "--weight_decay" "${TRAIN_PY_WEIGHT_DECAY:-0.01}" )
TRAIN_PY_ARGS+=( "--max_grad_norm" "${TRAIN_PY_MAX_GRAD_NORM:-1.0}" )

if [ "${TRAIN_PY_USE_AMP:-true}" == "true" ]; then TRAIN_PY_ARGS+=( "--use_amp" ); fi
TRAIN_PY_ARGS+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-${TRAIN_JOB_CPUS:-4}}" )
TRAIN_PY_ARGS+=( "--logging_steps" "${TRAIN_PY_LOGGING_STEPS:-100}" )
TRAIN_PY_ARGS+=( "--save_steps" "${TRAIN_PY_SAVE_STEPS:-200}" ) # Example default

# --- Add arguments for local evaluation based on config ---
# These TRAIN_PY_... variables for local eval should be in your project_config.sh
if [ "${TRAIN_PY_LOCAL_EVAL:-false}" == "true" ]; then
    TRAIN_PY_ARGS+=( "--local_eval" )
    TRAIN_PY_ARGS+=( "--evaluate_script_path" "${TRAIN_PY_EVALUATE_SCRIPT_PATH:-src/evaluate.py}" ) # Path to evaluate.py script

    if [ "${TRAIN_PY_TRIGGER_STANDARD_EVAL:-false}" == "true" ]; then
        TRAIN_PY_ARGS+=( "--trigger_standard_eval" )
        # --validation_dataset_path is already added above for train.py, local eval will use it.
    fi
    if [ "${TRAIN_PY_TRIGGER_PRIMING_EVAL:-false}" == "true" ]; then
        TRAIN_PY_ARGS+=( "--trigger_priming_eval" )
        # train.py's parser needs --priming_eval_dir_path if local eval is to trigger it.
        # This path is relative to HOST_PROJECT_DIR, for train.py to pass to evaluate.py
        if [ -n "${TRAIN_PY_PRIMING_EVAL_DIR_CONFIG_PATH}" ]; then # e.g., "eval/priming-corpuses"
             TRAIN_PY_ARGS+=( "--priming_eval_dir_path" "${TRAIN_PY_PRIMING_EVAL_DIR_CONFIG_PATH}" )
        else
            echo "WARNING: TRAIN_PY_LOCAL_EVAL and TRAIN_PY_TRIGGER_PRIMING_EVAL are true, but TRAIN_PY_PRIMING_EVAL_DIR_CONFIG_PATH is not set in config."
        fi
    fi
    TRAIN_PY_ARGS+=( "--eval_steps" "${TRAIN_PY_EVAL_STEPS:-200}" ) # Example default
    # priming_eval_steps defaults to eval_steps in train.py if not specified
    if [ -n "${TRAIN_PY_PRIMING_EVAL_STEPS}" ]; then
        TRAIN_PY_ARGS+=( "--priming_eval_steps" "${TRAIN_PY_PRIMING_EVAL_STEPS}" )
    fi
fi
# --- End local evaluation arguments ---

# Neptune args for train.py
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    TRAIN_PY_ARGS+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
fi
TRAIN_PY_ARGS+=( "--neptune_run_name" "${SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME}" )
NEPTUNE_TAGS_STRING_FOR_TRAIN_PY="${NEPTUNE_TRAINING_TAGS_DEFAULT:-training_phase model_10m} seed_${SEED_FOR_TRAINING} ${SHARED_RUN_ID}" # Adjusted for 10m example
TRAIN_PY_ARGS+=( "--neptune_tags" ${NEPTUNE_TAGS_STRING_FOR_TRAIN_PY} )

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_TRAIN:-expandable_segments:True}

# Ensure CUDA_VISIBLE_DEVICES is set for Singularity if running locally for single GPU
if [ -z "$SLURM_JOB_ID" ]; then # Likely a local run
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} # Default to GPU 0 if not set
    echo "INFO: Running locally, setting CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "Final train.py arguments:"
printf "  %q\n" "${TRAIN_PY_ARGS[@]}"

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