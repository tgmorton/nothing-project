#!/bin/bash
# eval_job.sh (Long-running multi-checkpoint/watcher evaluation job)

# --- Source Project Configuration ---
if [ -z "$CONFIG_FILE_PATH_ABS" ]; then
    echo "CRITICAL ERROR: CONFIG_FILE_PATH_ABS not provided to eval_job.sh"
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

# Slurm directives for this long-running job
# Job name and output/error paths are set by eval_orchestrator.sh (the submitter)
#SBATCH --partition=${EVAL_JOB_PARTITION:-general_gpu_p6000}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${EVAL_JOB_CPUS:-4}
#SBATCH --mem=${EVAL_JOB_MEM:-32G}
#SBATCH --time=${EVAL_JOB_TIME:-7-00:00:00} # Max time from config, e.g., 7 days
#SBATCH --gres=gpu:${EVAL_JOB_GPU_REQUEST:-1} # e.g., "1" or "a100:1"
#SBATCH --mail-type=${EVAL_JOB_MAIL_TYPE:-FAIL,END}
#SBATCH --mail-user=${EVAL_JOB_MAIL_USER:-your_email@example.com}

set -e
log_eval_job_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - MULTI_EVAL_JOB ($SLURM_JOB_ID): $1"
}

log_eval_job_message "=== Multi-Checkpoint/Watcher Evaluation Job Script Started ==="
log_eval_job_message "Slurm Job Name: $SLURM_JOB_NAME"
# ... other initial log messages ...

# Env vars from submitter:
# HOST_PROJECT_DIR, SHARED_OUTPUT_DIR_TO_WATCH, SHARED_RUN_ID,
# SEED_FOR_THIS_MULTI_EVAL, CHECKPOINT_READY_SENTINEL_FILENAME,
# EVALUATE_PY_OVERALL_OUTPUT_DIR_HOST (this is where evaluate.py will create subdirs)

# Validate received env vars (already done by submitter, but good check)
# ...

log_eval_job_message "Loading system modules: ${SYSTEM_MODULES_EVAL:-singularity/4.1.1 cuda/11.8}"
module load ${SYSTEM_MODULES_EVAL:-singularity/4.1.1 cuda/11.8}

NEPTUNE_CRED_FILE_PATH_EVAL="${NEPTUNE_CRED_FILE_PATH_EVAL:-$HOME/.neptune_creds}" # Eval might use different creds/project
if [ -f "$NEPTUNE_CRED_FILE_PATH_EVAL" ]; then
    source "$NEPTUNE_CRED_FILE_PATH_EVAL"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-${DEFAULT_NEPTUNE_PROJECT_EVAL}}" # Separate default for eval
else
    log_eval_job_message "WARNING: Neptune credentials file '$NEPTUNE_CRED_FILE_PATH_EVAL' not found."
    export SINGULARITYENV_NEPTUNE_PROJECT="${DEFAULT_NEPTUNE_PROJECT_EVAL}"
fi
export SINGULARITYENV_SHARED_RUN_ID="${SHARED_RUN_ID}" # For evaluate.py to link/name its Neptune session

HOST_SIF_PATH_EVAL_ABS="${HOST_PROJECT_DIR}/${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT_EVAL:-${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT}}"
HOST_DATA_BASE_DIR_EVAL_ABS="${HOST_PROJECT_DIR}/${DATA_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL:-${DATA_DIR_RELATIVE_TO_PROJECT_ROOT}}"
HOST_PRIMING_BASE_DIR_EVAL_ABS="${HOST_PROJECT_DIR}/${PRIMING_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL:-eval}"


CONTAINER_WORKSPACE_PATH_EVAL="${CONTAINER_WORKSPACE_PATH_EVAL:-/workspace}"
CONTAINER_DATA_DIR_PATH_EVAL="${CONTAINER_DATA_DIR_PATH_EVAL:-/data_eval_mnt}"
CONTAINER_PRIMING_DIR_PATH_EVAL="${CONTAINER_PRIMING_DIR_PATH_EVAL:-/priming_data_eval_mnt}"
CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH="${CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH:-/checkpoints_to_watch}"
CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH="${CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH:-/eval_results_base}"

if [ ! -f "$HOST_SIF_PATH_EVAL_ABS" ]; then log_eval_job_message "CRITICAL ERROR: SIF image not found at ${HOST_SIF_PATH_EVAL_ABS}"; exit 1; fi

log_eval_job_message "Starting evaluate.py in multi-checkpoint/watch mode..."
EVAL_PY_SCRIPT_PATH_CONTAINER="${CONTAINER_WORKSPACE_PATH_EVAL}/${EVAL_PY_SCRIPT_RELATIVE_PATH:-src/evaluate.py}"

# Construct evaluate.py arguments from config variables
EVAL_PY_ARGS=()
EVAL_PY_ARGS+=( "--watch_mode" )
EVAL_PY_ARGS+=( "--checkpoint_dir" "${CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH}" )
EVAL_PY_ARGS+=( "--output_dir" "${CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH}" )
EVAL_PY_ARGS+=( "--checkpoint_ready_sentinel" "${CHECKPOINT_READY_SENTINEL_FILENAME}" )
EVAL_PY_ARGS+=( "--watch_interval_seconds" "${EVAL_PY_WATCH_INTERVAL_SECONDS:-300}" )

EVAL_PY_ARGS+=( "--base_model_name_or_path" "${EVAL_PY_BASE_MODEL_NAME_OR_PATH:-gpt2}" )
EVAL_PY_ARGS+=( "--model_class_name" "${EVAL_PY_MODEL_CLASS_NAME:-GPT2LMHeadModel}" )

if [ "${EVAL_PY_RUN_STANDARD_EVAL:-true}" == "true" ]; then
    EVAL_PY_ARGS+=( "--run_standard_eval" )
    EVAL_PY_ARGS+=( "--validation_dataset_path" "${CONTAINER_DATA_DIR_PATH_EVAL}/${EVAL_PY_VALID_DATASET_RELATIVE_PATH}" )
    EVAL_PY_ARGS+=( "--eval_max_samples" "${EVAL_PY_STD_EVAL_MAX_SAMPLES:-50000}" )
fi

if [ "${EVAL_PY_RUN_PRIMING_EVAL:-true}" == "true" ]; then
    EVAL_PY_ARGS+=( "--run_priming_eval" )
    EVAL_PY_ARGS+=( "--priming_eval_dir_path" "${CONTAINER_PRIMING_DIR_PATH_EVAL}/${EVAL_PY_PRIMING_DIR_RELATIVE_PATH}" )
    EVAL_PY_ARGS+=( "--priming_eval_max_samples_per_file" "${EVAL_PY_PRIMING_MAX_SAMPLES_PER_FILE:-1000}" )
    EVAL_PY_ARGS+=( "--priming_delimiter" "${EVAL_PY_PRIMING_DELIMITER:-.}")
fi

EVAL_PY_ARGS+=( "--per_device_eval_batch_size" "${EVAL_PY_STD_BATCH_SIZE:-16}" )
EVAL_PY_ARGS+=( "--priming_per_device_eval_batch_size" "${EVAL_PY_PRIMING_BATCH_SIZE:-8}" )
EVAL_PY_ARGS+=( "--num_workers" "${EVAL_JOB_CPUS:-4}" ) # Use Slurm allocated CPUs
EVAL_PY_ARGS+=( "--seed" "${SEED_FOR_THIS_MULTI_EVAL}" )
if [ "${EVAL_PY_USE_AMP:-true}" == "true" ]; then EVAL_PY_ARGS+=( "--use_amp" ); fi

if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    EVAL_PY_ARGS+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
    # Neptune tags for the overall eval session can be added here if desired
    # EVAL_PY_ARGS+=( "--neptune_tags" "eval_session" "${SHARED_RUN_ID}" )
fi

log_eval_job_message "--- Arguments for evaluate.py ---"
printf "  %q\n" "${EVAL_PY_ARGS[@]}"
log_eval_job_message "---------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_EVAL:-expandable_segments:True}
EVALUATE_PY_CONSOLE_LOG="${EVALUATE_PY_OVERALL_OUTPUT_DIR_HOST}/evaluate_py_multi_console_output_${SLURM_JOB_ID}.log"

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE_PATH_EVAL}" \
    -B "${HOST_DATA_BASE_DIR_EVAL_ABS}":"${CONTAINER_DATA_DIR_PATH_EVAL}" \
    -B "${HOST_PRIMING_BASE_DIR_EVAL_ABS}":"${CONTAINER_PRIMING_DIR_PATH_EVAL}" \
    -B "${SHARED_OUTPUT_DIR_TO_WATCH}":"${CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH}" \
    -B "${EVALUATE_PY_OVERALL_OUTPUT_DIR_HOST}":"${CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH}" \
    "${HOST_SIF_PATH_EVAL_ABS}" \
    python3 "${EVAL_PY_SCRIPT_PATH_CONTAINER}" \
    "${EVAL_PY_ARGS[@]}" > "${EVALUATE_PY_CONSOLE_LOG}" 2>&1

EVALUATE_PY_EXIT_CODE=$?
if [ $EVALUATE_PY_EXIT_CODE -ne 0 ]; then
    log_eval_job_message "CRITICAL ERROR: evaluate.py (multi/watch) FAILED with exit code ${EVALUATE_PY_EXIT_CODE}."
    # Copy log to submit dir for easier access on failure
    cp "${EVALUATE_PY_CONSOLE_LOG}" "${SLURM_SUBMIT_DIR}/FAILED_evaluate_py_multi_console_${SLURM_JOB_ID}.log"
    exit 1
fi
log_eval_job_message "evaluate.py (multi/watch) completed successfully."
log_eval_job_message "=== Multi-Checkpoint/Watcher Evaluation Job Script Finished Successfully ==="