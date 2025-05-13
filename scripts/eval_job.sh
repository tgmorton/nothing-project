#!/bin/bash
# eval_job.sh
# This script is now a LONG-RUNNING job that processes multiple checkpoints
# using evaluate.py in normal or watch mode.

# === SBATCH Directives for this Multi-Checkpoint/Watcher Evaluation Job ===
# Job name is set by the submitter (eval_orchestrator.sh)
# Output/error paths are set by the submitter.
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU partition for actual evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for evaluate.py (adjust as needed)
#SBATCH --mem=32G                        # RAM for evaluate.py (adjust as needed)
#SBATCH --time=7-00:00:00                # <<< MAX TIME for this long-running job (e.g., 7 days)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mail-type=FAIL,END             # Notify on failure or end of this long job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

set -e # Exit on error for this script

log_eval_job_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - MULTI_EVAL_JOB ($SLURM_JOB_ID): $1"
}

log_eval_job_message "=== Multi-Checkpoint/Watcher Evaluation Job Script Started ==="
log_eval_job_message "Slurm Job Name: $SLURM_JOB_NAME"
log_eval_job_message "Running on Node: $SLURMD_NODENAME"
log_eval_job_message "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE, GPUs: $CUDA_VISIBLE_DEVICES"
log_eval_job_message "Working Directory: $(pwd)"

# --- Echo variables received from eval_orchestrator.sh via --export ---
log_eval_job_message "--- Received Environment Variables from Submitter ---"
log_eval_job_message "Host Project Dir: ${HOST_PROJECT_DIR}"
log_eval_job_message "Shared Output Dir to Watch (Host): ${SHARED_OUTPUT_DIR_TO_WATCH}"
log_eval_job_message "Shared Run ID (for context/tagging): ${SHARED_RUN_ID}"
log_eval_job_message "Seed for this Multi-Evaluation: ${SEED_FOR_THIS_MULTI_EVAL}"
log_eval_job_message "Checkpoint Ready Sentinel Filename: ${CHECKPOINT_READY_SENTINEL_FILENAME}"
log_eval_job_message "Evaluate.py Overall Output Dir (Host): ${EVALUATE_PY_OVERALL_OUTPUT_DIR}"
log_eval_job_message "----------------------------------------------------"

# --- Validate received variables ---
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_TO_WATCH" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_THIS_MULTI_EVAL" ] || \
   [ -z "$CHECKPOINT_READY_SENTINEL_FILENAME" ] || \
   [ -z "$EVALUATE_PY_OVERALL_OUTPUT_DIR" ]; then
    log_eval_job_message "CRITICAL ERROR: One or more essential environment variables not set by submitter! Dumping relevant env:"
    env | grep -E 'HOST_PROJECT_DIR|SHARED_OUTPUT_DIR_TO_WATCH|SHARED_RUN_ID|SEED_FOR_THIS_MULTI_EVAL|CHECKPOINT_READY_SENTINEL_FILENAME|EVALUATE_PY_OVERALL_OUTPUT_DIR'
    exit 1
fi
log_eval_job_message "All essential environment variables seem to be set."

# --- Load system modules ---
log_eval_job_message "Loading system modules: singularity, cuda..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed
log_eval_job_message "Modules loaded."

# --- Neptune Credentials for Singularity ---
log_eval_job_message "Setting up Neptune credentials for Singularity environment..."
NEPTUNE_CRED_FILE_EVAL_JOB="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE_EVAL_JOB" ]; then
    log_eval_job_message "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE_EVAL_JOB"
    source "$NEPTUNE_CRED_FILE_EVAL_JOB"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-thmorton/NothingProject}" # <<< UPDATE default
else
    log_eval_job_message "WARNING: Neptune credentials file '$NEPTUNE_CRED_FILE_EVAL_JOB' not found."
    export SINGULARITYENV_NEPTUNE_PROJECT="${SINGULARITYENV_NEPTUNE_PROJECT:-thmorton/NothingProject}" # Fallback
fi
log_eval_job_message "SINGULARITYENV_NEPTUNE_PROJECT set to: ${SINGULARITYENV_NEPTUNE_PROJECT}"
# Pass SHARED_RUN_ID to help evaluate.py construct Neptune run names or link to training
export SINGULARITYENV_SHARED_RUN_ID="${SHARED_RUN_ID}"
# The training run name might be useful for linking in Neptune
# If training_job.sh sets SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME, it could be passed here too.
# For now, evaluate.py can use SINGULARITYENV_SHARED_RUN_ID to build its session name.


# --- Define Paths on Host ---
log_eval_job_message "Defining host paths..."
HOST_SIF_PATH_EVAL_JOB="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE SIF name
HOST_DATA_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/data" # For validation dataset
HOST_PRIMING_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/eval" # For priming CSVs

# --- Define Container Mount Points ---
log_eval_job_message "Defining container mount points..."
CONTAINER_WORKSPACE_EVAL_JOB="/workspace"
CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB="/data_eval_mnt"
CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB="/priming_data_eval_mnt"
CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT="/checkpoints_to_watch" # SHARED_OUTPUT_DIR_TO_WATCH mounted here
CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT="/eval_results_base" # EVALUATE_PY_OVERALL_OUTPUT_DIR mounted here

# --- Preparations ---
if [ ! -f "$HOST_SIF_PATH_EVAL_JOB" ]; then
    log_eval_job_message "CRITICAL ERROR: Singularity image not found at ${HOST_SIF_PATH_EVAL_JOB}"; exit 1;
fi
# EVALUATE_PY_OVERALL_OUTPUT_DIR is created by eval_orchestrator.sh (submitter)
log_eval_job_message "evaluate.py will write its results into subdirectories of (host path): ${EVALUATE_PY_OVERALL_OUTPUT_DIR}"

log_eval_job_message "Starting evaluate.py in multi-checkpoint/watch mode..."

EVAL_PY_SCRIPT_CONTAINER_PATH="${CONTAINER_WORKSPACE_EVAL_JOB}/src/evaluate.py"
# Paths for datasets needed by evaluate.py (these are fixed for the run)
EVAL_PY_VALID_DATA_CONTAINER_PATH="${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}/processed/test_set_10m" # <<< ADJUST
EVAL_PY_PRIMING_DATA_CONTAINER_PATH="${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}/priming-corpuses" # <<< ADJUST
EVAL_PY_BASE_MODEL_PATH_CONTAINER="${CONTAINER_WORKSPACE_EVAL_JOB}/models/gpt2_base" # Example, if you have a base model snapshot, else use 'gpt2' string

# --- Construct arguments for evaluate.py ---
log_eval_job_message "Constructing arguments for evaluate.py..."
declare -a PYTHON_ARGS_FOR_EVALUATE_PY

# Core mode selection: watch the directory where checkpoints appear
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--watch_mode" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_dir" "${CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT}" ) # Dir to scan/watch
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--output_dir" "${CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT}" ) # Base for all outputs
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_ready_sentinel" "${CHECKPOINT_READY_SENTINEL_FILENAME}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--watch_interval_seconds" "5" ) # <<< PARAMETERIZE: e.g., 5 minutes

# Base model info
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--base_model_name_or_path" "gpt2" ) # <<< UPDATE if using a custom base path
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--model_class_name" "GPT2LMHeadModel" )

# Evaluation types to run for each checkpoint
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_standard_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--validation_dataset_path" "$EVAL_PY_VALID_DATA_CONTAINER_PATH" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_priming_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_eval_dir_path" "$EVAL_PY_PRIMING_DATA_CONTAINER_PATH" )

# Batch sizes, workers, seed, AMP
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--per_device_eval_batch_size" "16" ) # <<< PARAMETERIZE
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_per_device_eval_batch_size" "8" ) # <<< PARAMETERIZE
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--seed" "${SEED_FOR_THIS_MULTI_EVAL}" )
if [ "${USE_AMP_FOR_EVAL:-true}" == "true" ]; then # Control AMP via env var if needed
    PYTHON_ARGS_FOR_EVALUATE_PY+=( "--use_amp" )
fi

# Neptune integration (evaluate.py will use these to init its session)
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
    # evaluate.py will use SHARED_RUN_ID from env to construct its own Neptune run name
    # and can log individual checkpoint metrics to it.
fi
# Potentially add --neptune_tags "multi_eval_job" etc. if needed by evaluate.py

log_eval_job_message "--- Arguments for evaluate.py ---"
printf "  %q\n" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}"
log_eval_job_message "---------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Console output from evaluate.py for the entire long run
EVALUATE_PY_CONSOLE_LOG="${EVALUATE_PY_OVERALL_OUTPUT_DIR}/evaluate_py_multi_console_output.log"
log_eval_job_message "Console output of the long-running evaluate.py (stdout/stderr) will be redirected to: ${EVALUATE_PY_CONSOLE_LOG}"
# Ensure the base output directory exists (already done by submitter)

log_eval_job_message "Executing Singularity command for evaluate.py..."
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE_EVAL_JOB}" \
    -B "${HOST_DATA_BASE_DIR_EVAL_JOB}":"${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${HOST_PRIMING_BASE_DIR_EVAL_JOB}":"${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${SHARED_OUTPUT_DIR_TO_WATCH}":"${CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT}" \
    -B "${EVALUATE_PY_OVERALL_OUTPUT_DIR}":"${CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT}" \
    "${HOST_SIF_PATH_EVAL_JOB}" \
    python3 "${EVAL_PY_SCRIPT_CONTAINER_PATH}" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}" > "${EVALUATE_PY_CONSOLE_LOG}" 2>&1

EVALUATE_PY_EXIT_CODE=$?

if [ $EVALUATE_PY_EXIT_CODE -ne 0 ]; then
    log_eval_job_message "CRITICAL ERROR: evaluate.py (multi-checkpoint/watch mode) FAILED with exit code ${EVALUATE_PY_EXIT_CODE}."
    log_eval_job_message "Check evaluate.py console output at: ${EVALUATE_PY_CONSOLE_LOG}"
    log_eval_job_message "Also check this Slurm job's main error file."
    # Copy the detailed log to a more prominent place if it fails
    cp "${EVALUATE_PY_CONSOLE_LOG}" "${SLURM_SUBMIT_DIR}/FAILED_evaluate_py_multi_console_output_${SLURM_JOB_ID}.log"
    exit 1
fi

log_eval_job_message "evaluate.py (multi-checkpoint/watch mode) completed successfully."
log_eval_job_message "evaluate.py console output logged to: ${EVALUATE_PY_CONSOLE_LOG}"
log_eval_job_message "Individual checkpoint results should be in subdirectories of (host path): ${EVALUATE_PY_OVERALL_OUTPUT_DIR}"
log_eval_job_message "=== Multi-Checkpoint/Watcher Evaluation Job Script Finished Successfully ==="