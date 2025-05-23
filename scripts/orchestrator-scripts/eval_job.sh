#!/bin/bash
# eval_job.sh
# This script is submitted by eval_orchestrator.sh to run evaluate.py for a single checkpoint.
# Job name, output/error paths are set by eval_orchestrator.sh.
#SBATCH --job-name=single_eval_job # Will be overridden by eval_orchestrator
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU partition for actual evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for evaluate.py
#SBATCH --mem=32G                        # RAM for evaluate.py
#SBATCH --time=7-0:00:00                  # Max time for a single evaluation run (adjust as needed)
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL                 # Notify only on failure of an individual eval job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error for this specific evaluation job
set -e
# set -o pipefail # Consider this if you want pipeline failures to cause an exit

# Function for logging messages from this script
log_eval_job_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - EVAL_JOB ($SLURM_JOB_ID): $1"
}

log_eval_job_message "=== Single Evaluation Job Script Started ==="
log_eval_job_message "Slurm Job Name: $SLURM_JOB_NAME"
log_eval_job_message "Running on Node: $SLURMD_NODENAME"
log_eval_job_message "Allocated CPUs: $SLURM_CPUS_PER_TASK"
log_eval_job_message "Allocated Memory: $SLURM_MEM_PER_GPU or $SLURM_MEM_PER_NODE"
log_eval_job_message "Allocated GPUs: $CUDA_VISIBLE_DEVICES (Slurm raw: ${SLURM_JOB_GPUS:-N/A})"
log_eval_job_message "Working Directory: $(pwd)"

# --- Echo variables received from eval_orchestrator.sh via --export ---
log_eval_job_message "--- Received Environment Variables ---"
log_eval_job_message "Host Project Dir: ${HOST_PROJECT_DIR}"
log_eval_job_message "Shared Output Dir (Host, for checkpoint source): ${SHARED_OUTPUT_DIR_HOST}"
log_eval_job_message "Shared Run ID (for context/tagging): ${SHARED_RUN_ID}"
log_eval_job_message "Seed for this Evaluation: ${SEED_FOR_THIS_EVAL}"
log_eval_job_message "Checkpoint Directory Name to Evaluate: ${CHECKPOINT_DIR_NAME_TO_EVAL}"
log_eval_job_message "Host Path for THIS Eval Job's Results (where evaluate.py outputs go): ${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}"
log_eval_job_message "------------------------------------"

# --- Validate received variables ---
log_eval_job_message "Validating received environment variables..."
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_HOST" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_THIS_EVAL" ] || \
   [ -z "$CHECKPOINT_DIR_NAME_TO_EVAL" ] || \
   [ -z "$HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS" ]; then
    log_eval_job_message "CRITICAL ERROR: One or more essential environment variables not set by eval_orchestrator! Dumping environment:"
    env | grep -E 'HOST_PROJECT_DIR|SHARED_OUTPUT_DIR_HOST|SHARED_RUN_ID|SEED_FOR_THIS_EVAL|CHECKPOINT_DIR_NAME_TO_EVAL|HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS'
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
    log_eval_job_message "SINGULARITYENV_NEPTUNE_PROJECT set to: ${SINGULARITYENV_NEPTUNE_PROJECT}"
else
    log_eval_job_message "WARNING: Neptune credentials file '$NEPTUNE_CRED_FILE_EVAL_JOB' not found."
    export SINGULARITYENV_NEPTUNE_PROJECT="${SINGULARITYENV_NEPTUNE_PROJECT:-thmorton/NothingProject}" # <<< UPDATE default
    log_eval_job_message "SINGULARITYENV_NEPTUNE_PROJECT (fallback) set to: ${SINGULARITYENV_NEPTUNE_PROJECT}"
fi
if [ -z "$SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME" ]; then
    export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}"
    log_eval_job_message "SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME constructed as: ${SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME}"
else
    log_eval_job_message "SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME received as: ${SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME}"
fi

# --- Define Paths on Host ---
log_eval_job_message "Defining host paths..."
HOST_SIF_PATH_EVAL_JOB="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE SIF name
HOST_DATA_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/data"
HOST_PRIMING_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/eval"
log_eval_job_message "SIF Path: ${HOST_SIF_PATH_EVAL_JOB}"
log_eval_job_message "Data Base Dir: ${HOST_DATA_BASE_DIR_EVAL_JOB}"
log_eval_job_message "Priming Base Dir: ${HOST_PRIMING_BASE_DIR_EVAL_JOB}"

# --- Define Container Mount Points ---
log_eval_job_message "Defining container mount points..."
CONTAINER_WORKSPACE_EVAL_JOB="/workspace"
CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB="/data_eval_mnt"
CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB="/priming_data_eval_mnt"
CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT="/mnt_all_checkpoints"
CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET="/eval_job_output_target"

# --- Preparations ---
log_eval_job_message "Performing preparations..."
if [ ! -f "$HOST_SIF_PATH_EVAL_JOB" ]; then
    log_eval_job_message "CRITICAL ERROR: Singularity image not found at ${HOST_SIF_PATH_EVAL_JOB}"
    exit 1
fi
log_eval_job_message "Singularity image confirmed at ${HOST_SIF_PATH_EVAL_JOB}."
mkdir -p "$(dirname "$HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS")"
log_eval_job_message "Ensured parent directory for evaluation job results exists: $(dirname "$HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS")"
log_eval_job_message "evaluate.py will write its specific results into (host path): ${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}"

log_eval_job_message "Starting evaluate.py for checkpoint: '${CHECKPOINT_DIR_NAME_TO_EVAL}'..."

EVAL_PY_SCRIPT_CONTAINER_PATH="${CONTAINER_WORKSPACE_EVAL_JOB}/src/evaluate.py"
CHECKPOINT_TO_EVAL_CONTAINER_PATH="${CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT}/${CHECKPOINT_DIR_NAME_TO_EVAL}"
EVAL_PY_VALID_DATA_CONTAINER_PATH="${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}/processed/test_set_10m" # <<< ADJUST
EVAL_PY_PRIMING_DATA_CONTAINER_PATH="${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}/priming-corpuses" # <<< ADJUST

# --- Construct arguments for evaluate.py ---
log_eval_job_message "Constructing arguments for evaluate.py..."
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# CORRECTED: Removed 'local' from array declaration
declare -a PYTHON_ARGS_FOR_EVALUATE_PY
# Or simply: PYTHON_ARGS_FOR_EVALUATE_PY=()
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_path" "${CHECKPOINT_TO_EVAL_CONTAINER_PATH}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--output_dir" "${CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_label" "${CHECKPOINT_DIR_NAME_TO_EVAL}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--seed" "${SEED_FOR_THIS_EVAL}" )

PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_standard_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--validation_dataset_path" "$EVAL_PY_VALID_DATA_CONTAINER_PATH" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_priming_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_eval_dir_path" "$EVAL_PY_PRIMING_DATA_CONTAINER_PATH" )

PYTHON_ARGS_FOR_EVALUATE_PY+=( "--per_device_eval_batch_size" "16" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_per_device_eval_batch_size" "8" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--use_amp" )

NEPTUNE_EVAL_JOB_RUN_NAME="eval_job_${SHARED_RUN_ID}_${CHECKPOINT_DIR_NAME_TO_EVAL}"
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_run_name" "${NEPTUNE_EVAL_JOB_RUN_NAME}" )

declare -a NEPTUNE_EVAL_JOB_TAGS_ARRAY # Explicitly declare as array
NEPTUNE_EVAL_JOB_TAGS_ARRAY=("evaluation_job_submission" "${SHARED_RUN_ID}" "${CHECKPOINT_DIR_NAME_TO_EVAL}" "seed_${SEED_FOR_THIS_EVAL}")
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_tags" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "${NEPTUNE_EVAL_JOB_TAGS_ARRAY[@]}" )

if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
fi

log_eval_job_message "--- Arguments for evaluate.py ---"
printf "  %q\n" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}"
log_eval_job_message "---------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVALUATE_PY_CONSOLE_LOG="${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}/evaluate_py_console_output_${CHECKPOINT_DIR_NAME_TO_EVAL}.log"
log_eval_job_message "Console output of evaluate.py (stdout/stderr) will be redirected to: ${EVALUATE_PY_CONSOLE_LOG}"
mkdir -p "${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}"


SINGULARITY_COMMAND_TO_RUN_STR="singularity exec --nv \\
    -B \"${HOST_PROJECT_DIR}\":\"${CONTAINER_WORKSPACE_EVAL_JOB}\" \\
    -B \"${HOST_DATA_BASE_DIR_EVAL_JOB}\":\"${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}\" \\
    -B \"${HOST_PRIMING_BASE_DIR_EVAL_JOB}\":\"${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}\" \\
    -B \"${SHARED_OUTPUT_DIR_HOST}\":\"${CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT}\" \\
    -B \"${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}\":\"${CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET}\" \\
    \"${HOST_SIF_PATH_EVAL_JOB}\" \\
    python3 \"${EVAL_PY_SCRIPT_CONTAINER_PATH}\""

for arg_idx in "${!PYTHON_ARGS_FOR_EVALUATE_PY[@]}"; do
    arg_val="${PYTHON_ARGS_FOR_EVALUATE_PY[$arg_idx]}"
    # Correctly quote arguments for the string command representation
    if [[ "$arg_val" == *" "* ]]; then
        SINGULARITY_COMMAND_TO_RUN_STR+=" \"${arg_val//\"/\\\"}\"" # Escape internal quotes for the string
    else
        SINGULARITY_COMMAND_TO_RUN_STR+=" ${arg_val}"
    fi
done

log_eval_job_message "--- Full Singularity Command to be Executed (for debugging) ---"
echo "${SINGULARITY_COMMAND_TO_RUN_STR}"
log_eval_job_message "---------------------------------------------"

log_eval_job_message "Executing Singularity command..."
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE_EVAL_JOB}" \
    -B "${HOST_DATA_BASE_DIR_EVAL_JOB}":"${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${HOST_PRIMING_BASE_DIR_EVAL_JOB}":"${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT}" \
    -B "${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}":"${CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET}" \
    "${HOST_SIF_PATH_EVAL_JOB}" \
    python3 "${EVAL_PY_SCRIPT_CONTAINER_PATH}" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}" > "${EVALUATE_PY_CONSOLE_LOG}" 2>&1

EVALUATE_PY_EXIT_CODE=$?

if [ $EVALUATE_PY_EXIT_CODE -ne 0 ]; then
    log_eval_job_message "CRITICAL ERROR: evaluate.py FAILED with exit code ${EVALUATE_PY_EXIT_CODE} for checkpoint '${CHECKPOINT_DIR_NAME_TO_EVAL}'."
    log_eval_job_message "Check evaluate.py console output at: ${EVALUATE_PY_CONSOLE_LOG}"
    log_eval_job_message "Also check this Slurm job's error file."
    exit 1
fi

log_eval_job_message "evaluate.py completed successfully for checkpoint '${CHECKPOINT_DIR_NAME_TO_EVAL}'."
log_eval_job_message "evaluate.py console output logged to: ${EVALUATE_PY_CONSOLE_LOG}"
log_eval_job_message "Results from evaluate.py (JSON, CSVs) should be in (host path): ${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}"
log_eval_job_message "=== Single Evaluation Job Script Finished Successfully ==="