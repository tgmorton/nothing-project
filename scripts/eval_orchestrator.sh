#!/bin/bash
# eval_orchestrator.sh
# This script now submits a single, long-running evaluation job (eval_job.sh)
# that handles multiple checkpoints and watch mode.

# === SBATCH Directives for This Orchestrator Script (very lightweight) ===
#SBATCH --job-name=eval_orch_submitter # Will be overridden by main_orchestrator
#SBATCH --partition=general          # CPU Partition for this submitter script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # Minimal CPUs
#SBATCH --mem=1G                     # Minimal RAM
#SBATCH --time=0-00:10:00            # Short time, just to submit the actual eval job
#SBATCH --mail-type=FAIL             # Notify only if this submission script fails
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

set -e # Exit on error for this submission script

echo "=== Evaluation Orchestrator (Submitter) Started: $(date) ==="
echo "Eval Orchestrator (Submitter) Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Host Project Dir: ${HOST_PROJECT_DIR}"
echo "Shared Output Directory on Host (for eval_job to watch): ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID (for context): ${SHARED_RUN_ID}"
echo "Seed for Evaluation Context: ${SEED_FOR_EVAL}"
echo "Checkpoint Ready Sentinel Filename: ${CHECKPOINT_READY_SENTINEL_FILENAME}"


# --- Validate essential variables ---
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_HOST" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_EVAL" ] || \
   [ -z "$CHECKPOINT_READY_SENTINEL_FILENAME" ]; then
    echo "CRITICAL ERROR (Eval Orchestrator Submitter): Essential env vars not set!"
    env | grep -E 'HOST_PROJECT_DIR|SHARED_OUTPUT_DIR_HOST|SHARED_RUN_ID|SEED_FOR_EVAL|CHECKPOINT_READY_SENTINEL_FILENAME'
    exit 1
fi

# --- Define Paths ---
PATH_TO_ACTUAL_EVAL_JOB_SBATCH="${HOST_PROJECT_DIR}/scripts/eval_job.sh" # Path to the long-running eval job
EVAL_JOB_SLURM_LOGS_SUBDIR_NAME="long_running_eval_slurm_logs" # Subdirectory within SHARED_OUTPUT_DIR_HOST
LONG_EVAL_JOB_SLURM_LOGS_DIR="${SHARED_OUTPUT_DIR_HOST}/${EVAL_JOB_SLURM_LOGS_SUBDIR_NAME}"

if [ ! -f "$PATH_TO_ACTUAL_EVAL_JOB_SBATCH" ]; then
    echo "CRITICAL ERROR: The target evaluation script (eval_job.sh) not found at ${PATH_TO_ACTUAL_EVAL_JOB_SBATCH}"
    exit 1
fi
mkdir -p "$LONG_EVAL_JOB_SLURM_LOGS_DIR"
echo "Logs for the long-running evaluation job will be in: ${LONG_EVAL_JOB_SLURM_LOGS_DIR}"

# --- Submit the Single Long-Running Evaluation Job ---
# This job will run evaluate.py with --watch_mode
log_eval_orch_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - EVAL_ORCH_SUBMITTER ($SLURM_JOB_ID): $1"
}

log_eval_orch_message "Attempting to submit the long-running multi-checkpoint evaluation job (eval_job.sh)..."

# The output directory for evaluate.py itself (where it creates subdirs per checkpoint)
# will be within SHARED_OUTPUT_DIR_HOST, e.g., SHARED_OUTPUT_DIR_HOST/eval_results_multi
EVALUATE_PY_BASE_OUTPUT_DIR_NAME="eval_results_multi"
HOST_PATH_FOR_EVALUATE_PY_BASE_RESULTS="${SHARED_OUTPUT_DIR_HOST}/${EVALUATE_PY_BASE_OUTPUT_DIR_NAME}"
mkdir -p "$HOST_PATH_FOR_EVALUATE_PY_BASE_RESULTS"


# Variables to export to eval_job.sh:
# HOST_PROJECT_DIR, SHARED_OUTPUT_DIR_HOST (as the watch dir), SHARED_RUN_ID, SEED_FOR_EVAL,
# CHECKPOINT_READY_SENTINEL_FILENAME, HOST_PATH_FOR_EVALUATE_PY_BASE_RESULTS.
# Other Slurm settings for eval_job.sh are within that script's #SBATCH directives.

LONG_EVAL_SLURM_JOB_NAME="multi_eval_watch_${SHARED_RUN_ID}"

SUBMIT_CMD_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_TO_WATCH="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_THIS_MULTI_EVAL="${SEED_FOR_EVAL}",CHECKPOINT_READY_SENTINEL_FILENAME="${CHECKPOINT_READY_SENTINEL_FILENAME}",EVALUATE_PY_OVERALL_OUTPUT_DIR="${HOST_PATH_FOR_EVALUATE_PY_BASE_RESULTS}" \
    --job-name="${LONG_EVAL_SLURM_JOB_NAME}" \
    --output="${LONG_EVAL_JOB_SLURM_LOGS_DIR}/${LONG_EVAL_SLURM_JOB_NAME}_%j.out" \
    --error="${LONG_EVAL_JOB_SLURM_LOGS_DIR}/${LONG_EVAL_SLURM_JOB_NAME}_%j.err" \
    "${PATH_TO_ACTUAL_EVAL_JOB_SBATCH}")

sbatch_exit_code=$?
if [ $sbatch_exit_code -ne 0 ]; then
    log_eval_orch_message "ERROR: sbatch command FAILED (exit code ${sbatch_exit_code}) for submitting the long-running eval job. sbatch output: ${SUBMIT_CMD_OUTPUT}"
    exit 1 # This submitter script failed
fi

submitted_job_id=$(echo "$SUBMIT_CMD_OUTPUT" | awk '{print $NF}')
if ! [[ "$submitted_job_id" =~ ^[0-9]+$ ]]; then
    log_eval_orch_message "ERROR: Failed to parse Job ID from sbatch output for long-running eval job. Output: ${SUBMIT_CMD_OUTPUT}"
    exit 1 # This submitter script failed
fi

log_eval_orch_message "Successfully submitted long-running multi-checkpoint evaluation job. Slurm Job ID: ${submitted_job_id}."
log_eval_orch_message "This job will now monitor ${SHARED_OUTPUT_DIR_HOST} for checkpoints."

echo "=== Evaluation Orchestrator (Submitter) Finished Successfully: $(date) ==="
# This script's work is done after submitting the main eval job.
# main_orchestrator.sh will wait for this script (eval_orchestrator.sh) to complete.
# The actual evaluation results depend on the long-running job.