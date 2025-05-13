#!/bin/bash
# eval_orchestrator.sh (Submitter for the long-running eval_job.sh)

# --- Source Project Configuration ---
if [ -z "$CONFIG_FILE_PATH_ABS" ]; then
    echo "CRITICAL ERROR: CONFIG_FILE_PATH_ABS not provided to eval_orchestrator.sh"
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

#SBATCH --job-name=${EVAL_ORCH_SUBMITTER_JOB_NAME:-eval_orch_submit}
#SBATCH --partition=${EVAL_ORCH_SUBMITTER_PARTITION:-general}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${EVAL_ORCH_SUBMITTER_CPUS:-1}
#SBATCH --mem=${EVAL_ORCH_SUBMITTER_MEM:-1G}
#SBATCH --time=${EVAL_ORCH_SUBMITTER_TIME:-0-00:10:00} # Short, just to submit
#SBATCH --mail-type=${EVAL_ORCH_SUBMITTER_MAIL_TYPE:-FAIL}
#SBATCH --mail-user=${EVAL_ORCH_SUBMITTER_MAIL_USER:-your_email@example.com}

set -e
echo "=== Evaluation Orchestrator (Submitter) Started: $(date) ==="
echo "Eval Orchestrator (Submitter) Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Host Project Dir (from env): ${HOST_PROJECT_DIR}"
echo "Shared Output Dir for eval_job to watch (from env): ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID (from env): ${SHARED_RUN_ID}"
echo "Seed for Eval Context (from env): ${SEED_FOR_EVAL}"
echo "Checkpoint Ready Sentinel Filename (from env): ${CHECKPOINT_READY_SENTINEL_FILENAME}"

# Validate essential env vars
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_HOST" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_EVAL" ] || \
   [ -z "$CHECKPOINT_READY_SENTINEL_FILENAME" ]; then
    echo "CRITICAL ERROR (Eval Orchestrator Submitter): Essential env vars from main_orchestrator missing!"
    exit 1
fi

PATH_TO_ACTUAL_EVAL_JOB_SCRIPT="${HOST_PROJECT_DIR}/${EVAL_JOB_SCRIPT_RELATIVE_PATH:-scripts/eval_job.sh}"
LONG_EVAL_JOB_SLURM_LOGS_DIR="${SHARED_OUTPUT_DIR_HOST}/${EVAL_JOB_SLURM_LOGS_SUBDIR_NAME:-long_running_eval_slurm_logs}"

if [ ! -f "$PATH_TO_ACTUAL_EVAL_JOB_SCRIPT" ]; then
    echo "CRITICAL ERROR: Target eval_job.sh script not found at ${PATH_TO_ACTUAL_EVAL_JOB_SCRIPT}"; exit 1;
fi
mkdir -p "$LONG_EVAL_JOB_SLURM_LOGS_DIR"
echo "Logs for the long-running evaluation job will be in: ${LONG_EVAL_JOB_SLURM_LOGS_DIR}"

log_eval_orch_submitter_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - EVAL_ORCH_SUBMITTER ($SLURM_JOB_ID): $1"
}

log_eval_orch_submitter_message "Attempting to submit the long-running multi-checkpoint evaluation job (eval_job.sh)..."

EVALUATE_PY_BASE_OUTPUT_DIR_ON_HOST="${SHARED_OUTPUT_DIR_HOST}/${EVALUATE_PY_MULTI_OUTPUT_SUBDIR_NAME:-eval_results_multi}"
mkdir -p "$EVALUATE_PY_BASE_OUTPUT_DIR_ON_HOST"

# Slurm job name for the long-running eval job
LONG_EVAL_SLURM_JOB_NAME="${EVAL_JOB_NAME_PREFIX:-multi_eval_watch}_${SHARED_RUN_ID}"

SUBMIT_CMD_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_TO_WATCH="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_THIS_MULTI_EVAL="${SEED_FOR_EVAL}",CHECKPOINT_READY_SENTINEL_FILENAME="${CHECKPOINT_READY_SENTINEL_FILENAME}",EVALUATE_PY_OVERALL_OUTPUT_DIR_HOST="${EVALUATE_PY_BASE_OUTPUT_DIR_ON_HOST}",CONFIG_FILE_PATH_ABS="$(readlink -f $CONFIG_FILE_PATH_ABS)" \
    --job-name="${LONG_EVAL_SLURM_JOB_NAME}" \
    --output="${LONG_EVAL_JOB_SLURM_LOGS_DIR}/${LONG_EVAL_SLURM_JOB_NAME}_%j.out" \
    --error="${LONG_EVAL_JOB_SLURM_LOGS_DIR}/${LONG_EVAL_SLURM_JOB_NAME}_%j.err" \
    "${PATH_TO_ACTUAL_EVAL_JOB_SCRIPT}")

sbatch_exit_code=$?
if [ $sbatch_exit_code -ne 0 ]; then
    log_eval_orch_submitter_message "ERROR: sbatch FAILED (code ${sbatch_exit_code}) for long-running eval job. Output: ${SUBMIT_CMD_OUTPUT}"; exit 1;
fi

submitted_job_id=$(echo "$SUBMIT_CMD_OUTPUT" | awk '{print $NF}')
if ! [[ "$submitted_job_id" =~ ^[0-9]+$ ]]; then
    log_eval_orch_submitter_message "ERROR: Failed to parse Job ID from sbatch output. Output: ${SUBMIT_CMD_OUTPUT}"; exit 1;
fi

log_eval_orch_submitter_message "Successfully submitted long-running multi-checkpoint evaluation job. Slurm Job ID: ${submitted_job_id}."
log_eval_orch_submitter_message "This job will monitor ${SHARED_OUTPUT_DIR_HOST}."
echo "=== Evaluation Orchestrator (Submitter) Finished Successfully: $(date) ==="