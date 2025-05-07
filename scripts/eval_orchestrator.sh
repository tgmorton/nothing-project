#!/bin/bash
# eval_orchestrator.sh

# === SBATCH Directives for Evaluation Orchestrator ===
# This job runs on a CPU node and submits individual GPU evaluation jobs.
#SBATCH --job-name=eval_orchestrator # Will be overridden by main_orchestrator
#SBATCH --partition=general          # <<< CPU Partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # Minimal CPUs for polling
#SBATCH --mem=4G                     # Minimal RAM
#SBATCH --time=24:40:00              # Should be long enough to monitor the entire training output
                                     # Slightly less than main_orchestrator's time.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# set -e # Be cautious with set -e in a long-running polling loop.

echo "=== Evaluation Orchestrator Started: $(date) ==="
echo "Evaluation Orchestrator Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Host Project Dir (from Main Orchestrator): ${HOST_PROJECT_DIR}"
echo "Shared Output Directory on Host (to monitor): ${SHARED_OUTPUT_DIR_HOST}"
echo "Shared Run ID (for context): ${SHARED_RUN_ID}"
echo "Seed for this Evaluation Context: ${SEED_FOR_EVAL}"

# --- Validate essential variables ---
if [ -z "$HOST_PROJECT_DIR" ] || [ -z "$SHARED_OUTPUT_DIR_HOST" ] || [ -z "$SHARED_RUN_ID" ]; then
    echo "CRITICAL ERROR (Eval Orchestrator): Essential env vars not set by main_orchestrator!"
    exit 1
fi
if [ -z "$SEED_FOR_EVAL" ]; then
    echo "WARNING (Eval Orchestrator): SEED_FOR_EVAL not set. Using default 'unknown_seed'."
    SEED_FOR_EVAL="unknown_seed"
fi

# --- Define Paths ---
PATH_TO_EVAL_JOB_SBATCH="${HOST_PROJECT_DIR}/scripts/eval_job.sh" # Path to the new GPU eval job script
EVAL_JOB_CHILD_LOGS_DIR="${SHARED_OUTPUT_DIR_HOST}/eval_job_slurm_logs" # Logs for sbatched eval_job.sh instances

if [ ! -f "$PATH_TO_EVAL_JOB_SBATCH" ]; then
    echo "CRITICAL ERROR: eval_job.sh script not found at ${PATH_TO_EVAL_JOB_SBATCH}"
    exit 1
fi
mkdir -p "$EVAL_JOB_CHILD_LOGS_DIR"
echo "Logs for individual sbatched evaluation jobs will be in: ${EVAL_JOB_CHILD_LOGS_DIR}"

# --- Helper function for logging from this orchestrator ---
log_eval_orch_message() {
    echo "$(date): EVAL_ORCH: $1"
}

# --- Function to submit an individual evaluation job ---
submit_evaluation_job_for_checkpoint() {
    local checkpoint_dir_name_to_eval=$1 # e.g., "checkpoint-1000" or "final_model"

    log_eval_orch_message "Attempting to submit evaluation job for checkpoint: '${checkpoint_dir_name_to_eval}'"

    # Define a specific output directory on the host for this eval job's results (JSON, raw CSVs)
    # This path will be passed to eval_job.sh, which then passes it to evaluate.py
    local host_path_for_this_eval_job_results="${SHARED_OUTPUT_DIR_HOST}/eval_results/${checkpoint_dir_name_to_eval}"
    # No need to mkdir -p here; eval_job.sh or evaluate.py can do that.

    # Construct job name for the individual eval job
    local eval_slurm_job_name="eval_${SHARED_RUN_ID}_${checkpoint_dir_name_to_eval}"

    # Submit eval_job.sh
    # Pass all necessary info via --export. eval_job.sh will use these.
    SUBMIT_EVAL_JOB_CMD_OUTPUT=$(sbatch \
        --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_THIS_EVAL="${SEED_FOR_EVAL}",CHECKPOINT_DIR_NAME_TO_EVAL="${checkpoint_dir_name_to_eval}",HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS="${host_path_for_this_eval_job_results}" \
        --job-name="${eval_slurm_job_name}" \
        --output="${EVAL_JOB_CHILD_LOGS_DIR}/${eval_slurm_job_name}_%j.out" \
        --error="${EVAL_JOB_CHILD_LOGS_DIR}/${eval_slurm_job_name}_%j.err" \
        "${PATH_TO_EVAL_JOB_SBATCH}")

    local sbatch_exit_code=$?
    if [ $sbatch_exit_code -ne 0 ]; then
        log_eval_orch_message "ERROR: sbatch command FAILED for submitting eval job for '${checkpoint_dir_name_to_eval}'. sbatch output: ${SUBMIT_EVAL_JOB_CMD_OUTPUT}"
        return 1 # Failure
    fi

    local submitted_eval_job_id=$(echo "$SUBMIT_EVAL_JOB_CMD_OUTPUT" | awk '{print $NF}')
    if ! [[ "$submitted_eval_job_id" =~ ^[0-9]+$ ]]; then
        log_eval_orch_message "ERROR: Failed to parse Job ID from sbatch output for '${checkpoint_dir_name_to_eval}'. Output: ${SUBMIT_EVAL_JOB_CMD_OUTPUT}"
        return 1 # Failure
    fi

    log_eval_orch_message "Successfully submitted evaluation job for '${checkpoint_dir_name_to_eval}'. Slurm Job ID: ${submitted_eval_job_id}."
    return 0 # Success
}


# --- Main Polling Loop ---
log_eval_orch_message "Starting polling of ${SHARED_OUTPUT_DIR_HOST} for new checkpoints..."
declare -A PROCESSED_CHECKPOINTS_LOG # Associative array to keep track of submitted evaluation jobs

TRAINING_COMPLETED_SENTINEL="${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"
POLL_INTERVAL_SECONDS=120 # Check every 2 minutes (adjust as needed)
# Timeout for this orchestrator's main loop (should be less than its Slurm job time limit)
ORCHESTRATOR_LOOP_TIMEOUT_SECONDS=$((24 * 60 * 60 + 30 * 60)) # 24h 30m
SECONDS=0 # Reset bash timer

while (( SECONDS < ORCHESTRATOR_LOOP_TIMEOUT_SECONDS )); do
    training_finished_flag=false
    if [ -f "${TRAINING_COMPLETED_SENTINEL}" ]; then
        log_eval_orch_message "'${TRAINING_COMPLETED_SENTINEL}' detected. Training assumed complete."
        training_finished_flag=true
    fi

    # Scan for checkpoint directories
    # Using find to robustly handle filenames and list only directories
    found_new_checkpoint_in_scan=false
    find "${SHARED_OUTPUT_DIR_HOST}" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d $'\0' checkpoint_dir_path_on_host; do
        current_checkpoint_name=$(basename "$checkpoint_dir_path_on_host")

        # Check if it's a valid checkpoint name pattern and not yet processed
        if [[ "$current_checkpoint_name" == checkpoint-* || "$current_checkpoint_name" == "final_model" ]]; then
            if [[ -z "${PROCESSED_CHECKPOINTS_LOG[$current_checkpoint_name]}" ]]; then
                log_eval_orch_message "New unprocessed checkpoint directory found: '${current_checkpoint_name}'."
                submit_evaluation_job_for_checkpoint "$current_checkpoint_name"
                # On successful submission, mark as processed (or attempt recorded)
                # A more robust system might check squeue or sacct for actual submission success beyond sbatch exit code.
                PROCESSED_CHECKPOINTS_LOG["$current_checkpoint_name"]=1 # Mark as submission attempted
                found_new_checkpoint_in_scan=true
            fi
        fi
    done

    # If training is finished and no new checkpoints were found in the last scan, we can exit.
    if [[ "$training_finished_flag" == true ]] && [[ "$found_new_checkpoint_in_scan" == false ]]; then
        # One final check: ensure 'final_model' was processed if it exists and training is done
        final_model_path_check="${SHARED_OUTPUT_DIR_HOST}/final_model"
        if [ -d "$final_model_path_check" ] && [[ -z "${PROCESSED_CHECKPOINTS_LOG['final_model']}" ]]; then
            log_eval_orch_message "Training finished, and 'final_model' directory exists but wasn't processed. Submitting job for it now."
            submit_evaluation_job_for_checkpoint "final_model"
            PROCESSED_CHECKPOINTS_LOG["final_model"]=1
            # After this, the next loop iteration will see training_finished_flag=true and found_new_checkpoint_in_scan=false (likely)
        else
            log_eval_orch_message "Training finished and no new unprocessed checkpoints found in the latest scan. Shutting down evaluation orchestrator."
            break # Exit the while loop
        fi
    fi

    if [[ "$training_finished_flag" == true ]]; then
        log_eval_orch_message "Training finished. Polling for any last-minute checkpoints or final_model processing. Interval: ${POLL_INTERVAL_SECONDS}s."
    else
        log_eval_orch_message "Polling for new checkpoints. Next check in ${POLL_INTERVAL_SECONDS}s."
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done

# After the loop (either by break or timeout)
if (( SECONDS >= ORCHESTRATOR_LOOP_TIMEOUT_SECONDS )); then
    log_eval_orch_message "WARNING: Evaluation Orchestrator main loop timed out after ${ORCHESTRATOR_LOOP_TIMEOUT_SECONDS} seconds."
    if [[ "$training_finished_flag" == false ]]; then
        log_eval_orch_message "CRITICAL WARNING: Loop timed out AND '${TRAINING_COMPLETED_SENTINEL}' was NOT found. Training might be incomplete or stuck."
    fi
fi

log_eval_orch_message "Evaluation Orchestrator completed its monitoring task."
echo "=== Evaluation Orchestrator Finished: $(date) ==="