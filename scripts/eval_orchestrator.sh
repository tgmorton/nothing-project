#!/bin/bash
# eval_orchestrator.sh

# === SBATCH Directives for Evaluation Orchestrator ===
#SBATCH --job-name=eval_orchestrator # Will be overridden by main_orchestrator
#SBATCH --partition=general          # <<< CPU Partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # Minimal CPUs for polling
#SBATCH --mem=4G                     # Minimal RAM
#SBATCH --time=7-0:00:00              # Should be long enough to monitor training output
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# set -e # Removed for the main loop to allow continued polling even if one sbatch submission fails

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
PATH_TO_EVAL_JOB_SBATCH="${HOST_PROJECT_DIR}/scripts/eval_job.sh" # Path to the GPU eval job script
EVAL_JOB_CHILD_LOGS_DIR="${SHARED_OUTPUT_DIR_HOST}/eval_job_slurm_logs"

if [ ! -f "$PATH_TO_EVAL_JOB_SBATCH" ]; then
    echo "CRITICAL ERROR: eval_job.sh script not found at ${PATH_TO_EVAL_JOB_SBATCH}"
    exit 1
fi
mkdir -p "$EVAL_JOB_CHILD_LOGS_DIR"
echo "Logs for individual sbatched evaluation jobs will be in: ${EVAL_JOB_CHILD_LOGS_DIR}"

# --- Helper function for logging ---
log_eval_orch_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') - EVAL_ORCH ($SLURM_JOB_ID): $1"
}

# --- Function to submit an individual evaluation job ---
# Returns 0 on successful sbatch submission, 1 on failure.
submit_evaluation_job_for_checkpoint() {
    local checkpoint_dir_name_to_eval=$1
    
    log_eval_orch_message "Attempting to submit evaluation job for checkpoint: '${checkpoint_dir_name_to_eval}'"
    local host_path_for_this_eval_job_results="${SHARED_OUTPUT_DIR_HOST}/eval_results/${checkpoint_dir_name_to_eval}"
    local eval_slurm_job_name="eval_${SHARED_RUN_ID}_${checkpoint_dir_name_to_eval}"

    # Ensure the directory for this specific eval job's results exists on host before sbatching
    # This is where evaluate.py will write its JSON/CSVs.
    mkdir -p "$host_path_for_this_eval_job_results"
    if [ $? -ne 0 ]; then
        log_eval_orch_message "ERROR: Failed to create results directory on host: ${host_path_for_this_eval_job_results} for checkpoint '${checkpoint_dir_name_to_eval}'."
        return 1 # Failure
    fi


    SUBMIT_EVAL_JOB_CMD_OUTPUT=$(sbatch \
        --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_THIS_EVAL="${SEED_FOR_EVAL}",CHECKPOINT_DIR_NAME_TO_EVAL="${checkpoint_dir_name_to_eval}",HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS="${host_path_for_this_eval_job_results}" \
        --job-name="${eval_slurm_job_name}" \
        --output="${EVAL_JOB_CHILD_LOGS_DIR}/${eval_slurm_job_name}_%j.out" \
        --error="${EVAL_JOB_CHILD_LOGS_DIR}/${eval_slurm_job_name}_%j.err" \
        "${PATH_TO_EVAL_JOB_SBATCH}")
    
    local sbatch_exit_code=$?
    if [ $sbatch_exit_code -ne 0 ]; then
        log_eval_orch_message "ERROR: sbatch command FAILED (exit code ${sbatch_exit_code}) for submitting eval job for '${checkpoint_dir_name_to_eval}'. sbatch output: ${SUBMIT_EVAL_JOB_CMD_OUTPUT}"
        return 1
    fi

    local submitted_eval_job_id=$(echo "$SUBMIT_EVAL_JOB_CMD_OUTPUT" | awk '{print $NF}')
    if ! [[ "$submitted_eval_job_id" =~ ^[0-9]+$ ]]; then
        log_eval_orch_message "ERROR: Failed to parse Job ID from sbatch output for '${checkpoint_dir_name_to_eval}'. Output: ${SUBMIT_EVAL_JOB_CMD_OUTPUT}"
        return 1
    fi

    log_eval_orch_message "Successfully submitted evaluation job for '${checkpoint_dir_name_to_eval}'. Slurm Job ID: ${submitted_eval_job_id}."
    return 0
}

# --- Main Polling Loop ---
log_eval_orch_message "Starting polling of ${SHARED_OUTPUT_DIR_HOST} for new checkpoints..."
declare -A PROCESSED_CHECKPOINTS_LOG # Associative array to keep track of submitted evaluation jobs

TRAINING_COMPLETED_SENTINEL="${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"
POLL_INTERVAL_SECONDS=120
ORCHESTRATOR_LOOP_TIMEOUT_SECONDS=$((24 * 60 * 60 + 30 * 60)) # 24h 30m
SECONDS=0

while (( SECONDS < ORCHESTRATOR_LOOP_TIMEOUT_SECONDS )); do
    training_finished_flag=false
    if [ -f "${TRAINING_COMPLETED_SENTINEL}" ]; then
        log_eval_orch_message "'${TRAINING_COMPLETED_SENTINEL}' detected. Training assumed complete."
        training_finished_flag=true
    fi

    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # MODIFIED SECTION TO AVOID SUBSHELL FOR PROCESSED_CHECKPOINTS_LOG
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    log_eval_orch_message "Scanning for checkpoint directories in ${SHARED_OUTPUT_DIR_HOST}..."
    
    # Use an array to store checkpoint names found in this scan
    declare -a current_checkpoints_found_in_scan
    # Populate the array using find and mapfile (bash 4+)
    # -maxdepth 1: only look in the immediate directory
    # -mindepth 1: don't include the SHARED_OUTPUT_DIR_HOST itself
    # -type d: only directories
    # -printf "%f\n": print only the basename of the directory
    mapfile -t current_checkpoints_found_in_scan < <(find "${SHARED_OUTPUT_DIR_HOST}" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" 2>/dev/null)
    
    # If find fails (e.g. SHARED_OUTPUT_DIR_HOST doesn't exist yet), current_checkpoints_found_in_scan might be empty.
    # The loop below will simply not iterate.

    any_new_checkpoint_job_submitted_this_iteration=false

    for current_checkpoint_name in "${current_checkpoints_found_in_scan[@]}"; do
        # Check if it's a valid checkpoint name pattern
        if [[ "$current_checkpoint_name" == checkpoint-* || "$current_checkpoint_name" == "final_model" ]]; then
            # Check if this checkpoint has already had a job submitted for it
            if [[ -z "${PROCESSED_CHECKPOINTS_LOG[$current_checkpoint_name]}" ]]; then
                log_eval_orch_message "New unprocessed checkpoint directory found: '${current_checkpoint_name}'."
                submit_evaluation_job_for_checkpoint "$current_checkpoint_name"
                submit_attempt_status=$? # Get the return status of the submission function

                if [ $submit_attempt_status -eq 0 ]; then
                    # Mark as processed ONLY if submission was successful
                    PROCESSED_CHECKPOINTS_LOG["$current_checkpoint_name"]=1 
                    log_eval_orch_message "Marked '${current_checkpoint_name}' as processed (submission successful)."
                else
                    log_eval_orch_message "WARNING: Submission FAILED for '${current_checkpoint_name}'. It will be re-attempted in the next scan."
                    # Do NOT mark as processed, so it gets picked up again.
                fi
                any_new_checkpoint_job_submitted_this_iteration=true # A submission was attempted
            fi
        fi
    done
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # END OF MODIFIED SECTION
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Logic for exiting the main loop
    if [[ "$training_finished_flag" == true ]]; then
        # If training is done, we need to check if all existing relevant checkpoints have been processed.
        # The 'any_new_checkpoint_job_submitted_this_iteration' helps decide if we need more scans.
        # A more robust check would be to re-scan and see if any valid, unprocessed checkpoints *still* exist.
        
        # Perform a specific check for 'final_model' if training is finished and it hasn't been processed
        final_model_dir_on_host="${SHARED_OUTPUT_DIR_HOST}/final_model"
        if [ -d "$final_model_dir_on_host" ] && [[ -z "${PROCESSED_CHECKPOINTS_LOG['final_model']}" ]]; then
            log_eval_orch_message "Training finished, and 'final_model' directory exists but is not yet marked as processed. Submitting job for it."
            submit_evaluation_job_for_checkpoint "final_model"
            submit_final_model_status=$?
            if [ $submit_final_model_status -eq 0 ]; then
                PROCESSED_CHECKPOINTS_LOG["final_model"]=1
                log_eval_orch_message "Marked 'final_model' as processed."
            else
                log_eval_orch_message "WARNING: Submission FAILED for 'final_model'. It may be re-attempted if loop continues."
            fi
            # Even if submission failed, we might want to loop once more if training is done.
            # The any_new_checkpoint_job_submitted_this_iteration will ensure at least one more poll if submission happened.
            any_new_checkpoint_job_submitted_this_iteration=true
        fi

        # If training is finished AND no new checkpoint jobs were submitted in this iteration
        # (meaning all found checkpoints were already processed, or no new ones appeared)
        # AND final_model (if it exists) is processed, then we can consider exiting.
        if [[ "$any_new_checkpoint_job_submitted_this_iteration" == false ]]; then
            # If final_model exists, ensure it's processed. If it doesn't exist, we don't need to wait for it.
            if [ -d "$final_model_dir_on_host" ] && [[ -n "${PROCESSED_CHECKPOINTS_LOG['final_model']}" ]]; then
                log_eval_orch_message "Training finished, no new checkpoints found, and 'final_model' processed. Shutting down evaluation orchestrator."
                break
            elif [ ! -d "$final_model_dir_on_host" ]; then # final_model directory doesn't even exist
                log_eval_orch_message "Training finished, no new checkpoints found, and 'final_model' directory does not exist. Shutting down evaluation orchestrator."
                break
            else
                # final_model exists but wasn't processed, or some other condition. Keep polling a bit longer.
                log_eval_orch_message "Training finished. Waiting for 'final_model' processing or final quiet period."
            fi
        fi
    fi
    
    if [[ "$training_finished_flag" == true ]]; then
        log_eval_orch_message "Training finished. Polling for any final checkpoint processing. Interval: ${POLL_INTERVAL_SECONDS}s."
    else
        log_eval_orch_message "Polling for new checkpoints. Next check in ${POLL_INTERVAL_SECONDS}s."
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done

# After the loop
if (( SECONDS >= ORCHESTRATOR_LOOP_TIMEOUT_SECONDS )); then
    log_eval_orch_message "WARNING: Evaluation Orchestrator main loop timed out after ${ORCHESTRATOR_LOOP_TIMEOUT_SECONDS} seconds."
    if [[ "$training_finished_flag" == false ]]; then
        log_eval_orch_message "CRITICAL WARNING: Loop timed out AND '${TRAINING_COMPLETED_SENTINEL}' was NOT found."
    fi
fi

log_eval_orch_message "Evaluation Orchestrator completed its monitoring and submission tasks."
echo "=== Evaluation Orchestrator Finished: $(date) ==="