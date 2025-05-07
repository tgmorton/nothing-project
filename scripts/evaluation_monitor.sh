#!/bin/bash
# evaluation_monitor.sbatch

# === SBATCH Directives for Evaluation Monitor ===
# Job name, output/error paths will be overridden by main_orchestrator.
#SBATCH --job-name=eval_monitor_job
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU for evaluation (e.g., p6000, or general_gpu_a5000 if preferred)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for the monitor script and for evaluate.py
#SBATCH --mem=32G                        # RAM for evaluate.py (adjust as needed)
#SBATCH --time=24:30:00                  # Should be slightly longer than the max expected training time
                                         # to ensure it can monitor and evaluate everything.
#SBATCH --mail-type=END,FAIL             # Notify on end/fail of this monitor job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# set -e # Be cautious with 'set -e' inside the main monitoring loop,
         # as an error in one evaluation shouldn't necessarily kill the whole monitor
         # if other checkpoints still need processing. Error handling is done within the loop.

echo "=== Evaluation Monitor Job Started: $(date) ==="
echo "Evaluation Monitor Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Current Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Host Project Dir (from Orchestrator): ${HOST_PROJECT_DIR}"
echo "Shared Output Directory on Host (from Orchestrator): ${SHARED_OUTPUT_DIR_HOST}" # Corrected variable name
echo "Shared Run ID (from Orchestrator): ${SHARED_RUN_ID}"
echo "Seed for this Evaluation Context (from Orchestrator): ${SEED_FOR_EVAL}"

# --- Validate essential variables received from orchestrator ---
if [ -z "$HOST_PROJECT_DIR" ] || [ -z "$SHARED_OUTPUT_DIR_HOST" ] || [ -z "$SHARED_RUN_ID" ]; then
    echo "CRITICAL ERROR (Evaluation Monitor): Essential environment variables (HOST_PROJECT_DIR, SHARED_OUTPUT_DIR_HOST, SHARED_RUN_ID) were not set by the orchestrator!"
    exit 1
fi
# SEED_FOR_EVAL is useful but might have a default if not strictly required for all eval logic
if [ -z "$SEED_FOR_EVAL" ]; then
    echo "WARNING (Evaluation Monitor): SEED_FOR_EVAL not set by orchestrator. Using default 'unknown_seed' for tagging if needed."
    SEED_FOR_EVAL="unknown_seed" # Provide a default if it's used in tagging/naming
fi

# --- Check for inotifywait utility ---
if ! command -v inotifywait &> /dev/null; then
    echo "CRITICAL ERROR: inotifywait command could not be found. Please install 'inotify-tools' package on the compute node, or implement a Python-based file watcher."
    exit 1
fi

# --- Load necessary system modules ---
echo "Loading system modules: singularity, cuda..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed on your cluster

# --- Securely Load Neptune Credentials (for evaluate.py to use via Singularity env) ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds" # Assumes $HOME is correctly set for the Slurm user
NEPTUNE_API_TOKEN_FOR_SINGULARITY_ENV=""
NEPTUNE_PROJECT_FOR_SINGULARITY_ENV=""

if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE for Singularity environment."
    # Source in a subshell to avoid polluting this script's env if not needed directly
    # However, for exporting to SINGULARITYENV_, direct sourcing is fine.
    source "$NEPTUNE_CRED_FILE" # This exports NEPTUNE_API_TOKEN and NEPTUNE_PROJECT to this script's env
    NEPTUNE_API_TOKEN_FOR_SINGULARITY_ENV="${NEPTUNE_API_TOKEN:-}"
    NEPTUNE_PROJECT_FOR_SINGULARITY_ENV="${NEPTUNE_PROJECT:-thmorton/NothingProject}" # <<< UPDATE default project
else
    echo "WARNING (Evaluation Monitor): Neptune credentials file not found at $NEPTUNE_CRED_FILE. Neptune logging in evaluate.py might fail if token not otherwise available."
    NEPTUNE_PROJECT_FOR_SINGULARITY_ENV="thmorton/NothingProject" # <<< UPDATE default project
fi
export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN_FOR_SINGULARITY_ENV}"
export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT_FOR_SINGULARITY_ENV}"
# Pass the training run's Neptune name to evaluate.py via environment for linking/tagging
# SHARED_RUN_ID is already seed-specific (e.g., s42_jID_tsTIMESTAMP)
export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}"


# --- Define Paths on Host (using variables from orchestrator) ---
HOST_SIF_PATH="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE SIF name if different
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

# --- Define Container Mount Points (these are aliases inside Singularity) ---
CONTAINER_WORKSPACE="/workspace"                        # For project code (src/evaluate.py)
CONTAINER_DATA_DIR_EVAL="/data_eval_mnt"                # For validation datasets
CONTAINER_PRIMING_DIR_EVAL="/priming_data_eval_mnt"     # For priming datasets
CONTAINER_SHARED_OUTPUT_MOUNT="/mnt_checkpoints_output" # Mount point for SHARED_OUTPUT_DIR_HOST (where checkpoints are)
# CONTAINER_EVAL_RESULTS_TARGET_DIR is defined per-evaluation call inside run_evaluation_script

# Directory for this monitor job's own logs about evaluate.py executions
# This is on the HOST, within the SHARED_OUTPUT_DIR_HOST for the run.
EVALUATION_PY_EXECUTION_LOGS_DIR="${SHARED_OUTPUT_DIR_HOST}/evaluation_script_execution_logs"
mkdir -p "$EVALUATION_PY_EXECUTION_LOGS_DIR"
echo "Logs for individual evaluate.py script executions will be in: ${EVALUATION_PY_EXECUTION_LOGS_DIR}"

# --- Helper function to log messages from the monitor script ---
log_monitor_message() {
    echo "$(date): EVAL_MONITOR_INFO: $1"
}

# --- Function to run the evaluation script for a given checkpoint ---
run_evaluation_script() {
    local checkpoint_directory_name=$1 # e.g., "checkpoint-1000" or "final_model"

    # Construct full host path to the checkpoint directory
    local checkpoint_dir_host_path="${SHARED_OUTPUT_DIR_HOST}/${checkpoint_directory_name}"
    # Construct the path to this checkpoint as it will appear *inside the container*
    local checkpoint_path_for_evalpy_container="${CONTAINER_SHARED_OUTPUT_MOUNT}/${checkpoint_directory_name}"

    # Define a unique output directory *on the host* for this specific evaluation's results
    local host_eval_run_specific_output_dir="${SHARED_OUTPUT_DIR_HOST}/eval_results/${checkpoint_directory_name}"
    mkdir -p "$host_eval_run_specific_output_dir"
    # Define where this results directory will be mounted *inside the container* for evaluate.py to write to
    local container_eval_run_specific_output_target_dir="/eval_run_output_target" # evaluate.py --output_dir points here

    log_monitor_message "--------------------------------------------------------------------"
    log_monitor_message "Preparing to evaluate checkpoint: '${checkpoint_directory_name}'"
    log_monitor_message "Checkpoint Source (Host): ${checkpoint_dir_host_path}"
    log_monitor_message "Checkpoint Path for evaluate.py (Container): ${checkpoint_path_for_evalpy_container}"
    log_monitor_message "Evaluation Results Output Dir (Host): ${host_eval_run_specific_output_dir}"
    log_monitor_message "Evaluation Results Output Target (Container): ${container_eval_run_specific_output_target_dir}"
    log_monitor_message "Context: Shared Run ID='${SHARED_RUN_ID}', Seed='${SEED_FOR_EVAL}'"
    log_monitor_message "--------------------------------------------------------------------"

    # Define paths *inside the container* for datasets needed by evaluate.py
    local eval_py_validation_data_container_path="${CONTAINER_DATA_DIR_EVAL}/processed/test_set_10m" # <<< ADJUST if different
    local eval_py_priming_data_container_path="${CONTAINER_PRIMING_DIR_EVAL}/priming-corpuses"    # <<< ADJUST if different
    local eval_py_script_container_path="${CONTAINER_WORKSPACE}/src/evaluate.py"

    # Construct arguments for evaluate.py
    local -a PYTHON_ARGS_LIST # Use a bash array for clarity and safety
    PYTHON_ARGS_LIST+=( "--checkpoint_path" "${checkpoint_path_for_evalpy_container}" )
    PYTHON_ARGS_LIST+=( "--output_dir" "${container_eval_run_specific_output_target_dir}" )
    PYTHON_ARGS_LIST+=( "--checkpoint_label" "${checkpoint_directory_name}" ) # Pass the dir name as a label

    # Add flags for which evaluations to run (these are for evaluate.py)
    PYTHON_ARGS_LIST+=( "--run_standard_eval" ) # Example: always run standard perplexity
    PYTHON_ARGS_LIST+=( "--validation_dataset_path" "$eval_py_validation_data_container_path" )
    PYTHON_ARGS_LIST+=( "--run_priming_eval" )  # Example: always run priming
    PYTHON_ARGS_LIST+=( "--priming_eval_dir_path" "$eval_py_priming_data_container_path" )

    # Pass the SEED_FOR_EVAL (received from main_orchestrator) to evaluate.py's --seed argument
    if [ -n "$SEED_FOR_EVAL" ] && [[ "$SEED_FOR_EVAL" != "unknown_seed" ]]; then
        PYTHON_ARGS_LIST+=( "--seed" "$SEED_FOR_EVAL" )
    else
        PYTHON_ARGS_LIST+=( "--seed" "42" ) # Default seed for evaluate.py if not specifically provided
        log_monitor_message "WARNING: SEED_FOR_EVAL was not definitively set, using default seed 42 for evaluate.py."
    fi

    # Other parameters for evaluate.py
    PYTHON_ARGS_LIST+=( "--per_device_eval_batch_size" "16" ) # Example batch size
    PYTHON_ARGS_LIST+=( "--priming_per_device_eval_batch_size" "8" )  # Example batch size
    PYTHON_ARGS_LIST+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" ) # Use Slurm allocated CPUs
    PYTHON_ARGS_LIST+=( "--use_amp" ) # Example: assume AMP is desired if model was trained with it

    # Neptune arguments for evaluate.py
    # SINGULARITYENV_NEPTUNE_API_TOKEN and SINGULARITYENV_NEPTUNE_PROJECT are already exported.
    # Construct a specific Neptune run name for this evaluation instance.
    # SHARED_RUN_ID is like "s42_jJOBID_tsTIMESTAMP"
    local neptune_eval_run_name_for_py="eval_${SHARED_RUN_ID}_${checkpoint_directory_name}"
    PYTHON_ARGS_LIST+=( "--neptune_run_name" "${neptune_eval_run_name_for_py}" )

    # Construct Neptune tags, including an explicit seed tag.
    local -a neptune_tags_for_eval_py=("evaluation" "${SHARED_RUN_ID}" "${checkpoint_directory_name}")
    if [ -n "$SEED_FOR_EVAL" ] && [[ "$SEED_FOR_EVAL" != "unknown_seed" ]]; then
        neptune_tags_for_eval_py+=("seed_${SEED_FOR_EVAL}")
    fi
    # Add other tags if needed, e.g., GPU type. This monitor job knows its GPU via SLURM_JOB_PARTITION or similar.
    # neptune_tags_for_eval_py+=("${SLURM_JOB_PARTITION}")
    PYTHON_ARGS_LIST+=( "--neptune_tags" ) # Flag
    PYTHON_ARGS_LIST+=( "${neptune_tags_for_eval_py[@]}" ) # Expand array elements as separate arguments

    # Pass the Neptune project if available (evaluate.py will use SINGULARITYENV_ anyway)
    if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
         PYTHON_ARGS_LIST+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
    fi

    log_monitor_message "Arguments for evaluate.py:"
    # Print arguments one per line, quoting them for clarity
    printf "  %q\n" "${PYTHON_ARGS_LIST[@]}"

    # Set PyTorch CUDA Allocator Config if needed by evaluate.py
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Define log file for this specific Singularity execution of evaluate.py
    local eval_py_exec_log_file="${EVALUATION_PY_EXECUTION_LOGS_DIR}/exec_log_eval_py_${checkpoint_directory_name}_$(date +%H%M%S).txt"
    log_monitor_message "evaluate.py execution STDOUT/STDERR will be logged to: ${eval_py_exec_log_file}"

    # Execute evaluate.py inside Singularity
    singularity exec --nv \
        -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
        -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR_EVAL}" \
        -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR_EVAL}" \
        -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_SHARED_OUTPUT_MOUNT}" \
        -B "${host_eval_run_specific_output_dir}":"${container_eval_run_specific_output_target_dir}" \
        "${HOST_SIF_PATH}" \
        python3 "${eval_py_script_container_path}" "${PYTHON_ARGS_LIST[@]}" > "$eval_py_exec_log_file" 2>&1

    local evaluate_py_exit_code=$?
    if [ $evaluate_py_exit_code -eq 0 ]; then
        log_monitor_message "evaluate.py for checkpoint '${checkpoint_directory_name}' SUCCEEDED. Log: ${eval_py_exec_log_file}"
    else
        log_monitor_message "ERROR: evaluate.py for checkpoint '${checkpoint_directory_name}' FAILED with exit code ${evaluate_py_exit_code}. Check log: ${eval_py_exec_log_file}"
        # Consider further action: e.g., if one eval fails, should the monitor stop?
        # For now, it continues to monitor for other checkpoints.
    fi
    log_monitor_message "--------------------------------------------------------------------"
}

# --- Main Monitoring Loop ---
log_monitor_message "Starting to monitor ${SHARED_OUTPUT_DIR_HOST} for new checkpoints..."
declare -A PROCESSED_CHECKPOINTS_TRACKER # Associative array to track processed checkpoints (e.g., PROCESSED_CHECKPOINTS_TRACKER["checkpoint-1000"]=1)

# Define a timeout for the main monitoring loop to prevent it from running indefinitely
# if the training job fails silently without creating TRAINING_COMPLETED.txt.
# Slurm job time limit is the ultimate safeguard.
# This timeout is for the entire while loop's duration.
MAIN_LOOP_TIMEOUT_SECONDS=$((24 * 60 * 60 + 10 * 60)) # 24 hours and 10 minutes
SECONDS=0 # Bash built-in timer, reset to 0

TRAINING_COMPLETED_FLAG_FILE="${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"

while (( SECONDS < MAIN_LOOP_TIMEOUT_SECONDS )); do
    # Check for the training completion sentinel file first on each iteration
    if [ -f "${TRAINING_COMPLETED_FLAG_FILE}" ]; then
        log_monitor_message "TRAINING_COMPLETED.txt sentinel file detected at ${TRAINING_COMPLETED_FLAG_FILE}."
        # After training completes, perform one final scan for any checkpoints that might have appeared
        # just before the sentinel file or were missed due to inotifywait timing.
        log_monitor_message "Performing a final scan of ${SHARED_OUTPUT_DIR_HOST} for any remaining checkpoints..."
        # Loop through subdirectories in SHARED_OUTPUT_DIR_HOST
        find "${SHARED_OUTPUT_DIR_HOST}" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d $'\0' potential_ckpt_dir_path; do
            dir_basename=$(basename "$potential_ckpt_dir_path")
            if [[ "$dir_basename" == checkpoint-* || "$dir_basename" == "final_model" ]]; then
                if [[ -z "${PROCESSED_CHECKPOINTS_TRACKER[$dir_basename]}" ]]; then # If not yet processed
                    log_monitor_message "Final scan: Found unprocessed directory '${dir_basename}'. Processing now."
                    run_evaluation_script "$dir_basename"
                    PROCESSED_CHECKPOINTS_TRACKER["$dir_basename"]=1 # Mark as processed
                fi
            fi
        done
        log_monitor_message "Final scan complete. Exiting evaluation monitor as training is finished."
        break # Exit the while loop
    fi

    # Use inotifywait to watch for new directories (events: create, moved_to)
    # -t 60: timeout for inotifywait itself, so the loop can re-check TRAINING_COMPLETED_FLAG_FILE periodically.
    # --format '%e %f': output event type and filename.
    # Watch the SHARED_OUTPUT_DIR_HOST for new items.
    inotify_output_line=$(inotifywait -q -t 60 -e create -e moved_to --format '%e %f' "${SHARED_OUTPUT_DIR_HOST}" 2>/dev/null)
    # -q for quiet, suppress non-event messages from inotifywait

    if [ $? -eq 0 ] && [ -n "$inotify_output_line" ]; then # Event occurred and output is not empty
        # Read multiple events if they occurred quickly (though inotifywait often returns one line per call here)
        echo "$inotify_output_line" | while IFS= read -r event_details; do
            event_type=$(echo "$event_details" | awk '{print $1}')
            item_name=$(echo "$event_details" | awk '{print $2}')
            full_item_path_on_host="${SHARED_OUTPUT_DIR_HOST}/${item_name}"

            log_monitor_message "Detected filesystem event: '${event_type}' for item: '${item_name}'"

            # Check if the new item is a directory and matches expected checkpoint patterns
            if [ -d "$full_item_path_on_host" ]; then
                if [[ "$item_name" == checkpoint-* || "$item_name" == "final_model" ]]; then
                    # Check if this checkpoint has already been processed
                    if [[ -z "${PROCESSED_CHECKPOINTS_TRACKER[$item_name]}" ]]; then
                        log_monitor_message "New valid checkpoint directory detected: '${item_name}'. Initiating evaluation."
                        run_evaluation_script "$item_name"
                        PROCESSED_CHECKPOINTS_TRACKER["$item_name"]=1 # Mark as processed
                    else
                        log_monitor_message "Checkpoint directory '${item_name}' was already processed or seen. Skipping."
                    fi
                # Optional: Log if other unexpected directories are created
                # else
                #    log_monitor_message "Detected new directory '${item_name}', but it does not match checkpoint pattern. Ignoring."
                fi
            # Optional: Log if non-directory items are created (e.g., files like .DS_Store, etc.)
            # else
            #    log_monitor_message "Detected new item '${item_name}', but it is not a directory. Ignoring for checkpoint processing."
            fi
        done
    else # inotifywait timed out (ret code > 0 or empty output) or other error
        # This is normal if no new checkpoints were saved in the last 60 seconds.
        # The loop will continue, checking TRAINING_COMPLETED_FLAG_FILE and then restarting inotifywait.
        log_monitor_message "No new checkpoint activity in the last 60 seconds (or inotifywait error). Continuing to monitor..."
    fi
    # No explicit sleep here, as inotifywait -t 60 provides the periodic check.
done


# After the loop finishes (either by break due to TRAINING_COMPLETED_FLAG_FILE or by MAIN_LOOP_TIMEOUT_SECONDS)
if (( SECONDS >= MAIN_LOOP_TIMEOUT_SECONDS )); then
    log_monitor_message "WARNING: Evaluation monitor main loop timed out after ${MAIN_LOOP_TIMEOUT_SECONDS} seconds."
    # Perform one absolute final check for the completion flag if timeout occurred, as it might have appeared right at the end.
    if [ -f "${TRAINING_COMPLETED_FLAG_FILE}" ]; then
        log_monitor_message "TRAINING_COMPLETED.txt sentinel file was found after main loop timeout. Assuming training did complete."
    else
        log_monitor_message "CRITICAL WARNING: Main loop timed out AND TRAINING_COMPLETED.txt sentinel file was NOT found. Training might be incomplete, stuck, or failed to create the sentinel file."
        # Consider sending a specific alert or exiting with a different code if this happens.
    fi
fi

log_monitor_message "Shutting down evaluation monitor."
echo "=== Evaluation Monitor Job Finished: $(date) ==="