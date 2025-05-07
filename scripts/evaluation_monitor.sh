#!/bin/bash
# === SBATCH Directives for Evaluation Monitor ===
#SBATCH --job-name=eval_monitor_job      # Will be overridden by orchestrator
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU for evaluation (p6000 or A5000, etc.)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for the monitor and evaluate.py
#SBATCH --mem=32G                        # RAM for evaluate.py
#SBATCH --gres=gpu:1                     # <<< Request 1 GPU for evaluation
#SBATCH --time=24:30:00                  # Slightly longer than training
#SBATCH --output=logs/default_eval_%j.out # Will be overridden by orchestrator
#SBATCH --error=logs/default_eval_%j.err  # Will be overridden by orchestrator
#SBATCH --mail-type=END
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# set -e # Be cautious with 'set -e' inside a loop that needs to continue on some errors

echo "=== Evaluation Monitor Job Started: $(date) ==="
echo "Evaluation Monitor Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Host Project Dir (from Orchestrator): ${HOST_PROJECT_DIR}"
echo "Shared Output Directory (from Orchestrator): ${SHARED_OUTPUT_DIR}"
echo "Shared Run ID (from Orchestrator): ${SHARED_RUN_ID}"

# Validate variables from orchestrator
if [ -z "$HOST_PROJECT_DIR" ] || [ -z "$SHARED_OUTPUT_DIR" ] || [ -z "$SHARED_RUN_ID" ]; then
    echo "ERROR: Critical environment variables (HOST_PROJECT_DIR, SHARED_OUTPUT_DIR, SHARED_RUN_ID) not set by orchestrator!"
    exit 1
fi

# Check for inotifywait
if ! command -v inotifywait &> /dev/null; then
    echo "ERROR: inotifywait command could not be found. Please install inotify-tools on the compute node."
    echo "Alternatively, implement a Python-based watcher inside the Singularity container."
    exit 1
fi

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Securely Load Neptune Credentials (for evaluate.py) ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds"
NEPTUNE_API_TOKEN_FOR_SINGULARITY=""
NEPTUNE_PROJECT_FOR_SINGULARITY=""

if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE" # This exports NEPTUNE_API_TOKEN and NEPTUNE_PROJECT to the script's env
    NEPTUNE_API_TOKEN_FOR_SINGULARITY="${NEPTUNE_API_TOKEN:-}"
    NEPTUNE_PROJECT_FOR_SINGULARITY="${NEPTUNE_PROJECT:-thmorton/NothingProject}" # Fallback for project
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE."
    NEPTUNE_PROJECT_FOR_SINGULARITY="thmorton/NothingProject" # Fallback for project <<< UPDATE
fi
export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN_FOR_SINGULARITY}"
export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT_FOR_SINGULARITY}"
# This is the Neptune run name from the training job. evaluate.py might use this for tagging/linking.
export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}"


# --- Define Paths on Host (using orchestrator-provided HOST_PROJECT_DIR) ---
HOST_SIF_PATH="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE if SIF name/location differs
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

# --- Define Container Paths (consistent for evaluation runs) ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR_EVAL="/data_eval"          # Distinct mount for clarity if needed
CONTAINER_PRIMING_DIR_EVAL="/eval_data_eval"  # Distinct mount for clarity if needed
CONTAINER_SHARED_OUTPUT_MOUNT="/mnt_shared_output" # Mount point for SHARED_OUTPUT_DIR

# Directory for evaluation-specific logs within the SHARED_OUTPUT_DIR
EVALUATION_RUN_LOGS_DIR="${SHARED_OUTPUT_DIR}/evaluation_job_logs"
mkdir -p "$EVALUATION_RUN_LOGS_DIR"

# --- Function to run evaluation using evaluate.py ---
run_evaluation_script() {
    local checkpoint_dirname=$1 # e.g., "checkpoint-1200" or "final_model"
    # Construct full path to the checkpoint *inside the container*
    local container_checkpoint_path="${CONTAINER_SHARED_OUTPUT_MOUNT}/${checkpoint_dirname}"

    # Define a unique output directory for this specific evaluation run *on the host*
    local host_eval_output_subdir="${SHARED_OUTPUT_DIR}/eval_results/${checkpoint_dirname}"
    mkdir -p "$host_eval_output_subdir"
    # Define where this subdir will be mounted *inside the container* for evaluate.py to write to
    local container_eval_output_path="/eval_output_target"

    echo "-----------------------------------------------------"
    echo "$(date): Preparing to evaluate: ${checkpoint_dirname}"
    echo "Host Checkpoint Dir: ${SHARED_OUTPUT_DIR}/${checkpoint_dirname}"
    echo "Container Checkpoint Path for evaluate.py: ${container_checkpoint_path}"
    echo "Host Evaluation Output Dir for this run: ${host_eval_output_subdir}"
    echo "Container Evaluation Output Path for evaluate.py: ${container_eval_output_path}"
    echo "-----------------------------------------------------"

    # Paths for evaluate.py script *inside the container*
    local EVAL_PY_VALID_DATA_PATH="${CONTAINER_DATA_DIR_EVAL}/processed/test_set_10m"
    local EVAL_PY_PRIMING_DATA_PATH="${CONTAINER_PRIMING_DIR_EVAL}/priming-corpuses"
    local EVAL_PY_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/evaluate.py"

    # Construct arguments for evaluate.py
    PYTHON_ARGS=()
    PYTHON_ARGS+=( "--checkpoint_path" "${container_checkpoint_path}" )
    PYTHON_ARGS+=( "--output_dir" "${container_eval_output_path}" ) # evaluate.py writes here

    PYTHON_ARGS+=( "--run_standard_eval" ) # Assuming these are flags in your evaluate.py
    PYTHON_ARGS+=( "--validation_dataset_path" "$EVAL_PY_VALID_DATA_PATH" )
    PYTHON_ARGS+=( "--run_priming_eval" )  # Assuming these are flags
    PYTHON_ARGS+=( "--priming_eval_dir_path" "$EVAL_PY_PRIMING_DATA_PATH" )

    PYTHON_ARGS+=( "--seed" "${SEED:-42}" ) # Use SEED from env if set, else default
    PYTHON_ARGS+=( "--per_device_eval_batch_size" "16" ) # Adjust as needed
    PYTHON_ARGS+=( "--priming_per_device_eval_batch_size" "8" ) # Adjust
    PYTHON_ARGS+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" )
    PYTHON_ARGS+=( "--use_amp" ) # If evaluate.py supports it

    # Neptune arguments for the evaluation run
    # Each evaluation can be a new Neptune run, tagged to link to the main training run.
    local neptune_eval_run_name="eval_${SHARED_RUN_ID}_${checkpoint_dirname}"
    local neptune_eval_tags="evaluation ${SHARED_RUN_ID} ${checkpoint_dirname} p6000_eval" # Customize

    PYTHON_ARGS+=( "--neptune_run_name" "${neptune_eval_run_name}" )
    PYTHON_ARGS+=( "--neptune_tags" ${neptune_eval_tags} )
    # SINGULARITYENV_NEPTUNE_PROJECT is already exported globally for singularity

    echo "Arguments for evaluate.py:"
    printf "* %q\n" "${PYTHON_ARGS[@]}"

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # If needed by evaluate.py

    # Log file for this specific evaluation execution
    local eval_exec_log="${EVALUATION_RUN_LOGS_DIR}/eval_${checkpoint_dirname}_$(date +%H%M%S).log"

    singularity exec --nv \
        -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
        -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR_EVAL}" \
        -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR_EVAL}" \
        -B "${SHARED_OUTPUT_DIR}":"${CONTAINER_SHARED_OUTPUT_MOUNT}" \
        -B "${host_eval_output_subdir}":"${container_eval_output_path}" \
        "${HOST_SIF_PATH}" \
        python3 "${EVAL_PY_SCRIPT_PATH}" "${PYTHON_ARGS[@]}" > "$eval_exec_log" 2>&1

    if [ $? -eq 0 ]; then
        echo "$(date): Evaluation for ${checkpoint_dirname} SUCCEEDED. Log: ${eval_exec_log}"
    else
        echo "$(date): Evaluation for ${checkpoint_dirname} FAILED. Check log: ${eval_exec_log}"
    fi
    echo "-----------------------------------------------------"
}

# --- Monitoring Loop ---
echo "$(date): Starting to monitor ${SHARED_OUTPUT_DIR} for new checkpoints..."
declare -A PROCESSED_CHECKPOINTS # Associative array to track processed checkpoints

# Loop for a maximum duration or until training_completed sentinel is found
# Slurm time limit is the ultimate stop. This is an additional safeguard.
LOOP_TIMEOUT_SECONDS=$((24 * 60 * 60 + 15 * 60)) # 24h 15m
SECONDS=0 # Bash timer

while (( SECONDS < LOOP_TIMEOUT_SECONDS )); do
    # Check for TRAINING_COMPLETED.txt first
    if [ -f "${SHARED_OUTPUT_DIR}/TRAINING_COMPLETED.txt" ]; then
        echo "$(date): TRAINING_COMPLETED.txt detected."
        # Perform one last scan for any missed checkpoints or final_model
        for item in "${SHARED_OUTPUT_DIR}"/*/; do # Iterate only directories
            item_name=$(basename "$item")
            if [[ "$item_name" == checkpoint-* || "$item_name" == "final_model" ]]; then
                if [[ -z "${PROCESSED_CHECKPOINTS[$item_name]}" ]]; then # If not processed
                    echo "$(date): Found unprocessed item ${item_name} after training completion signal. Processing..."
                    run_evaluation_script "$item_name"
                    PROCESSED_CHECKPOINTS["$item_name"]=1
                fi
            fi
        done
        echo "$(date): All processing finished after training completion. Exiting monitor."
        break # Exit the while loop
    fi

    # Use inotifywait to watch for new directories (CREATE event where a new item is a directory)
    # and MOVED_TO in case directories are moved in.
    # Timeout for inotifywait to allow periodic check of TRAINING_COMPLETED.txt
    inotify_event_info=$(inotifywait -t 60 -e create -e moved_to --format '%e %f' "${SHARED_OUTPUT_DIR}" 2>/dev/null)

    if [ -n "$inotify_event_info" ]; then
        while IFS= read -r line; do
            event_type=$(echo "$line" | awk '{print $1}')
            new_item_name=$(echo "$line" | awk '{print $2}')
            full_new_item_path="${SHARED_OUTPUT_DIR}/${new_item_name}"

            echo "$(date): Detected event '${event_type}' for item '${new_item_name}'"

            # Process only if it's a directory and matches checkpoint-* or final_model pattern
            if [ -d "$full_new_item_path" ]; then
                if [[ "$new_item_name" == checkpoint-* || "$new_item_name" == "final_model" ]]; then
                    if [[ -z "${PROCESSED_CHECKPOINTS[$new_item_name]}" ]]; then # If not yet processed
                        echo "$(date): New valid directory detected: $new_item_name. Queueing for evaluation."
                        run_evaluation_script "$new_item_name"
                        PROCESSED_CHECKPOINTS["$new_item_name"]=1 # Mark as processed
                    else
                        echo "$(date): Directory $new_item_name already processed or seen, skipping."
                    fi
                fi
            fi
        done <<< "$inotify_event_info"
    else
        echo -n "." # Print a dot for heartbeat during no-event periods
    fi
    # Brief sleep if inotifywait timed out, before checking TRAINING_COMPLETED.txt again
    # This is already handled by inotifywait's timeout.
done

if (( SECONDS >= LOOP_TIMEOUT_SECONDS )); then
    echo "$(date): Evaluation monitor loop timed out after ${LOOP_TIMEOUT_SECONDS}s."
    # Check one last time for the completion signal if loop timed out
    if [ -f "${SHARED_OUTPUT_DIR}/TRAINING_COMPLETED.txt" ]; then
        echo "$(date): TRAINING_COMPLETED.txt found after loop timeout. Perform final scan."
        for item in "${SHARED_OUTPUT_DIR}"/*/; do
             item_name=$(basename "$item")
            if [[ "$item_name" == checkpoint-* || "$item_name" == "final_model" ]]; then
                if [[ -z "${PROCESSED_CHECKPOINTS[$item_name]}" ]]; then
                    run_evaluation_script "$item_name"
                    PROCESSED_CHECKPOINTS["$item_name"]=1
                fi
            fi
        done
    else
        echo "$(date): WARNING - Loop timed out AND TRAINING_COMPLETED.txt not found. Training might be incomplete or stuck."
    fi
fi

echo "=== Evaluation Monitor Job Finished: $(date) ==="