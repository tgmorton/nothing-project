#!/bin/bash
# === SBATCH Directives for Evaluation Monitor ===
#SBATCH --job-name=eval_monitor_job      # Will be overridden by orchestrator
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU for evaluation (p6000 or A5000, etc.)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for the monitor and evaluate.py
#SBATCH --mem=32G                        # RAM for evaluate.py
#SBATCH --time=7-00:00:00                   # Slightly longer than training
#SBATCH --output=../logs/default_eval_%j.out # Will be overridden by orchestrator
#SBATCH --error=../logs/default_eval_%j.err  # Will be overridden by orchestrator
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
HOST_SIF_PATH="${HOST_PROJECT_DIR}/python39_annotate.sif" # <<< UPDATE if SIF name/location differs
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


# --- Function to run evaluation ---
run_evaluation_script() {
    local checkpoint_dirname=$1 # e.g., checkpoint-400 or final_model
    # SHARED_OUTPUT_DIR_HOST is the host path to the shared output directory for the current seed's run
    # It should be available as an environment variable exported by main_orchestrator.sh
    local checkpoint_host_full_path="${SHARED_OUTPUT_DIR_HOST}/${checkpoint_dirname}"

    # CONTAINER_SHARED_OUTPUT_MOUNT is the mount point inside the container for SHARED_OUTPUT_DIR_HOST
    # (e.g., /mnt_shared_output defined in evaluation_monitor.sh's singularity command)
    local checkpoint_container_path_for_eval_py="${CONTAINER_SHARED_OUTPUT_MOUNT}/${checkpoint_dirname}"

    # Define a unique output directory for this specific evaluation's results *on the host*
    local host_eval_specific_results_dir="${SHARED_OUTPUT_DIR_HOST}/eval_results/${checkpoint_dirname}"
    mkdir -p "$host_eval_specific_results_dir"
    # Define where this specific results directory will be mounted *inside the container* for evaluate.py to write to
    local container_eval_specific_results_target_dir="/eval_output_target"


    echo # Blank line for readability
    echo "--------------------------------------------------------------------"
    echo "$(date): EVALUATION MONITOR: Preparing to evaluate checkpoint: ${checkpoint_dirname}"
    echo "Checkpoint Source (Host): ${checkpoint_host_full_path}"
    echo "Checkpoint Path for evaluate.py (Container): ${checkpoint_container_path_for_eval_py}"
    echo "Evaluation Results Output Dir (Host): ${host_eval_specific_results_dir}"
    echo "Evaluation Results Output Target (Container): ${container_eval_specific_results_target_dir}"
    echo "Shared Run ID for this eval context: ${SHARED_RUN_ID}"
    echo "Seed for this eval context: ${SEED_FOR_EVAL:-"Not Set"}" # SEED_FOR_EVAL comes from main_orchestrator
    echo "--------------------------------------------------------------------"

    # Define paths for evaluate.py script *inside the container*
    # CONTAINER_DATA_DIR_EVAL and CONTAINER_PRIMING_DIR_EVAL are mount points defined in evaluation_monitor.sh
    local EVAL_PY_VALID_DATA_PATH_CONTAINER="${CONTAINER_DATA_DIR_EVAL}/processed/test_set_10m" # Adjust as needed
    local EVAL_PY_PRIMING_DATA_PATH_CONTAINER="${CONTAINER_PRIMING_DIR_EVAL}/priming-corpuses"  # Adjust as needed
    local EVAL_PY_SCRIPT_PATH_CONTAINER="${CONTAINER_WORKSPACE}/src/evaluate.py" # Path to evaluate.py inside container

    # Construct arguments for evaluate.py
    PYTHON_ARGS=()
    PYTHON_ARGS+=( "--checkpoint_path" "${checkpoint_container_path_for_eval_py}" )
    PYTHON_ARGS+=( "--output_dir" "${container_eval_specific_results_target_dir}" ) # evaluate.py writes here
    PYTHON_ARGS+=( "--checkpoint_label" "${checkpoint_dirname}" ) # Pass the checkpoint name as a label

    # Add flags/paths conditionally based on what evaluations to run
    # These flags are for evaluate.py
    PYTHON_ARGS+=( "--run_standard_eval" ) # Example: always run standard eval
    PYTHON_ARGS+=( "--validation_dataset_path" "$EVAL_PY_VALID_DATA_PATH_CONTAINER" )
    PYTHON_ARGS+=( "--run_priming_eval" )  # Example: always run priming eval
    PYTHON_ARGS+=( "--priming_eval_dir_path" "$EVAL_PY_PRIMING_DATA_PATH_CONTAINER" )

    # Pass the seed for reproducibility within evaluate.py (e.g., for dataset sampling if any)
    # SEED_FOR_EVAL is expected to be set by main_orchestrator via --export
    if [ -n "$SEED_FOR_EVAL" ]; then
        PYTHON_ARGS+=( "--seed" "$SEED_FOR_EVAL" )
    else
        PYTHON_ARGS+=( "--seed" "42" ) # Default seed for evaluate.py if not explicitly passed
        logger "WARNING (Evaluation Monitor): SEED_FOR_EVAL not set, using default 42 for evaluate.py."
    fi

    # Other evaluation parameters for evaluate.py
    PYTHON_ARGS+=( "--per_device_eval_batch_size" "16" ) # Example
    PYTHON_ARGS+=( "--priming_per_device_eval_batch_size" "8" )  # Example
    PYTHON_ARGS+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" ) # Use Slurm allocated CPUs
    PYTHON_ARGS+=( "--use_amp" ) # Example: assume AMP is desired if model supports it

    # Neptune arguments for the evaluate.py script
    # SINGULARITYENV_NEPTUNE_API_TOKEN and SINGULARITYENV_NEPTUNE_PROJECT are exported globally in evaluation_monitor.sh
    # evaluate.py will pick those up.
    # SHARED_RUN_ID already contains seed information (e.g., s42_jID_tsTIMESTAMP)
    # Construct a specific Neptune run name for this evaluation instance.
    local neptune_eval_run_name_for_py="eval_${SHARED_RUN_ID}_${checkpoint_dirname}"
    PYTHON_ARGS+=( "--neptune_run_name" "${neptune_eval_run_name_for_py}" )

    # Construct Neptune tags, including an explicit seed tag if SEED_FOR_EVAL is available
    local neptune_eval_tags_list=("evaluation" "${SHARED_RUN_ID}" "${checkpoint_dirname}")
    if [ -n "$SEED_FOR_EVAL" ]; then
        neptune_eval_tags_list+=("seed_${SEED_FOR_EVAL}")
    fi
    # Add other relevant tags if needed, e.g., GPU type if it varies for eval
    # neptune_eval_tags_list+=("p6000_eval_node")
    local neptune_tags_arg_for_py=$(IFS=" "; echo "${neptune_eval_tags_list[*]}") # Convert bash array to space-separated string
    PYTHON_ARGS+=( "--neptune_tags" ${neptune_tags_arg_for_py} ) # Pass as multiple arguments if tags contain spaces

    # If neptune_project is explicitly passed to evaluate.py (it will use SINGULARITYENV_ otherwise)
    if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
         PYTHON_ARGS+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
    fi
    # Note: SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME is also available if evaluate.py wants to use it

    echo "Arguments for evaluate.py:"
    printf "  %q\n" "${PYTHON_ARGS[@]}" # Print each argument quoted on a new line for clarity

    # Set PyTorch CUDA Allocator Config if needed by evaluate.py
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Define log file for this specific Singularity execution of evaluate.py
    # EVALUATION_RUN_LOGS_DIR is defined globally in evaluation_monitor.sh (e.g., "${SHARED_OUTPUT_DIR_HOST}/evaluation_job_logs")
    local eval_py_execution_log="${EVALUATION_RUN_LOGS_DIR}/eval_py_${checkpoint_dirname}_$(date +%H%M%S).log"
    echo "evaluate.py execution log will be: ${eval_py_execution_log}"

    # Execute Singularity for evaluate.py
    # Bind mounts:
    # HOST_PROJECT_DIR -> CONTAINER_WORKSPACE (for src/evaluate.py)
    # HOST_DATA_BASE_DIR -> CONTAINER_DATA_DIR_EVAL (for validation_dataset_path)
    # HOST_PRIMING_BASE_DIR -> CONTAINER_PRIMING_DIR_EVAL (for priming_eval_dir_path)
    # SHARED_OUTPUT_DIR_HOST -> CONTAINER_SHARED_OUTPUT_MOUNT (for --checkpoint_path)
    # host_eval_specific_results_dir -> container_eval_specific_results_target_dir (for --output_dir of evaluate.py)
    singularity exec --nv \
        -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
        -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR_EVAL}" \
        -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR_EVAL}" \
        -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_SHARED_OUTPUT_MOUNT}" \
        -B "${host_eval_specific_results_dir}":"${container_eval_specific_results_target_dir}" \
        "${HOST_SIF_PATH}" \
        python3 "${EVAL_PY_SCRIPT_PATH_CONTAINER}" "${PYTHON_ARGS[@]}" > "$eval_py_execution_log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$(date): EVALUATION MONITOR: evaluate.py for ${checkpoint_dirname} SUCCEEDED. Log: ${eval_py_execution_log}"
    else
        echo "$(date): EVALUATION MONITOR: evaluate.py for ${checkpoint_dirname} FAILED with exit code ${exit_code}. Check log: ${eval_py_execution_log}"
        # Optionally, you could add logic here to retry or mark the overall run as problematic.
    fi
    echo "--------------------------------------------------------------------"
    echo # Blank line for readability
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