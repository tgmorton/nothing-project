#!/bin/bash
# main_orchestrator.sh
# Job name, output/error paths will be overridden if launched by sweep_coordinator
# or can be set here for standalone runs.
#SBATCH --job-name=main_orch_standalone
#SBATCH --partition=general       # CPU partition for orchestrator itself
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00            # Max time for ONE full pipeline (train + eval monitor)
                                  # Slightly less than training_job's time to allow it to finish.
#SBATCH --output=../logs/main_orchestrator_%x_%j.out # Default for standalone
#SBATCH --error=../logs/main_orchestrator_%x_%j.err  # Default for standalone
#SBATCH --mail-type=END,FAIL      # Notify on end/fail of this orchestrator job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error for this orchestrator script
set -e
# set -o pipefail # Exit if any command in a pipeline fails

echo "=== Main Orchestrator Script Started: $(date) ==="
echo "Main Orchestrator Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"

# --- Handle HOST_PROJECT_DIR (from parent/submitter or default) ---
DEFAULT_HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # <<< UPDATE THIS DEFAULT
if [ -n "$HOST_PROJECT_DIR" ]; then
    echo "Using HOST_PROJECT_DIR from environment: ${HOST_PROJECT_DIR}"
else
    HOST_PROJECT_DIR="${DEFAULT_HOST_PROJECT_DIR}"
    echo "HOST_PROJECT_DIR not set in environment, using default: ${HOST_PROJECT_DIR}"
fi
if [ ! -d "$HOST_PROJECT_DIR" ]; then
    echo "CRITICAL ERROR (Main Orchestrator): Effective HOST_PROJECT_DIR does not exist: ${HOST_PROJECT_DIR}"
    exit 1
fi

# --- Handle CURRENT_SEED_TO_RUN (from parent/submitter or default for standalone) ---
DEFAULT_SEED_FOR_STANDALONE="42" # Default seed if running standalone
if [ -n "$CURRENT_SEED_TO_RUN" ]; then
    echo "Using CURRENT_SEED_TO_RUN from environment: ${CURRENT_SEED_TO_RUN}"
else
    CURRENT_SEED_TO_RUN="${DEFAULT_SEED_FOR_STANDALONE}"
    echo "CURRENT_SEED_TO_RUN not set in environment, using default for standalone run: ${CURRENT_SEED_TO_RUN}"
fi
# Validate seed is a number (optional, but good practice)
if ! [[ "$CURRENT_SEED_TO_RUN" =~ ^[0-9]+$ ]]; then
    echo "CRITICAL ERROR (Main Orchestrator): CURRENT_SEED_TO_RUN is not a valid number: '${CURRENT_SEED_TO_RUN}'"
    exit 1
fi

# --- Define Shared Base Paths (using effective HOST_PROJECT_DIR) ---
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output" # Base for all experimental outputs
# Logs for children jobs (training, eval_monitor) launched by THIS main_orchestrator instance
CHILD_JOB_LOGS_DIR="${HOST_PROJECT_DIR}/logs/main_orch_children/s${CURRENT_SEED_TO_RUN}_job${SLURM_JOB_ID}"

# --- Create a Unique Run ID Incorporating the Seed and Orchestrator Job ID ---
# This ensures outputs and logs for this specific seed run are grouped.
# Using SLURM_JOB_NAME in SHARED_RUN_ID can be useful if job name is unique.
# Using SLURM_JOB_ID is more robust for uniqueness.
JOB_ID_FOR_RUN_NAME="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" # Use array job ID if part of array, else job ID
SHARED_RUN_ID="s${CURRENT_SEED_TO_RUN}_j${JOB_ID_FOR_RUN_NAME}_ts$(date +%Y%m%d_%H%M%S)"
SHARED_OUTPUT_DIR_HOST="${HOST_OUTPUT_BASE_DIR}/${SHARED_RUN_ID}"

echo "Shared Run ID for Seed ${CURRENT_SEED_TO_RUN}: ${SHARED_RUN_ID}"
echo "Shared Output Directory (Host) for this run: ${SHARED_OUTPUT_DIR_HOST}"

# --- Create Directories for this run ---
mkdir -p "${SHARED_OUTPUT_DIR_HOST}"
mkdir -p "${CHILD_JOB_LOGS_DIR}"
echo "Ensured shared output and child log directories exist for this run."

# Define paths to child sbatch scripts (relative to HOST_PROJECT_DIR)
PATH_TO_TRAINING_JOB="${HOST_PROJECT_DIR}/scripts/training_job.sh" # <<< UPDATE THIS PATH
PATH_TO_EVAL_MONITOR_JOB="${HOST_PROJECT_DIR}/scripts/evaluation_monitor.sh" # <<< UPDATE THIS PATH

if [ ! -f "$PATH_TO_TRAINING_JOB" ]; then echo "CRITICAL ERROR: training_job.sh not found at ${PATH_TO_TRAINING_JOB}"; exit 1; fi
if [ ! -f "$PATH_TO_EVAL_MONITOR_JOB" ]; then echo "CRITICAL ERROR: evaluation_monitor.sh not found at ${PATH_TO_EVAL_MONITOR_JOB}"; exit 1; fi


# --- Launch Training Job ---
echo "Main Orchestrator: Submitting Training Job for Seed ${CURRENT_SEED_TO_RUN} (Run ID: ${SHARED_RUN_ID})..."
# Pass all necessary variables including the specific SEED_FOR_TRAINING
# SHARED_OUTPUT_DIR_HOST is the critical path for the training job to write to.
TRAIN_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_TRAINING="${CURRENT_SEED_TO_RUN}" \
    --job-name="train_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_TRAINING_JOB}")

if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Training Job. Output from sbatch: ${TRAIN_JOB_ID_OUTPUT}"; exit 1; fi
TRAIN_JOB_ID_NUM=$(echo "$TRAIN_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$TRAIN_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Training Job ID from sbatch output: '${TRAIN_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Training Job for Seed ${CURRENT_SEED_TO_RUN} submitted. Job ID: ${TRAIN_JOB_ID_NUM}."

# --- Launch Evaluation Monitor Job ---
echo "Main Orchestrator: Submitting Evaluation Monitor Job for Seed ${CURRENT_SEED_TO_RUN} (Run ID: ${SHARED_RUN_ID})..."
# Pass SEED_FOR_EVAL for explicit tagging if needed by evaluate.py, SHARED_RUN_ID already contains seed info
# SHARED_OUTPUT_DIR_HOST is also critical for the monitor to watch.
EVAL_MONITOR_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_EVAL="${CURRENT_SEED_TO_RUN}" \
    --job-name="eval_mon_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/eval_monitor_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/eval_monitor_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_EVAL_MONITOR_JOB}")

if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Evaluation Monitor Job. Output from sbatch: ${EVAL_MONITOR_JOB_ID_OUTPUT}"; exit 1; fi
EVAL_MONITOR_JOB_ID_NUM=$(echo "$EVAL_MONITOR_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$EVAL_MONITOR_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Eval Monitor Job ID from sbatch output: '${EVAL_MONITOR_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Evaluation Monitor Job for Seed ${CURRENT_SEED_TO_RUN} submitted. Job ID: ${EVAL_MONITOR_JOB_ID_NUM}."

# --- Function to wait for child Slurm jobs specific to this orchestrator ---
# Returns 0 on COMPLETED, 1 on other terminal states (FAILED, CANCELLED, etc.)
wait_for_child_job() {
    local job_id=$1
    local job_name_desc=$2
    echo "Main Orchestrator: Waiting for child job '${job_name_desc}' ($job_id)..."
    while true; do
        status_line=$(sacct -j "$job_id" --format=State --noheader | head -n 1)
        current_child_status=$(echo "$status_line" | awk '{print $1}' | sed 's/(.*)//') # Clean status
        case "$current_child_status" in
            COMPLETED)
                echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) COMPLETED."
                return 0 # Success
                ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY)
                echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) ended with status: $current_child_status. This orchestrator will also report failure."
                return 1 # Critical failure
                ;;
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING)
                echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) status: $current_child_status. Waiting..."
                ;;
            "") # Empty status from sacct might mean job is old or never existed.
                # Check squeue as a fallback for very recently submitted jobs.
                if squeue -h -j "$job_id" | grep -q "$job_id"; then
                    echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) status empty from sacct, but found in squeue. Assuming PENDING/RUNNING..."
                else
                    # This is ambiguous. Could be a very fast completion, or a silent sbatch failure not caught earlier.
                    echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) no longer in sacct or squeue. This is unusual. Assuming it finished. Check its logs for actual status."
                    # For now, let's assume it's okay to break the wait loop. The job's own logs are the source of truth.
                    # The critical part is that if the job *did* fail, its Slurm exit code should reflect that.
                    return 0 # Let's assume okay and rely on the job's output/error files.
                fi
                ;;
            *) # Unknown or unexpected status
                echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) has unknown status: '$current_child_status'. Waiting..."
                ;;
        esac
        sleep 120 # Check child job status every 2 minutes
    done
}

# Wait for the training job to complete
wait_for_child_job "$TRAIN_JOB_ID_NUM" "Training Job (Seed ${CURRENT_SEED_TO_RUN}, Run ${SHARED_RUN_ID})"
train_wait_status=$? # Capture return status
if [ $train_wait_status -ne 0 ]; then
    echo "CRITICAL ERROR (Main Orchestrator): Training job ${TRAIN_JOB_ID_NUM} did not complete successfully. Halting this orchestrator run."
    exit 1 # This non-zero exit will signal failure to the sweep_coordinator if applicable
fi

# After training job completes, verify the sentinel file exists.
# This is an important handshake for the evaluation monitor.
TRAINING_SENTINEL_FILE="${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"
if [ ! -f "${TRAINING_SENTINEL_FILE}" ]; then
    echo "CRITICAL WARNING (Main Orchestrator): Training job ${TRAIN_JOB_ID_NUM} completed, BUT the sentinel file '${TRAINING_SENTINEL_FILE}' was NOT FOUND."
    echo "The Evaluation Monitor job might wait indefinitely or fail if it strictly relies on this file."
    # Depending on policy, you might want to exit 1 here to mark the orchestrator run as failed.
    # For now, it's a warning, allowing the evaluation monitor to proceed with its own checks.
    # Consider: exit 1
fi

# Wait for the evaluation monitor job to complete
wait_for_child_job "$EVAL_MONITOR_JOB_ID_NUM" "Evaluation Monitor (Seed ${CURRENT_SEED_TO_RUN}, Run ${SHARED_RUN_ID})"
eval_wait_status=$? # Capture return status
if [ $eval_wait_status -ne 0 ]; then
    echo "CRITICAL ERROR (Main Orchestrator): Evaluation Monitor job ${EVAL_MONITOR_JOB_ID_NUM} did not complete successfully. Halting this orchestrator run."
    exit 1 # This non-zero exit will signal failure to the sweep_coordinator if applicable
fi

echo "=== Main Orchestrator Script Finished Successfully for Seed ${CURRENT_SEED_TO_RUN} (Run ID: ${SHARED_RUN_ID}): $(date) ==="
# A zero exit code here signals success to the sweep_coordinator.