#!/bin/bash
# main_orchestrator.sh

# --- Source Project Configuration ---
CONFIG_FILE="../project_config.sh" # Assuming config is in parent directory
if [ -f "$CONFIG_FILE" ]; then
    echo "Sourcing project configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "CRITICAL ERROR: Project configuration file not found at $CONFIG_FILE"
    exit 1
fi
# --- End Source Project Configuration ---

# Slurm directives will now ideally be set by the user submitting this script,
# or they can have defaults from the config file if not overridden.
# For this example, we'll assume some critical ones can be defaulted from config.
#SBATCH --job-name=${MAIN_ORCH_JOB_NAME:-main_orch_standalone}
#SBATCH --partition=${MAIN_ORCH_PARTITION:-general}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${MAIN_ORCH_CPUS:-1}
#SBATCH --mem=${MAIN_ORCH_MEM:-4G}
#SBATCH --time=${MAIN_ORCH_TIME:-7-0:00:00}
#SBATCH --output=${MAIN_ORCH_LOG_DIR:-../logs}/main_orchestrator_%x_%j.out
#SBATCH --error=${MAIN_ORCH_LOG_DIR:-../logs}/main_orchestrator_%x_%j.err
#SBATCH --mail-type=${MAIN_ORCH_MAIL_TYPE:-END,FAIL}
#SBATCH --mail-user=${MAIN_ORCH_MAIL_USER:-your_email@example.com}

set -e
# set -o pipefail

echo "=== Main Orchestrator Script Started: $(date) ==="
echo "Main Orchestrator Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Using Project Config: $CONFIG_FILE"

# HOST_PROJECT_DIR should be defined in project_config.sh
if [ -z "$HOST_PROJECT_DIR" ]; then
    echo "CRITICAL ERROR: HOST_PROJECT_DIR not set in $CONFIG_FILE"; exit 1;
fi
echo "Host Project Dir (from config): ${HOST_PROJECT_DIR}"
if [ ! -d "$HOST_PROJECT_DIR" ]; then echo "CRITICAL ERROR: HOST_PROJECT_DIR does not exist: ${HOST_PROJECT_DIR}"; exit 1; fi

# SEED_FOR_STANDALONE can be overridden by CURRENT_SEED_TO_RUN from environment (e.g. sweep script)
# SEED_FOR_STANDALONE is the default from project_config.sh
effective_seed="${CURRENT_SEED_TO_RUN:-${SEED_FOR_STANDALONE:-42}}" # Default to 42 if nothing is set
echo "Effective Seed for this Run: ${effective_seed}"
if ! [[ "$effective_seed" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Effective seed is not a number: '${effective_seed}'"; exit 1; fi


# OUTPUT_BASE_DIR should be defined in project_config.sh
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/${OUTPUT_BASE_DIR_RELATIVE_TO_PROJECT_ROOT:-src/.output}"
CHILD_JOB_LOGS_BASE_DIR="${HOST_PROJECT_DIR}/${CHILD_JOB_LOGS_DIR_RELATIVE_TO_PROJECT_ROOT:-logs/main_orch_children}"
CHILD_JOB_LOGS_DIR="${CHILD_JOB_LOGS_BASE_DIR}/s${effective_seed}_job${SLURM_JOB_ID}"

JOB_ID_FOR_RUN_NAME="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
SHARED_RUN_ID="s${effective_seed}_j${JOB_ID_FOR_RUN_NAME}_ts$(date +%Y%m%d_%H%M%S)"
SHARED_OUTPUT_DIR_HOST="${HOST_OUTPUT_BASE_DIR}/${SHARED_RUN_ID}"

# CHECKPOINT_READY_SENTINEL_FILENAME should be defined in project_config.sh
if [ -z "$CHECKPOINT_READY_SENTINEL_FILENAME" ]; then
    echo "CRITICAL ERROR: CHECKPOINT_READY_SENTINEL_FILENAME not set in $CONFIG_FILE"; exit 1;
fi

echo "Shared Run ID for Seed ${effective_seed}: ${SHARED_RUN_ID}"
echo "Shared Output Directory (Host) for this run: ${SHARED_OUTPUT_DIR_HOST}"
echo "Checkpoint Ready Sentinel Filename (from config): ${CHECKPOINT_READY_SENTINEL_FILENAME}"
mkdir -p "${SHARED_OUTPUT_DIR_HOST}"
mkdir -p "${CHILD_JOB_LOGS_DIR}"
echo "Ensured shared output and child log directories exist."

# Paths to child scripts from config
PATH_TO_TRAINING_JOB_SCRIPT="${HOST_PROJECT_DIR}/${TRAINING_JOB_SCRIPT_RELATIVE_PATH:-scripts/training_job.sh}"
PATH_TO_EVAL_ORCHESTRATOR_JOB_SCRIPT="${HOST_PROJECT_DIR}/${EVAL_ORCH_SCRIPT_RELATIVE_PATH:-scripts/eval_orchestrator.sh}"

if [ ! -f "$PATH_TO_TRAINING_JOB_SCRIPT" ]; then echo "CRITICAL ERROR: training_job.sh not found at configured path: ${PATH_TO_TRAINING_JOB_SCRIPT}"; exit 1; fi
if [ ! -f "$PATH_TO_EVAL_ORCHESTRATOR_JOB_SCRIPT" ]; then echo "CRITICAL ERROR: eval_orchestrator.sh not found at configured path: ${PATH_TO_EVAL_ORCHESTRATOR_JOB_SCRIPT}"; exit 1; fi

echo "Main Orchestrator: Submitting Training Job for Seed ${effective_seed}..."
TRAIN_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_TRAINING="${effective_seed}",CHECKPOINT_READY_SENTINEL_FILENAME="${CHECKPOINT_READY_SENTINEL_FILENAME}",CONFIG_FILE_PATH_ABS="$(readlink -f $CONFIG_FILE)" \
    --job-name="train_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_TRAINING_JOB_SCRIPT}")
# Note: Passing CONFIG_FILE_PATH_ABS so child jobs can source the same config

if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Training Job. Output: ${TRAIN_JOB_ID_OUTPUT}"; exit 1; fi
TRAIN_JOB_ID_NUM=$(echo "$TRAIN_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$TRAIN_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Training Job ID: '${TRAIN_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Training Job submitted. ID: ${TRAIN_JOB_ID_NUM}."

echo "Main Orchestrator: Submitting Evaluation Orchestrator Job for Seed ${effective_seed}..."
EVAL_ORCH_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_EVAL="${effective_seed}",CHECKPOINT_READY_SENTINEL_FILENAME="${CHECKPOINT_READY_SENTINEL_FILENAME}",CONFIG_FILE_PATH_ABS="$(readlink -f $CONFIG_FILE)" \
    --job-name="eval_orch_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/eval_orchestrator_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/eval_orchestrator_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_EVAL_ORCHESTRATOR_JOB_SCRIPT}")

if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Evaluation Orchestrator. Output: ${EVAL_ORCH_JOB_ID_OUTPUT}"; exit 1; fi
EVAL_ORCH_JOB_ID_NUM=$(echo "$EVAL_ORCH_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$EVAL_ORCH_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Eval Orchestrator Job ID: '${EVAL_ORCH_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Evaluation Orchestrator Job submitted. ID: ${EVAL_ORCH_JOB_ID_NUM}."

wait_for_child_job() {
    local job_id=$1
    local job_name_desc=$2
    echo "Main Orchestrator: Waiting for child job '${job_name_desc}' ($job_id)..."
    local wait_interval=${JOB_WAIT_INTERVAL_SECONDS:-120} # Default 120s, from config
    while true; do
        status_line=$(sacct -j "$job_id" --format=State --noheader | head -n 1)
        current_child_status=$(echo "$status_line" | awk '{print $1}' | sed 's/(.*)//')
        case "$current_child_status" in
            COMPLETED) echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) COMPLETED."; return 0 ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY) echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) status: $current_child_status. Orchestrator failing."; return 1 ;;
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING) echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) status: $current_child_status. Waiting..." ;;
            "") if squeue -h -j "$job_id" | grep -q "$job_id"; then echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) status empty (sacct), in squeue. Assuming PENDING/RUNNING..."; else echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) no longer in sacct/squeue. Assuming finished."; return 0; fi ;;
            *) echo "Main Orchestrator: Child job '${job_name_desc}' ($job_id) unknown status: '$current_child_status'. Waiting..." ;;
        esac
        sleep "${wait_interval}"
    done
}

wait_for_child_job "$TRAIN_JOB_ID_NUM" "Training Job (Seed ${effective_seed})"
train_wait_status=$?
if [ $train_wait_status -ne 0 ]; then echo "CRITICAL ERROR (Main Orchestrator): Training job failed."; exit 1; fi

TRAINING_COMPLETED_SENTINEL_FILE="${SHARED_OUTPUT_DIR_HOST}/${TRAINING_COMPLETION_SENTINEL_FILENAME:-TRAINING_COMPLETED.txt}" # From config
if [ ! -f "${TRAINING_COMPLETED_SENTINEL_FILE}" ]; then
    echo "CRITICAL WARNING (Main Orchestrator): Training job ${TRAIN_JOB_ID_NUM} completed, BUT training sentinel '${TRAINING_COMPLETED_SENTINEL_FILE}' NOT FOUND."
fi

wait_for_child_job "$EVAL_ORCH_JOB_ID_NUM" "Evaluation Orchestrator Submission (Seed ${effective_seed})"
eval_orch_wait_status=$?
if [ $eval_orch_wait_status -ne 0 ]; then echo "CRITICAL ERROR (Main Orchestrator): Evaluation Orchestrator submission job failed."; exit 1; fi

echo "Main Orchestrator: Training and Evaluation Orchestrator jobs have completed their submission phases."
echo "NOTE: The actual evaluation is a long-running job managed by evaluate.py's watch mode."
echo "Monitor the logs of the Slurm job submitted by eval_orchestrator.sh (likely named 'multi_eval_watch_...')."
echo "=== Main Orchestrator Script Finished Successfully for Seed ${effective_seed} (Run ID: ${SHARED_RUN_ID}): $(date) ==="