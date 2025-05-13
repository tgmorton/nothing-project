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
#SBATCH --time=7-0:00:00        # Max time for ONE full pipeline (train + eval orchestrator)
#SBATCH --output=../logs/main_orchestrator_%x_%j.out # Default for standalone
#SBATCH --error=../logs/main_orchestrator_%x_%j.err  # Default for standalone
#SBATCH --mail-type=END,FAIL      # Notify on end/fail of this orchestrator job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

set -e
# set -o pipefail

echo "=== Main Orchestrator Script Started: $(date) ==="
echo "Main Orchestrator Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"

DEFAULT_HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # <<< UPDATE THIS DEFAULT
if [ -n "$HOST_PROJECT_DIR" ]; then
    echo "Using HOST_PROJECT_DIR from environment: ${HOST_PROJECT_DIR}"
else
    HOST_PROJECT_DIR="${DEFAULT_HOST_PROJECT_DIR}"
    echo "HOST_PROJECT_DIR not set, using default: ${HOST_PROJECT_DIR}"
fi
if [ ! -d "$HOST_PROJECT_DIR" ]; then echo "CRITICAL ERROR: HOST_PROJECT_DIR does not exist: ${HOST_PROJECT_DIR}"; exit 1; fi

DEFAULT_SEED_FOR_STANDALONE="43"
if [ -n "$CURRENT_SEED_TO_RUN" ]; then
    echo "Using CURRENT_SEED_TO_RUN from environment: ${CURRENT_SEED_TO_RUN}"
else
    CURRENT_SEED_TO_RUN="${DEFAULT_SEED_FOR_STANDALONE}"
    echo "CURRENT_SEED_TO_RUN not set, using default: ${CURRENT_SEED_TO_RUN}"
fi
if ! [[ "$CURRENT_SEED_TO_RUN" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: CURRENT_SEED_TO_RUN is not a number: '${CURRENT_SEED_TO_RUN}'"; exit 1; fi

HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
CHILD_JOB_LOGS_DIR="${HOST_PROJECT_DIR}/logs/main_orch_children/s${CURRENT_SEED_TO_RUN}_job${SLURM_JOB_ID}"
JOB_ID_FOR_RUN_NAME="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
SHARED_RUN_ID="s${CURRENT_SEED_TO_RUN}_j${JOB_ID_FOR_RUN_NAME}_ts$(date +%Y%m%d_%H%M%S)"
SHARED_OUTPUT_DIR_HOST="${HOST_OUTPUT_BASE_DIR}/${SHARED_RUN_ID}"

echo "Shared Run ID for Seed ${CURRENT_SEED_TO_RUN}: ${SHARED_RUN_ID}"
echo "Shared Output Directory (Host) for this run: ${SHARED_OUTPUT_DIR_HOST}"
mkdir -p "${SHARED_OUTPUT_DIR_HOST}"
mkdir -p "${CHILD_JOB_LOGS_DIR}"
echo "Ensured shared output and child log directories exist."

PATH_TO_TRAINING_JOB="${HOST_PROJECT_DIR}/scripts/training_job.sh"
PATH_TO_EVAL_ORCHESTRATOR_JOB="${HOST_PROJECT_DIR}/scripts/eval_orchestrator.sh" # <<< MODIFIED

if [ ! -f "$PATH_TO_TRAINING_JOB" ]; then echo "CRITICAL ERROR: training_job.sh not found: ${PATH_TO_TRAINING_JOB}"; exit 1; fi
if [ ! -f "$PATH_TO_EVAL_ORCHESTRATOR_JOB" ]; then echo "CRITICAL ERROR: eval_orchestrator.sh not found: ${PATH_TO_EVAL_ORCHESTRATOR_JOB}"; exit 1; fi

echo "Main Orchestrator: Submitting Training Job for Seed ${CURRENT_SEED_TO_RUN}..."
TRAIN_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_TRAINING="${CURRENT_SEED_TO_RUN}" \
    --job-name="train_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/training_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_TRAINING_JOB}")
if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Training Job. Output: ${TRAIN_JOB_ID_OUTPUT}"; exit 1; fi
TRAIN_JOB_ID_NUM=$(echo "$TRAIN_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$TRAIN_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Training Job ID: '${TRAIN_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Training Job submitted. ID: ${TRAIN_JOB_ID_NUM}."

echo "Main Orchestrator: Submitting Evaluation Orchestrator Job for Seed ${CURRENT_SEED_TO_RUN}..." # <<< MODIFIED
EVAL_ORCH_JOB_ID_OUTPUT=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR_HOST="${SHARED_OUTPUT_DIR_HOST}",SHARED_RUN_ID="${SHARED_RUN_ID}",SEED_FOR_EVAL="${CURRENT_SEED_TO_RUN}" \
    --job-name="eval_orch_${SHARED_RUN_ID}" \
    --output="${CHILD_JOB_LOGS_DIR}/eval_orchestrator_${SHARED_RUN_ID}_%j.out" \
    --error="${CHILD_JOB_LOGS_DIR}/eval_orchestrator_${SHARED_RUN_ID}_%j.err" \
    "${PATH_TO_EVAL_ORCHESTRATOR_JOB}") # <<< MODIFIED
if [ $? -ne 0 ]; then echo "CRITICAL ERROR: sbatch failed for Evaluation Orchestrator. Output: ${EVAL_ORCH_JOB_ID_OUTPUT}"; exit 1; fi
EVAL_ORCH_JOB_ID_NUM=$(echo "$EVAL_ORCH_JOB_ID_OUTPUT" | awk '{print $NF}')
if ! [[ "$EVAL_ORCH_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then echo "CRITICAL ERROR: Failed to parse Eval Orchestrator Job ID: '${EVAL_ORCH_JOB_ID_OUTPUT}'"; exit 1; fi
echo "Main Orchestrator: Evaluation Orchestrator Job submitted. ID: ${EVAL_ORCH_JOB_ID_NUM}."

wait_for_child_job() {
    local job_id=$1
    local job_name_desc=$2
    echo "Main Orchestrator: Waiting for child job '${job_name_desc}' ($job_id)..."
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
        sleep 120
    done
}

wait_for_child_job "$TRAIN_JOB_ID_NUM" "Training Job (Seed ${CURRENT_SEED_TO_RUN})"
train_wait_status=$?
if [ $train_wait_status -ne 0 ]; then echo "CRITICAL ERROR (Main Orchestrator): Training job failed."; exit 1; fi

TRAINING_SENTINEL_FILE="${SHARED_OUTPUT_DIR_HOST}/TRAINING_COMPLETED.txt"
if [ ! -f "${TRAINING_SENTINEL_FILE}" ]; then
    echo "CRITICAL WARNING (Main Orchestrator): Training job ${TRAIN_JOB_ID_NUM} completed, BUT sentinel '${TRAINING_SENTINEL_FILE}' NOT FOUND."
fi

wait_for_child_job "$EVAL_ORCH_JOB_ID_NUM" "Evaluation Orchestrator (Seed ${CURRENT_SEED_TO_RUN})" # <<< MODIFIED
eval_orch_wait_status=$?
if [ $eval_orch_wait_status -ne 0 ]; then echo "CRITICAL ERROR (Main Orchestrator): Evaluation Orchestrator job failed."; exit 1; fi

echo "=== Main Orchestrator Script Finished Successfully for Seed ${CURRENT_SEED_TO_RUN} (Run ID: ${SHARED_RUN_ID}): $(date) ==="