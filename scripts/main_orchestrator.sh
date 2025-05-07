#!/bin/bash
#SBATCH --job-name=main_orchestrator
#SBATCH --partition=general       # CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=25:00:00           # Should be longer than max expected run time
#SBATCH --output=logs/orchestrator_%x_%A_%a.out
#SBATCH --error=logs/orchestrator_%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error
set -e

echo "=== Main Orchestrator Started: $(date) ==="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"

# --- Define Shared Base Paths ---
# These should be absolute paths
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # <<< UPDATE THIS
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
LOGS_DIR="${HOST_PROJECT_DIR}/logs" # Centralized logs

# --- Create a Unique Run ID and Shared Output Directory ---
# This ensures outputs from different orchestrated runs don't clash
SHARED_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
SHARED_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${SHARED_RUN_ID}"

echo "Shared Run ID: ${SHARED_RUN_ID}"
echo "Shared Output Directory (Host): ${SHARED_OUTPUT_DIR}"

# --- Create Directories ---
mkdir -p "${SHARED_OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}" # For orchestrator and child job logs
echo "Ensured shared output directory and log directory exist."

# --- Launch Training Job ---
echo "Submitting Training Job..."
# Pass necessary variables to the training job.
# SLURM_JOB_ACCOUNT and SLURM_JOB_PARTITION might be useful if you need to pass them.
TRAIN_JOB_ID=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}",SHARED_RUN_ID="${SHARED_RUN_ID}" \
    --job-name="train_${SHARED_RUN_ID}" \
    --output="${LOGS_DIR}/training_${SHARED_RUN_ID}_%j.out" \
    --error="${LOGS_DIR}/training_${SHARED_RUN_ID}_%j.err" \
    "${HOST_PROJECT_DIR}/scripts/training_job.sbatch") # <<< UPDATE PATH TO training_job.sbatch

TRAIN_JOB_ID_NUM=$(echo $TRAIN_JOB_ID | awk '{print $NF}') # Extract job ID number
echo "Training Job submitted with ID: ${TRAIN_JOB_ID_NUM}"

# --- Launch Evaluation Monitor Job ---
echo "Submitting Evaluation Monitor Job..."
# Pass necessary variables to the evaluation monitor job.
EVAL_MONITOR_JOB_ID=$(sbatch \
    --export=ALL,HOST_PROJECT_DIR="${HOST_PROJECT_DIR}",SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}",SHARED_RUN_ID="${SHARED_RUN_ID}" \
    --job-name="eval_mon_${SHARED_RUN_ID}" \
    --output="${LOGS_DIR}/eval_monitor_${SHARED_RUN_ID}_%j.out" \
    --error="${LOGS_DIR}/eval_monitor_${SHARED_RUN_ID}_%j.err" \
    "${HOST_PROJECT_DIR}/scripts/evaluation_monitor.sbatch") # <<< UPDATE PATH TO evaluation_monitor.sbatch

EVAL_MONITOR_JOB_ID_NUM=$(echo $EVAL_MONITOR_JOB_ID | awk '{print $NF}') # Extract job ID number
echo "Evaluation Monitor Job submitted with ID: ${EVAL_MONITOR_JOB_ID_NUM}"

# --- Optional: Wait for both jobs to complete ---
# This orchestrator script can exit after submitting, or it can wait.
# If it waits, it keeps the orchestrator job active in Slurm until children finish.
echo "Orchestrator has launched child jobs. Waiting for them to complete..."
echo "You can monitor child jobs with squeue or check their log files in ${LOGS_DIR}"

# Simple wait loop
wait_for_job() {
    local job_id_to_wait=$1
    local job_name_to_wait=$2
    echo "Waiting for Slurm job ${job_name_to_wait} ($job_id_to_wait) to complete..."
    # Check `sacct` for job state. Alternatives include `squeue`.
    while true; do
        job_status=$(sacct -j $job_id_to_wait --format=State --noheader | head -n 1 | awk '{$1=$1};1')
        if [[ "$job_status" == "COMPLETED" || \
              "$job_status" == "FAILED" || \
              "$job_status" == "CANCELLED" || \
              "$job_status" == "TIMEOUT" || \
              "$job_status" == "" # Job no longer in recent history
            ]]; then
            echo "Slurm job ${job_name_to_wait} ($job_id_to_wait) finished with status: ${job_status:-'Unknown or Purged'}"
            break
        fi
        sleep 120 # Check every 2 minutes
    done
}

# Uncomment to make orchestrator wait:
# wait_for_job "$TRAIN_JOB_ID_NUM" "Training"
# wait_for_job "$EVAL_MONITOR_JOB_ID_NUM" "Evaluation Monitor"

echo "=== Main Orchestrator Finished: $(date) ==="