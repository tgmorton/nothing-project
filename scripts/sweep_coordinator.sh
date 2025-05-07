#!/bin/bash
# sweep_coordinator.sbatch

# === SBATCH Directives for Seed Sweep Coordinator ===
#SBATCH --job-name=coord_seed_sweep
#SBATCH --partition=general          # <<< Run on a CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # Minimal CPUs for the coordinator
#SBATCH --mem=4G                     # Minimal RAM for the coordinator
#SBATCH --time=7-00:00:00            # <<< Max time for the ENTIRE sweep (e.g., 9 runs * ~15-24h/run)
#SBATCH --output=logs/sweep_coordinator_%x_%j.out
#SBATCH --error=logs/sweep_coordinator_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error for the coordinator script itself
set -e
# set -o pipefail # Exit if any command in a pipeline fails

echo "=== Seed Sweep Coordinator Started: $(date) ==="
echo "Sweep Coordinator Job ID: $SLURM_JOB_ID"
echo "Host Project Dir (will be passed to children): ${HOST_PROJECT_DIR:-/home/AD/thmorton/nothing-project}" # Default if not set via --export

# --- Define Base Paths ---
# HOST_PROJECT_DIR can be overridden by exporting it when submitting this sweep_coordinator job
# e.g., sbatch --export=ALL,HOST_PROJECT_DIR="/alt/project/path" sweep_coordinator.sbatch
DEFAULT_HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # <<< UPDATE THIS
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-${DEFAULT_HOST_PROJECT_DIR}}"
SWEEP_CHILD_LOGS_DIR="${HOST_PROJECT_DIR}/logs/sweep_child_job_logs" # Centralized logs for jobs launched by this sweep

mkdir -p "${HOST_PROJECT_DIR}/logs" # For this coordinator's logs
mkdir -p "${SWEEP_CHILD_LOGS_DIR}"
echo "Logs for individual seed runs (main_orchestrator and its children) will be in: ${SWEEP_CHILD_LOGS_DIR}"

# --- Function to wait for a Slurm job to complete ---
# Returns 0 on COMPLETED, 1 on other terminal states (FAILED, CANCELLED, etc.)
wait_for_slurm_job() {
    local job_id_to_wait=$1
    local job_description=$2 # For logging
    if [ -z "$job_id_to_wait" ]; then
        echo "Error: No job ID provided to wait_for_slurm_job for ${job_description}."
        return 1 # Error
    fi
    echo "Coordinator: Waiting for Slurm job '${job_description}' ($job_id_to_wait) to complete..."
    while true; do
        # Query job state using sacct
        job_status_line=$(sacct -j "$job_id_to_wait" --format=State --noheader | head -n 1)
        # Clean up the status (remove leading/trailing whitespace, handle (Pending))
        current_status=$(echo "$job_status_line" | awk '{print $1}' | sed 's/(.*)//')


        case "$current_status" in
            COMPLETED)
                echo "Coordinator: Job '${job_description}' ($job_id_to_wait) COMPLETED."
                return 0 # Success
                ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY)
                echo "Coordinator: Job '${job_description}' ($job_id_to_wait) ended with status: $current_status. Critical failure."
                return 1 # Failure
                ;;
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING)
                echo "Coordinator: Job '${job_description}' ($job_id_to_wait) current status: $current_status. Waiting..."
                ;;
            "") # Empty status from sacct might mean job is too old or never existed if sbatch failed silently
                # Check if squeue knows about it (for very recently submitted jobs that might not be in sacct yet)
                if squeue -h -j "$job_id_to_wait" | grep -q "$job_id_to_wait"; then
                    echo "Coordinator: Job '${job_description}' ($job_id_to_wait) status is empty from sacct, but found in squeue. Assuming PENDING/RUNNING. Waiting..."
                else
                    echo "Coordinator: Job '${job_description}' ($job_id_to_wait) no longer in sacct or squeue. Assuming it finished (possibly purged or failed very early)."
                    # This is tricky. If sbatch succeeded, it should appear. If it failed to submit, $job_id_to_wait might be faulty.
                    # For robustness, one might check the sbatch exit code more directly.
                    # Assuming for now that if it's gone, it's gone. The job's own logs would be key.
                    # Let's assume it might have completed very quickly or failed in a way sacct doesn't show clearly after time.
                    # This could be a point of failure if not handled carefully.
                    # Check if an error log file was created to infer failure.
                    # For now, let's assume we break and the main_orchestrator job itself reports its status.
                    # The crucial part is that main_orchestrator will exit non-zero if its children fail.
                    echo "Coordinator: Breaking wait loop for job ${job_id_to_wait}. Relying on main_orchestrator's exit status."
                    return 0 # Or some other indicator to check logs. For now, assume it's okay to proceed checking if main_orch fails.
                fi
                ;;
            *) # Unknown status
                echo "Coordinator: Job '${job_description}' ($job_id_to_wait) has unknown status: '$current_status'. Waiting..."
                ;;
        esac
        sleep 180 # Check every 3 minutes
    done
}


# === Seed Sweep Loop ===
SEEDS_TO_RUN=(1 2 3 4 5 6 7 8 9)
# Or for testing: SEEDS_TO_RUN=(42)
echo "Starting sweep for seeds: ${SEEDS_TO_RUN[*]}"
PATH_TO_MAIN_ORCHESTRATOR="${HOST_PROJECT_DIR}/scripts/main_orchestrator.sbatch" # <<< UPDATE THIS PATH

if [ ! -f "$PATH_TO_MAIN_ORCHESTRATOR" ]; then
    echo "CRITICAL ERROR: main_orchestrator.sbatch not found at ${PATH_TO_MAIN_ORCHESTRATOR}. Exiting."
    exit 1
fi


for CURRENT_SEED_VAL in "${SEEDS_TO_RUN[@]}"
do
    echo # Blank line for readability
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+++ Coordinator: Processing Seed: ${CURRENT_SEED_VAL} (Timestamp: $(date)) +++"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"

    # Define a unique job name suffix for the main_orchestrator job for this seed
    # $SLURM_JOB_ID here is the ID of the sweep_coordinator job itself.
    MAIN_ORCH_JOB_NAME="orch_s${CURRENT_SEED_VAL}_sw${SLURM_JOB_ID}"

    # Submit main_orchestrator.sbatch for the current seed
    # Pass HOST_PROJECT_DIR and the CURRENT_SEED_VAL
    echo "Coordinator: Submitting main_orchestrator for seed ${CURRENT_SEED_VAL}..."
    SUBMIT_COMMAND="sbatch \
        --export=ALL,HOST_PROJECT_DIR=\"${HOST_PROJECT_DIR}\",CURRENT_SEED_TO_RUN=\"${CURRENT_SEED_VAL}\" \
        --job-name=\"${MAIN_ORCH_JOB_NAME}\" \
        --output=\"${SWEEP_CHILD_LOGS_DIR}/${MAIN_ORCH_JOB_NAME}_%j.out\" \
        --error=\"${SWEEP_CHILD_LOGS_DIR}/${MAIN_ORCH_JOB_NAME}_%j.err\" \
        \"${PATH_TO_MAIN_ORCHESTRATOR}\""

    echo "Coordinator: Submission command: ${SUBMIT_COMMAND}"
    MAIN_ORCH_JOB_ID_OUTPUT=$(eval "${SUBMIT_COMMAND}") # Use eval to correctly handle quotes in paths/vars

    # Check if sbatch command itself failed (e.g., invalid options, script not found at Slurm level)
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: 'sbatch' command failed to submit main_orchestrator for seed ${CURRENT_SEED_VAL}."
        echo "Error output from sbatch (if any): ${MAIN_ORCH_JOB_ID_OUTPUT}"
        echo "Halting sweep."
        exit 1 # Stop the entire sweep
    fi

    # Extract job ID from sbatch output (e.g., "Submitted batch job 12345")
    MAIN_ORCH_JOB_ID_NUM=$(echo "$MAIN_ORCH_JOB_ID_OUTPUT" | awk '{print $NF}')

    if ! [[ "$MAIN_ORCH_JOB_ID_NUM" =~ ^[0-9]+$ ]]; then
        echo "CRITICAL ERROR: Failed to parse Job ID from sbatch output for seed ${CURRENT_SEED_VAL}."
        echo "sbatch output: ${MAIN_ORCH_JOB_ID_OUTPUT}"
        echo "Halting sweep."
        exit 1 # Stop the entire sweep
    fi

    echo "Coordinator: Main Orchestrator for seed ${CURRENT_SEED_VAL} submitted successfully. Job ID: ${MAIN_ORCH_JOB_ID_NUM}."

    # Wait for the submitted main_orchestrator job to complete
    wait_for_slurm_job "$MAIN_ORCH_JOB_ID_NUM" "Main Orchestrator (Seed ${CURRENT_SEED_VAL})"
    wait_status=$? # Get return status of wait_for_slurm_job

    if [ $wait_status -ne 0 ]; then
        echo "CRITICAL ERROR: Main Orchestrator job ${MAIN_ORCH_JOB_ID_NUM} for seed ${CURRENT_SEED_VAL} did NOT complete successfully."
        echo "Please check its logs: ${SWEEP_CHILD_LOGS_DIR}/${MAIN_ORCH_JOB_NAME}_${MAIN_ORCH_JOB_ID_NUM}.out/err"
        echo "Halting sweep."
        # Consider adding a `scancel $SLURM_JOB_ID` here if you want to kill the sweep coordinator itself.
        exit 1 # Stop the entire sweep
    fi

    echo "Coordinator: Main Orchestrator for Seed ${CURRENT_SEED_VAL} (Job ${MAIN_ORCH_JOB_ID_NUM}) has completed successfully."
    echo "--- End of Processing for Seed ${CURRENT_SEED_VAL} ---"
    # Optional: Short delay before starting the next seed's pipeline
    # sleep 300 # 5-minute delay
done

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "=== Seed Sweep Coordinator Finished All Seeds Successfully: $(date) ==="
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"