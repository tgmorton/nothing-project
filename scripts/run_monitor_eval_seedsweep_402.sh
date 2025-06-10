#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=gpt2_eval_monitor_seedsweep_SEQUENTIAL # Modified job name
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodelist=ssrde-c-402   # CRITICAL: Force the job onto a known-good node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=7-0:00:00
#SBATCH --output=../logs/%x_%j.out # Removed array-specific %a
#SBATCH --error=../logs/%x_%j.err  # Removed array-specific %a
# REMOVED: #SBATCH --array=1-10

# Exit on error
set -e

# === Environment Setup (Once per job) ===
echo "=== SEQUENTIAL JOB STARTED: $(date) ==="
echo "This job will process seeds 1 through 10 in order."
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "Username: $USER"

# --- Wait for and load modules (Once per job) ---
MODULE_INIT_SCRIPT="/etc/profile.d/modules.sh"
MODULE_CORE_FILE="/sscf/ssrde-storage/apps/modules/libexec/modulecmd.tcl"
WAIT_SECONDS=10
echo "Waiting up to ${WAIT_SECONDS}s for module system at ${MODULE_CORE_FILE} to be available..."
for ((i=0; i<WAIT_SECONDS; i++)); do
    if [ -r "${MODULE_CORE_FILE}" ]; then
        echo "Module system is ready."
        break
    fi
    sleep 1
done

if ! [ -r "${MODULE_CORE_FILE}" ]; then
    echo "FATAL: Module system core file was not readable after ${WAIT_SECONDS} seconds."
    exit 1
fi

source "${MODULE_INIT_SCRIPT}"
module load singularity/4.1.1 cuda/11.8

# --- Securely Load Neptune Credentials (Once per job) ---
NEPTUNE_CRED_FILE="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE" ]; then
    echo "Sourcing Neptune credentials from $NEPTUNE_CRED_FILE"
    source "$NEPTUNE_CRED_FILE"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    if [ -n "${NEPTUNE_PROJECT:-}" ]; then
      export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"
    fi
else
    echo "WARNING: Neptune credentials file not found at $NEPTUNE_CRED_FILE."
fi

# === Main Processing Loop ===
# This loop will run the evaluation for each seed, one after another.
for CURRENT_SEED in {1..10}; do
    echo "---"
    echo "--- Starting processing for Seed ${CURRENT_SEED} at $(date) ---"
    echo "---"

    # --- Define Paths on Host (Specific to the current seed) ---
    HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
    HOST_SIF_PATH="/home/AD/thmorton/nothing-project/python39_llm_env.sif"
    HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
    HOST_TRAINED_MODEL_PARENT_DIR="${HOST_PROJECT_DIR}/models/May20_seedsweep_ogLR_models/gpt2_p6000_sif_local_eval_run_May20_sweep_seed${CURRENT_SEED}"
    HOST_EVAL_MONITOR_OUTPUT_BASE_DIR="${HOST_TRAINED_MODEL_PARENT_DIR}/eval"
    HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"

    # --- Define Container Paths ---
    CONTAINER_WORKSPACE="/workspace"
    CONTAINER_DATA_DIR="/data"
    CONTAINER_TRAINED_MODEL_PARENT_DIR="/trained_model_checkpoints"
    CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR="/eval_monitor_outputs"
    CONTAINER_PRIMING_DIR="/eval"

    # --- Preparations (for the current seed) ---
    echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
    echo "Trained Model Parent Directory (Host): ${HOST_TRAINED_MODEL_PARENT_DIR}"
    echo "Evaluation Monitor Output Base Directory (Host): ${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}"

    if [ ! -d "$HOST_TRAINED_MODEL_PARENT_DIR" ]; then
        echo "WARNING: Trained model directory not found for seed ${CURRENT_SEED}, skipping: ${HOST_TRAINED_MODEL_PARENT_DIR}"
        continue # Skip to the next iteration of the loop
    fi

    mkdir -p "${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}"
    echo "Ensured host evaluation monitor output directory exists for seed ${CURRENT_SEED}"

    # === Evaluation Monitor Script Execution (for the current seed) ===
    echo "Starting Python evaluation_monitor.py script for seed ${CURRENT_SEED}..."

    # --- Define Neptune args ---
    NEPTUNE_PROJECT_ARG="--neptune_project thmorton/NothingProject"
    CONCEPTUAL_TRAINING_RUN_NAME="gpt2_p6000_sif_local_eval_run_May20_sweep_seed${CURRENT_SEED}"
    NEPTUNE_TAGS_FOR_MONITOR="p6000 eval_monitor singularity py39 sequential_run seed_${CURRENT_SEED}"

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    WATCH_MODE_ARG="" # Run once and exit

    # --- Execute Singularity Command ---
    singularity exec --nv \
        -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
        -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
        -B "${HOST_TRAINED_MODEL_PARENT_DIR}":"${CONTAINER_TRAINED_MODEL_PARENT_DIR}" \
        -B "${HOST_EVAL_MONITOR_OUTPUT_BASE_DIR}":"${CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR}" \
        -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
        "${HOST_SIF_PATH}" \
        python3 "${CONTAINER_WORKSPACE}/src/evaluation_monitor.py" \
            --model_parent_dir "${CONTAINER_TRAINED_MODEL_PARENT_DIR}" \
            --output_base_dir "${CONTAINER_EVAL_MONITOR_OUTPUT_BASE_DIR}" \
            ${WATCH_MODE_ARG} \
            \
            --run_priming_eval \
            --priming_eval_dir_path "/eval/just_shota" \
            \
            --base_model_name "gpt2" \
            --model_class_name "GPT2LMHeadModel" \
            --per_device_eval_batch_size 8 \
            --priming_per_device_eval_batch_size 8 \
            --eval_max_samples 5000 \
            --priming_eval_max_samples_per_file 1000 \
            \
            --use_amp \
            --num_workers ${SLURM_CPUS_PER_TASK:-4} \
            --seed ${CURRENT_SEED} \
            --no_skip_processed_checkpoints \
            \
            ${NEPTUNE_PROJECT_ARG} \
            --neptune_tags ${NEPTUNE_TAGS_FOR_MONITOR} \
            --neptune_training_run_name "${CONCEPTUAL_TRAINING_RUN_NAME}" \
            --run_standard_eval \
            --validation_dataset_path "/data/processed/test_set_10m"

    echo "--- Finished processing for Seed ${CURRENT_SEED} at $(date) ---"
done

# === Job Completion ===
echo "=== SEQUENTIAL JOB FINISHED: $(date) ==="
