#!/bin/bash

# === SBATCH Directives (Defaults - can be overridden by sbatch command) ===
#SBATCH --job-name=eval_job            # Default job name (will be overridden)
#SBATCH --partition=general_gpu_p6000  # Target queue (MODIFY IF NEEDED)
#SBATCH --nodes=1                      # Request one node
#SBATCH --ntasks-per-node=1            # Run one task (the python script)
#SBATCH --cpus-per-task=4              # Request CPUs (MODIFY IF NEEDED for eval)
#SBATCH --mem=24G                      # Request RAM (MODIFY IF NEEDED for eval)
#SBATCH --gres=gpu:1                   # Request 1 GPU (MODIFY IF NEEDED for eval)
#SBATCH --time=02:00:00                # Time limit (HH:MM:SS) - e.g., 2 hours for eval
#SBATCH --output=slurm_eval_%j.out     # Default output (overridden by sbatch cmd using EVAL_OUT_DIR)
#SBATCH --error=slurm_eval_%j.err      # Default error (overridden by sbatch cmd using EVAL_OUT_DIR)

# Exit on error
set -e

# === Environment Setup ===
echo "=== Evaluation Job Started: $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Read Environment Variables Passed by sbatch --export ---
# These variables are set by the modified train.py script when submitting
echo "--- Reading Environment Variables ---"
echo "Checkpoint Path (CKPT_PATH): '${CKPT_PATH}'"
echo "Evaluation Output Dir (EVAL_OUT_DIR): '${EVAL_OUT_DIR}'"
echo "Run Standard Eval (RUN_STD_EVAL): '${RUN_STD_EVAL}'"
echo "Run Priming Eval (RUN_PRIME_EVAL): '${RUN_PRIME_EVAL}'"
echo "Validation Data Path (VALID_DATA_PATH): '${VALID_DATA_PATH:-Not Set}'" # Use :- for optional vars
echo "Priming Data Path (PRIME_DATA_PATH): '${PRIME_DATA_PATH:-Not Set}'"
echo "Seed (SEED): '${SEED:-Not Set}'"
echo "Neptune Project (NEPTUNE_PROJECT): '${NEPTUNE_PROJECT:-Not Set}'"
echo "Neptune Run ID (NEPTUNE_RUN_ID): '${NEPTUNE_RUN_ID:-Not Set}'"
# NEPTUNE_API_TOKEN is handled via SINGULARITYENV_ below

# --- Validate Required Variables ---
if [ -z "$CKPT_PATH" ]; then echo "ERROR: CKPT_PATH environment variable not set."; exit 1; fi
if [ -z "$EVAL_OUT_DIR" ]; then echo "ERROR: EVAL_OUT_DIR environment variable not set."; exit 1; fi
# Add checks for other required variables based on RUN_STD_EVAL/RUN_PRIME_EVAL if desired

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Handle Neptune API Token for Singularity ---
# Assumes NEPTUNE_API_TOKEN might be in the environment exported via --export=ALL
if [ -n "${NEPTUNE_API_TOKEN:-}" ]; then
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN}"
    echo "Exported NEPTUNE_API_TOKEN to SINGULARITYENV_"
else
    echo "WARNING: NEPTUNE_API_TOKEN not found in environment. Neptune logging in container might fail if not set otherwise."
fi
# Export project/run ID if set, evaluate.py can use these
if [ -n "${NEPTUNE_PROJECT:-}" ]; then export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT}"; fi
if [ -n "${NEPTUNE_RUN_ID:-}" ]; then export SINGULARITYENV_NEPTUNE_RUN_ID="${NEPTUNE_RUN_ID}"; fi


# --- Define Paths on Host (SSRDE Cluster) ---
# These should match the paths used in your training script for consistency
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # MODIFY IF NEEDED
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif" # MODIFY IF NEEDED
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"          # MODIFY IF NEEDED
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval"     # MODIFY IF NEEDED
# NOTE: HOST_OUTPUT_BASE_DIR is implicitly handled by the EVAL_OUT_DIR variable now

# Define corresponding paths *inside* the container (MUST MATCH BIND MOUNTS)
CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/output_eval" # <<< Use a distinct name for the eval output mount point
CONTAINER_PRIMING_DIR="/eval_data"  # <<< Use a distinct name for the priming data mount point

echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Data Base Directory (Host): ${HOST_DATA_BASE_DIR}"
echo "Priming Base Directory (Host): ${HOST_PRIMING_BASE_DIR}"

# Verify Singularity image exists
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi

# Ensure the evaluation output directory exists (it should have been created by train.py)
if [ ! -d "$EVAL_OUT_DIR" ]; then echo "WARNING: Evaluation output directory '$EVAL_OUT_DIR' not found. Attempting to create."; mkdir -p "$EVAL_OUT_DIR"; fi


# === Evaluate Script Execution (Inside Container) ===
echo "Starting Python evaluation script inside Singularity container..."

# --- Construct arguments for evaluate.py ---
PYTHON_ARGS=()
PYTHON_ARGS+=( "--checkpoint_path" "${CONTAINER_WORKSPACE}/${CKPT_PATH#${HOST_PROJECT_DIR}/}" ) # Map host path to container path
PYTHON_ARGS+=( "--output_dir" "${CONTAINER_OUTPUT_DIR}" ) # Eval script writes inside this mounted dir

# Add flags/paths conditionally based on environment variables
if [ "$RUN_STD_EVAL" = "1" ]; then
    PYTHON_ARGS+=( "--run_standard_eval" )
    if [ -n "$VALID_DATA_PATH" ]; then
        # Map validation data path relative to the data mount point
        CONTAINER_VALID_PATH="${CONTAINER_DATA_DIR}/${VALID_DATA_PATH#${HOST_DATA_BASE_DIR}/}"
        PYTHON_ARGS+=( "--validation_dataset_path" "$CONTAINER_VALID_PATH" )
    else
        echo "WARNING: RUN_STD_EVAL is 1, but VALID_DATA_PATH is not set."
    fi
fi

if [ "$RUN_PRIME_EVAL" = "1" ]; then
    PYTHON_ARGS+=( "--run_priming_eval" )
    if [ -n "$PRIME_DATA_PATH" ]; then
        # Map priming data path relative to the priming mount point
        CONTAINER_PRIME_PATH="${CONTAINER_PRIMING_DIR}/${PRIME_DATA_PATH#${HOST_PRIMING_BASE_DIR}/}"
        PYTHON_ARGS+=( "--priming_eval_dir_path" "$CONTAINER_PRIME_PATH" )
    else
        echo "WARNING: RUN_PRIME_EVAL is 1, but PRIME_DATA_PATH is not set."
    fi
fi

# Add other arguments (seed, batch sizes, etc. - these could be hardcoded or passed via --export too)
if [ -n "$SEED" ]; then PYTHON_ARGS+=( "--seed" "$SEED" ); fi
PYTHON_ARGS+=( "--per_device_eval_batch_size" "32" ) # Example: Hardcode or pass via export
PYTHON_ARGS+=( "--priming_per_device_eval_batch_size" "16" ) # Example: Hardcode or pass via export
PYTHON_ARGS+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" )
PYTHON_ARGS+=( "--use_amp" ) # Example: Assume AMP is usually desired if available

# Add Neptune args if project is set (Run ID is passed via env var automatically if exported)
if [ -n "${NEPTUNE_PROJECT:-}" ]; then PYTHON_ARGS+=( "--neptune_project" "${NEPTUNE_PROJECT}" ); fi
# evaluate.py will pick up SINGULARITYENV_NEPTUNE_RUN_ID if set

echo "Arguments for evaluate.py:"
printf '%q ' "${PYTHON_ARGS[@]}" # Print quoted arguments for clarity
echo # Newline

# --- Execute Singularity ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Keep if needed

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
    -B "${EVAL_OUT_DIR}":"${CONTAINER_OUTPUT_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/src/evaluate.py" "${PYTHON_ARGS[@]}"


# === Job Completion ===
echo "--- Evaluation Job Finished: $(date) ---"