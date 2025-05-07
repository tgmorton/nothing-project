#!/bin/bash
# eval_job.sbatch
# This script is submitted by eval_orchestrator.sbatch to run evaluate.py for a single checkpoint.
# Job name, output/error paths are set by eval_orchestrator.sbatch.
#SBATCH --job-name=single_eval_job # Will be overridden
#SBATCH --partition=general_gpu_p6000    # <<< Target GPU partition for actual evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                # CPUs for evaluate.py
#SBATCH --mem=32G                        # RAM for evaluate.py
#SBATCH --time=03:00:00                  # Max time for a single evaluation run (adjust as needed)
#SBATCH --mail-type=FAIL                 # Notify only on failure of an individual eval job
#SBATCH --mail-user=your_email@example.com # <<< UPDATE THIS

# Exit on error for this specific evaluation job
set -e
# set -o pipefail

echo "=== Single Evaluation Job Started: $(date) ==="
echo "Evaluation Job ID: $SLURM_JOB_ID (Job Name: $SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# --- Echo variables received from eval_orchestrator.sbatch via --export ---
echo "Host Project Dir: ${HOST_PROJECT_DIR}"
echo "Shared Output Dir (Host, for checkpoint source): ${SHARED_OUTPUT_DIR_HOST}" # Base where all checkpoints are
echo "Shared Run ID (for context/tagging): ${SHARED_RUN_ID}"
echo "Seed for this Evaluation: ${SEED_FOR_THIS_EVAL}"
echo "Checkpoint Directory Name to Evaluate: ${CHECKPOINT_DIR_NAME_TO_EVAL}"
echo "Host Path for THIS Eval Job's Results: ${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}" # Where evaluate.py should output its JSON/CSVs

# --- Validate received variables ---
if [ -z "$HOST_PROJECT_DIR" ] || \
   [ -z "$SHARED_OUTPUT_DIR_HOST" ] || \
   [ -z "$SHARED_RUN_ID" ] || \
   [ -z "$SEED_FOR_THIS_EVAL" ] || \
   [ -z "$CHECKPOINT_DIR_NAME_TO_EVAL" ] || \
   [ -z "$HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS" ]; then
    echo "CRITICAL ERROR (Single Eval Job): One or more essential environment variables not set by eval_orchestrator!"
    exit 1
fi

# --- Load system modules ---
echo "Loading system modules: singularity, cuda..."
module load singularity/4.1.1 cuda/11.8 # <<< MODIFY versions if needed

# --- Neptune Credentials for Singularity ---
# These should already be set by eval_orchestrator globally for SINGULARITYENV_
# or we can re-source here if preferred for explicitness, though it might be redundant.
NEPTUNE_CRED_FILE_EVAL_JOB="$HOME/.neptune_creds"
if [ -f "$NEPTUNE_CRED_FILE_EVAL_JOB" ]; then
    source "$NEPTUNE_CRED_FILE_EVAL_JOB"
    export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
    export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-thmorton/NothingProject}" # <<< UPDATE
else
    echo "WARNING (Single Eval Job): Neptune creds file not found. evaluate.py relies on pre-exported SINGULARITYENV_ vars or direct args."
    # Ensure a fallback project if evaluate.py relies on this being set somehow beyond args
     export SINGULARITYENV_NEPTUNE_PROJECT="${SINGULARITYENV_NEPTUNE_PROJECT:-thmorton/NothingProject}" # <<< UPDATE
fi
# Also ensure NEPTUNE_TRAINING_RUN_NAME is available for evaluate.py if it uses it
export SINGULARITYENV_NEPTUNE_TRAINING_RUN_NAME="train_${SHARED_RUN_ID}"


# --- Define Paths on Host ---
HOST_SIF_PATH_EVAL_JOB="${HOST_PROJECT_DIR}/python39_llm_env.sif" # <<< UPDATE SIF name
HOST_DATA_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/data"
HOST_PRIMING_BASE_DIR_EVAL_JOB="${HOST_PROJECT_DIR}/eval"

# --- Define Container Mount Points ---
CONTAINER_WORKSPACE_EVAL_JOB="/workspace"
CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB="/data_eval_mnt"
CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB="/priming_data_eval_mnt"
# Mount point for SHARED_OUTPUT_DIR_HOST (where all checkpoints reside)
CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT="/mnt_all_checkpoints"
# Mount point for HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS (where this eval job writes its specific results)
CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET="/eval_job_output_target"


# --- Preparations ---
echo "SIF Image Path (Host): ${HOST_SIF_PATH_EVAL_JOB}"
if [ ! -f "$HOST_SIF_PATH_EVAL_JOB" ]; then echo "CRITICAL ERROR: Singularity image not found: ${HOST_SIF_PATH_EVAL_JOB}"; exit 1; fi
# The specific output directory for this job's results is HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS.
# evaluate.py will create it if its --output_dir argument points to the container equivalent.
# mkdir -p "$HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS" # evaluate.py or its arg parser should handle this.

echo "Starting evaluate.py for checkpoint: '${CHECKPOINT_DIR_NAME_TO_EVAL}'..."

# Paths for evaluate.py, *inside the container*
EVAL_PY_SCRIPT_CONTAINER_PATH="${CONTAINER_WORKSPACE_EVAL_JOB}/src/evaluate.py"
# Path to the specific checkpoint directory *inside the container*
CHECKPOINT_TO_EVAL_CONTAINER_PATH="${CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT}/${CHECKPOINT_DIR_NAME_TO_EVAL}"
# Validation and priming data paths *inside the container*
EVAL_PY_VALID_DATA_CONTAINER_PATH="${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}/processed/test_set_10m" # <<< ADJUST
EVAL_PY_PRIMING_DATA_CONTAINER_PATH="${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}/priming-corpuses" # <<< ADJUST

# --- Construct arguments for evaluate.py ---
local -a PYTHON_ARGS_FOR_EVALUATE_PY
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_path" "${CHECKPOINT_TO_EVAL_CONTAINER_PATH}" )
# evaluate.py will write its results to this path inside the container, which is a mount of HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--output_dir" "${CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--checkpoint_label" "${CHECKPOINT_DIR_NAME_TO_EVAL}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--seed" "${SEED_FOR_THIS_EVAL}" )

PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_standard_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--validation_dataset_path" "$EVAL_PY_VALID_DATA_CONTAINER_PATH" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--run_priming_eval" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_eval_dir_path" "$EVAL_PY_PRIMING_DATA_CONTAINER_PATH" )

PYTHON_ARGS_FOR_EVALUATE_PY+=( "--per_device_eval_batch_size" "16" ) # Example
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--priming_per_device_eval_batch_size" "8" )  # Example
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--num_workers" "${SLURM_CPUS_PER_TASK:-4}" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--use_amp" )

# Neptune arguments for evaluate.py
# SHARED_RUN_ID (e.g., s42_jMAINORCHID_tsTIMESTAMP) provides context
NEPTUNE_EVAL_JOB_RUN_NAME="eval_${SHARED_RUN_ID}_${CHECKPOINT_DIR_NAME_TO_EVAL}"
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_run_name" "${NEPTUNE_EVAL_JOB_RUN_NAME}" )

NEPTUNE_EVAL_JOB_TAGS_ARRAY=("evaluation_job" "${SHARED_RUN_ID}" "${CHECKPOINT_DIR_NAME_TO_EVAL}" "seed_${SEED_FOR_THIS_EVAL}")
# NEPTUNE_EVAL_JOB_TAGS_ARRAY+=("p6000_gpu") # Example if you want to tag GPU type
PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_tags" )
PYTHON_ARGS_FOR_EVALUATE_PY+=( "${NEPTUNE_EVAL_JOB_TAGS_ARRAY[@]}" )

# Neptune project is usually picked up from SINGULARITYENV_NEPTUNE_PROJECT by evaluate.py
if [ -n "${SINGULARITYENV_NEPTUNE_PROJECT:-}" ]; then
    PYTHON_ARGS_FOR_EVALUATE_PY+=( "--neptune_project" "${SINGULARITYENV_NEPTUNE_PROJECT}" )
fi

echo "Arguments for evaluate.py:"
printf "  %q\n" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # If needed

# --- Execute evaluate.py inside Singularity ---
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE_EVAL_JOB}" \
    -B "${HOST_DATA_BASE_DIR_EVAL_JOB}":"${CONTAINER_DATA_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${HOST_PRIMING_BASE_DIR_EVAL_JOB}":"${CONTAINER_PRIMING_DIR_EVAL_MNT_EVAL_JOB}" \
    -B "${SHARED_OUTPUT_DIR_HOST}":"${CONTAINER_ALL_CHECKPOINTS_BASE_MOUNT}" \
    -B "${HOST_PATH_FOR_THIS_EVAL_JOB_RESULTS}":"${CONTAINER_THIS_EVAL_RESULTS_OUTPUT_TARGET}" \
    "${HOST_SIF_PATH_EVAL_JOB}" \
    python3 "${EVAL_PY_SCRIPT_CONTAINER_PATH}" "${PYTHON_ARGS_FOR_EVALUATE_PY[@]}"

EVALUATE_PY_EXIT_CODE=$?
if [ $EVALUATE_PY_EXIT_CODE -ne 0 ]; then
    echo "CRITICAL ERROR (Single Eval Job): evaluate.py FAILED with exit code ${EVALUATE_PY_EXIT_CODE} for checkpoint '${CHECKPOINT_DIR_NAME_TO_EVAL}'."
    exit 1 # Make this Slurm job fail
fi

echo "evaluate.py completed successfully for checkpoint '${CHECKPOINT_DIR_NAME_TO_EVAL}'."
echo "=== Single Evaluation Job Finished: $(date) ==="