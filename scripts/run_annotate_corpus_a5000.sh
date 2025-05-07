#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=annotate_corpus_a5000
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # annotate_corpus.py is single-threaded but benefits from multi-core for data loading/pytorch
#SBATCH --mem=32G          # Increased slightly for potentially large files in memory during chunking, 24G might still be fine
#SBATCH --time=24:00:00    # Adjust based on your corpus size
#SBATCH --output=../logs/annotate_corpus_%x_%j.out
#SBATCH --error=../logs/annotate_corpus_%x_%j.err

# Exit on error
set -e

# === Environment Setup ===
echo "=== Annotation Job Started: $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE" # Or SLURM_MEM_PER_TASK depending on Slurm version
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # Adjust versions if needed

# --- Define Paths on Host ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # Your main project directory
HOST_SIF_PATH="/home/AD/thmorton/nothing-project/python39_annotate.sif" # Your Singularity image

# Data paths for annotation
HOST_RAW_DATA_DIR="${HOST_PROJECT_DIR}/data/raw/text_data/train_10M" # INPUT: Where your .train files are
HOST_ANNOTATED_DATA_DIR="${HOST_PROJECT_DIR}/data/annotated_text_data/train_10M" # OUTPUT: Where annotated files will go

# Script path
HOST_ANNOTATE_SCRIPT_PATH="${HOST_PROJECT_DIR}/src/annotate_corpus.py"

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace" # Project directory mounted here
CONTAINER_RAW_DATA_DIR="/raw_data" # Mount point for input data
CONTAINER_ANNOTATED_DATA_DIR="/annotated_data" # Mount point for output data
CONTAINER_ANNOTATE_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/annotate_corpus.py" # Script path inside container

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Raw Data Directory (Host): ${HOST_RAW_DATA_DIR}"
echo "Annotated Data Directory (Host): ${HOST_ANNOTATED_DATA_DIR}"

if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "$HOST_RAW_DATA_DIR" ]; then echo "ERROR: Raw data directory not found at $HOST_RAW_DATA_DIR"; exit 1; fi
if [ ! -f "$HOST_ANNOTATE_SCRIPT_PATH" ]; then echo "ERROR: Annotation script not found at $HOST_ANNOTATE_SCRIPT_PATH"; exit 1; fi

mkdir -p "${HOST_ANNOTATED_DATA_DIR}" # Ensure output directory exists on the host
echo "Ensured host annotated data directory exists: ${HOST_ANNOTATED_DATA_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs" # For Slurm logs

# --- Set PyTorch CUDA Allocator Config (Can be useful for PyTorch generally) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF # Pass to container

# === Annotation Script Execution ===
echo "Starting annotation script inside Singularity container..."

# Note on installing packages inside Singularity at runtime:
# The following 'pip install --user' and 'spacy download' commands attempt to install
# packages into the user's home directory (or a specific path if PYTHONUSERBASE is set)
# *inside the container*. This works if $HOME is mounted and writable from within the container.
# If your SIF is strictly immutable and $HOME is not suitable, these packages
# should ideally be part of the SIF image itself.
# This approach is a common workaround.

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_RAW_DATA_DIR}":"${CONTAINER_RAW_DATA_DIR}" \
    -B "${HOST_ANNOTATED_DATA_DIR}":"${CONTAINER_ANNOTATED_DATA_DIR}" \
    "${HOST_SIF_PATH}" \
    bash -c "
        set -e
        echo '--- Container: Environment Info ---'
        pwd
        ls -la /
        echo USER: \$USER
        echo HOME: \$HOME
        id
        echo '--- Container: Attempting to ensure dependencies ---'

        # Ensure pip is available and then install packages using --user
        # The --user flag installs packages into the user's site-packages directory
        # (e.g., $HOME/.local/lib/python3.9/site-packages for Python 3.9)
        # This requires $HOME to be writable within the container.
        python3 -m pip install --upgrade pip
        echo 'Attempting to install spacy, tqdm, transformers, torch...'
        python3 -m pip install --user spacy tqdm transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Ensure CUDA version matches torch wheel

        # Check for spaCy model and download if not present
        echo 'Checking for spaCy model en_core_web_sm...'
        if ! python3 -m spacy info en_core_web_sm > /dev/null 2>&1; then
            echo 'Downloading spaCy model en_core_web_sm...'
            python3 -m spacy download en_core_web_sm
        else
            echo 'spaCy model en_core_web_sm seems to be available.'
        fi
        echo '--- Container: Dependency setup attempt finished ---'

        echo '--- Container: Starting annotation script ---'
        python3 \"${CONTAINER_ANNOTATE_SCRIPT_PATH}\" \
            \"${CONTAINER_RAW_DATA_DIR}\" \
            \"${CONTAINER_ANNOTATED_DATA_DIR}\" \
            --bert_model_name \"bert-base-uncased\" \
            --spacy_model_name \"en_core_web_sm\" \
            --k_top 10 \
            --chunk_size_chars 500000 # Default from your script, adjust if needed

        echo '--- Container: Annotation script finished ---'
    "

# === Job Completion ===
echo "=== Annotation Job Finished: $(date) ==="