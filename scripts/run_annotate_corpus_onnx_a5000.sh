#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=annotate_onnx_a5000 # Updated job name
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00 # Adjust based on your corpus size & ONNX speed
#SBATCH --output=../logs/annotate_corpus_onnx_%x_%j.out # Updated log name
#SBATCH --error=../logs/annotate_corpus_onnx_%x_%j.err # Updated log name

# Exit on error
set -e

# === Environment Setup ===
echo "=== ONNX Annotation Job Started: $(date) ===" # Updated title
echo "Current Time (PDT): $(date)" # Added timezone hint
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE" # Or SLURM_MEM_PER_TASK
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # Adjust versions if needed

# --- Define Paths on Host ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # Your main project directory
# --- IMPORTANT: Update SIF path to the one built with ONNX dependencies ---
HOST_SIF_PATH="/home/AD/thmorton/nothing-project/python39_annotate_onnx.sif" # ASSUMED NEW SIF NAME

# --- Path to the directory containing the converted ONNX model and tokenizer ---
HOST_ONNX_MODEL_DIR="${HOST_PROJECT_DIR}/distilbert-onnx" # EXAMPLE: Using DistilBERT ONNX model

# Data paths for annotation
HOST_RAW_DATA_DIR="${HOST_PROJECT_DIR}/data/raw/text_data/train_10M" # INPUT
HOST_ANNOTATED_DATA_DIR="${HOST_PROJECT_DIR}/data/annotated_text_data/train_10M" # OUTPUT

# Script path
HOST_ANNOTATE_SCRIPT_PATH="${HOST_PROJECT_DIR}/src/annotate_corpus.py" # The script modified for ONNX

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace" # Project directory mounted here
CONTAINER_RAW_DATA_DIR="/raw_data" # Mount point for input data
CONTAINER_ANNOTATED_DATA_DIR="/annotated_data" # Mount point for output data
CONTAINER_ONNX_MODEL_DIR="/onnx_model" # Mount point for ONNX model/tokenizer files
CONTAINER_ANNOTATE_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/annotate_corpus.py" # Script path inside container

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "ONNX Model Directory (Host): ${HOST_ONNX_MODEL_DIR}"
echo "Raw Data Directory (Host): ${HOST_RAW_DATA_DIR}"
echo "Annotated Data Directory (Host): ${HOST_ANNOTATED_DATA_DIR}"

if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "$HOST_ONNX_MODEL_DIR" ]; then echo "ERROR: ONNX model directory not found at $HOST_ONNX_MODEL_DIR"; exit 1; fi
if [ ! -f "${HOST_ONNX_MODEL_DIR}/model.onnx" ]; then echo "ERROR: model.onnx not found in $HOST_ONNX_MODEL_DIR"; exit 1; fi # Check for model file
if [ ! -d "$HOST_RAW_DATA_DIR" ]; then echo "ERROR: Raw data directory not found at $HOST_RAW_DATA_DIR"; exit 1; fi
if [ ! -f "$HOST_ANNOTATE_SCRIPT_PATH" ]; then echo "ERROR: Annotation script not found at $HOST_ANNOTATE_SCRIPT_PATH"; exit 1; fi

mkdir -p "${HOST_ANNOTATED_DATA_DIR}" # Ensure output directory exists on the host
echo "Ensured host annotated data directory exists: ${HOST_ANNOTATED_DATA_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs" # For Slurm logs

# --- Set PyTorch CUDA Allocator Config (Less critical for ONNX RT, but harmless) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF # Pass to container

# === Annotation Script Execution ===
echo "Starting ONNX annotation script inside Singularity container..."
echo "Assuming dependencies (onnxruntime-gpu, transformers, spacy, etc.) are pre-installed in SIF: ${HOST_SIF_PATH}"

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_RAW_DATA_DIR}":"${CONTAINER_RAW_DATA_DIR}" \
    -B "${HOST_ANNOTATED_DATA_DIR}":"${CONTAINER_ANNOTATED_DATA_DIR}" \
    -B "${HOST_ONNX_MODEL_DIR}":"${CONTAINER_ONNX_MODEL_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_ANNOTATE_SCRIPT_PATH}" \
        "${CONTAINER_RAW_DATA_DIR}" \
        "${CONTAINER_ANNOTATED_DATA_DIR}" \
        --model_path "${CONTAINER_ONNX_MODEL_DIR}/model.onnx" \
        --tokenizer_path "${CONTAINER_ONNX_MODEL_DIR}" \
        --spacy_model_name "en_core_web_sm" \
        --k_top 5 \
        --chunk_size_chars 500000 # Default from your script, adjust if needed

# === Job Completion ===
echo "=== ONNX Annotation Job Finished: $(date) ==="