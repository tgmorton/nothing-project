#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=annotate_onnx_a5000
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/annotate_corpus_onnx_%x_%j.out
#SBATCH --error=../logs/annotate_corpus_onnx_%x_%j.err

# Exit on error
set -e

# === Environment Setup ===
echo "=== ONNX Annotation Job Started: $(date) ==="
echo "Current Time (PDT): $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # <<< CRITICAL: Ensure this is loaded

# --- Define Paths on Host ---
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
# --- Ensure this SIF was built with onnxruntime-gpu installed ---
HOST_SIF_PATH="/home/AD/thmorton/nothing-project/python39_annotate_onnx.sif"
HOST_ONNX_MODEL_DIR="${HOST_PROJECT_DIR}/distilbert-onnx" # Using DistilBERT ONNX
HOST_RAW_DATA_DIR="${HOST_PROJECT_DIR}/data/raw/text_data/train_10M"
HOST_ANNOTATED_DATA_DIR="${HOST_PROJECT_DIR}/data/annotated_text_data/train_10M"
HOST_ANNOTATE_SCRIPT_PATH="${HOST_PROJECT_DIR}/src/annotate_corpus.py"

# --- Define Container Paths ---
CONTAINER_WORKSPACE="/workspace"
CONTAINER_RAW_DATA_DIR="/raw_data"
CONTAINER_ANNOTATED_DATA_DIR="/annotated_data"
CONTAINER_ONNX_MODEL_DIR="/onnx_model"
CONTAINER_ANNOTATE_SCRIPT_PATH="${CONTAINER_WORKSPACE}/src/annotate_corpus_onnx.py"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
# ... (rest of path checks and directory creations) ...
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "$HOST_ONNX_MODEL_DIR" ]; then echo "ERROR: ONNX model directory not found at $HOST_ONNX_MODEL_DIR"; exit 1; fi
if [ ! -f "${HOST_ONNX_MODEL_DIR}/model.onnx" ]; then echo "ERROR: model.onnx not found in $HOST_ONNX_MODEL_DIR"; exit 1; fi
if [ ! -d "$HOST_RAW_DATA_DIR" ]; then echo "ERROR: Raw data directory not found at $HOST_RAW_DATA_DIR"; exit 1; fi
if [ ! -f "$HOST_ANNOTATE_SCRIPT_PATH" ]; then echo "ERROR: Annotation script not found at $HOST_ANNOTATE_SCRIPT_PATH"; exit 1; fi
mkdir -p "${HOST_ANNOTATED_DATA_DIR}"
echo "Ensured host annotated data directory exists: ${HOST_ANNOTATED_DATA_DIR}"
mkdir -p "${HOST_PROJECT_DIR}/logs"

# --- Environment Variables for ONNX Runtime and Container ---
# 1. Pass LD_LIBRARY_PATH (set by 'module load cuda/11.8') into the container
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "Passing LD_LIBRARY_PATH to container: $LD_LIBRARY_PATH"

# 2. Set ONNX Runtime thread settings (to address pthread_setaffinity warnings)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_WAIT_POLICY=ACTIVE
export SINGULARITYENV_OMP_NUM_THREADS=$OMP_NUM_THREADS
export SINGULARITYENV_OMP_WAIT_POLICY=$OMP_WAIT_POLICY
echo "Setting OMP_NUM_THREADS=${OMP_NUM_THREADS} for container"

# 3. PyTorch Allocator Config (less critical but keep)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF

# === Annotation Script Execution ===
echo "Starting ONNX annotation script inside Singularity container..."
echo "SIF: ${HOST_SIF_PATH}"
echo "Assuming ONNX Runtime GPU, Transformers, spaCy etc. are pre-installed in SIF."

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
        --k_top 10 \
        --chunk_size_chars 500000

# === Job Completion ===
echo "=== ONNX Annotation Job Finished: $(date) ==="