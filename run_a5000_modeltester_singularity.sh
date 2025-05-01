#!/bin/bash

# === SBATCH Directives for A5000 Hyperparameter Sweep ===
#SBATCH --job-name=a5k_hp_sweep
#SBATCH --partition=general_gpu_a5000   # <<< Target A5000 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          # <<< Increased CPUs for potential data loading variance
#SBATCH --mem=64G              # <<< Increased RAM for safety across runs
#SBATCH --gres=gpu:a5000:1       # <<< Explicitly request 1 A5000
#SBATCH --time=24:00:00          # <<< Request sufficient TOTAL time for all runs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit on error
set -e

# === Environment Setup (Load modules, Neptune creds - as before) ===
echo "=== Job Started (A5000 HP Sweep): $(date) ==="
# ... (your existing setup: echo vars, module load, source creds) ...
module load singularity/4.1.1 cuda/11.8 # <<< Or appropriate CUDA for A5000 if different
source "$HOME/.neptune_creds"
export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-thmorton/NothingProject}" # Set default project

# === Path Definitions (As before) ===
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif"
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
# HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval" # Not needed if eval is off

CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/.output" # Output base mount inside container
# CONTAINER_PRIMING_DIR="/eval"   # Not needed

# --- Check Singularity Image ---
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Hyperparameter Sweep Configuration ===
MAX_STEPS_PER_RUN=2000  # <<< Train each configuration for this many OPTIMIZER steps
SEED=42                 # <<< Use a fixed seed for comparability

# --- Define Parameter Grid ---
model_sizes=('10m' '100m')
learning_rates=(1e-4 3e-4 5e-4)
# Define batch configs separately per model size based on A5000 (24GB VRAM)
# Format: "batch_size gradient_accumulation_steps"
# Effective BS = 128 targeted here, adjust as needed
batch_configs_10m=('32 4' '64 2')   # Likely fit easily on A5000
batch_configs_100m=('16 8' '32 4') # Might fit larger BS than P6000 due to faster compute/AMP

# Base output dir for this entire sweep job
SWEEP_NAME="a5k_hp_sweep_${SLURM_JOB_ID}"
SWEEP_HOST_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${SWEEP_NAME}"
SWEEP_CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}/${SWEEP_NAME}" # Relative path within mount
mkdir -p "$SWEEP_HOST_OUTPUT_DIR"
echo "Sweep Output Base (Host): $SWEEP_HOST_OUTPUT_DIR"
echo "Sweep Output Base (Container): $SWEEP_CONTAINER_OUTPUT_DIR"


# === Loop Through Configurations ===
TOTAL_RUNS=0
for ms in "${model_sizes[@]}"; do
  # Select correct batch configs based on model size
  declare -a batch_configs # Ensure scope for loop below
  if [[ "$ms" == "10m" ]]; then
    batch_configs=("${batch_configs_10m[@]}")
  elif [[ "$ms" == "100m" ]]; then
    batch_configs=("${batch_configs_100m[@]}")
  else
    echo "WARNING: Unknown model size '$ms'. Skipping."
    continue
  fi

  for lr in "${learning_rates[@]}"; do
    for bc in "${batch_configs[@]}"; do
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
      # Extract batch size and accum steps
      read -r bs accum <<< "$bc" # Read the space-separated pair

      # --- Create unique names/paths for this specific run ---
      run_id="run${TOTAL_RUNS}_ms${ms}_lr${lr}_bs${bs}_ac${accum}"
      echo "############################################################"
      echo "### Starting Run ${TOTAL_RUNS}: ${run_id}"
      echo "############################################################"

      HOST_RUN_OUTPUT_DIR="${SWEEP_HOST_OUTPUT_DIR}/${run_id}"
      # The path used inside the python script needs to be relative to the container mount point
      CONTAINER_RUN_OUTPUT_DIR_ARG="${SWEEP_CONTAINER_OUTPUT_DIR}/${run_id}"
      mkdir -p "$HOST_RUN_OUTPUT_DIR" # Ensure host dir exists for logs etc.

      NEPTUNE_RUN_NAME="${SWEEP_NAME}_${run_id}"
      NEPTUNE_TAGS="a5000 hp_sweep $ms lr${lr} bs${bs} ac${accum}" # Specific tags

      # Set dataset path based on model size (assuming your naming convention)
      CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_${ms}"
      # Validation/Priming paths not needed if eval triggers are off

      # --- Execute Singularity Command for this configuration ---
      singularity exec --nv \
          -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
          -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
          -B "${HOST_OUTPUT_BASE_DIR}":"${CONTAINER_OUTPUT_DIR}" \
          "${HOST_SIF_PATH}" \
          python3 "${CONTAINER_WORKSPACE}/src/train.py" \
              --model "gpt2" \
              --model_size "$ms" \
              --train_dataset_path "$CONTAINER_TRAIN_DATA_PATH" \
              --output_dir "$CONTAINER_RUN_OUTPUT_DIR_ARG" \
              \
              --max_steps "$MAX_STEPS_PER_RUN"   # <<< LIMIT DURATION
              # --num_train_epochs 1             # <<< Alternative: Limit to 1 epoch
              --per_device_train_batch_size "$bs" \
              --gradient_accumulation_steps "$accum" \
              \
              --learning_rate "$lr" \
              --lr_scheduler_type "cosine" \
              --num_warmup_steps 100        # Warmup steps might need scaling for short runs? Keep simple for now.
              --weight_decay 0.01 \
              --max_grad_norm 1.0 \
              \
              --use_amp \
              --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
              --seed "$SEED" \               # Use same base seed
              \
              --logging_steps 50           # Log frequently during short run
              --save_steps "$MAX_STEPS_PER_RUN"      # Only save checkpoint at the very end
              # --- Disable evaluation args ---
              # --eval_steps ...
              # --priming_eval_steps ...
              # --local_eval
              # --submit_eval_script_path ...
              # --evaluate_script_path ...
              # --validation_dataset_path ...
              # --priming_eval_dir_path ...
              # --trigger_standard_eval  # REMOVE/COMMENT OUT
              # --trigger_priming_eval # REMOVE/COMMENT OUT
              \
              --neptune_project "${SINGULARITYENV_NEPTUNE_PROJECT}" \ # Use env var if set
              --neptune_run_name "${NEPTUNE_RUN_NAME}" \
              --neptune_tags ${NEPTUNE_TAGS} # Note: tags need to be space separated

      echo "------------------------------------------------------------"
      echo "### Finished Run ${TOTAL_RUNS}: ${run_id}"
      echo "------------------------------------------------------------"
      sleep 10 # Small pause between runs for safety (filesystem sync etc.)

    done # batch configs
  done # learning rates
done # model sizes

echo "==== Hyperparameter Sweep Complete (${TOTAL_RUNS} runs) ===="