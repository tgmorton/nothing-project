#!/bin/bash

# === SBATCH Directives for A5000 Hyperparameter Sweep ===
#SBATCH --job-name=a5k_hp_sweep_eval
#SBATCH --partition=general_gpu_a5000   # <<< Target A5000 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12          # <<< Increased CPUs
#SBATCH --mem=64G              # <<< Increased RAM
#SBATCH --gres=gpu:a5000:1       # <<< Explicitly request 1 A5000
#SBATCH --time=24:00:00          # <<< Increased time slightly for local evals
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit on error
set -e

# === Environment Setup (Load modules, Neptune creds - as before) ===
echo "=== Job Started (A5000 HP Sweep w/ Local Eval): $(date) ==="
module load singularity/4.1.1 cuda/11.8 # <<< Or appropriate CUDA for A5000 if different
source "$HOME/.neptune_creds"
export SINGULARITYENV_NEPTUNE_API_TOKEN="${NEPTUNE_API_TOKEN:-}"
export SINGULARITYENV_NEPTUNE_PROJECT="${NEPTUNE_PROJECT:-thmorton/NothingProject}"

# === Path Definitions ===
HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project"
HOST_SIF_PATH="/home/AD/thmorton/python39_llm_env.sif"
HOST_DATA_BASE_DIR="${HOST_PROJECT_DIR}/data"
HOST_OUTPUT_BASE_DIR="${HOST_PROJECT_DIR}/src/.output"
HOST_PRIMING_BASE_DIR="${HOST_PROJECT_DIR}/eval" # <<< Define priming base path

CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_OUTPUT_DIR="/.output"
CONTAINER_PRIMING_DIR="/eval"   # <<< Define priming container path

# --- Check Singularity Image ---
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"

# === Hyperparameter Sweep Configuration ===
MAX_STEPS_PER_RUN=2000  # Train each configuration for this many OPTIMIZER steps
NUM_EVALS_PER_RUN=4     # <<< Number of evaluations during the MAX_STEPS run
SEED=42

# --- Define Parameter Grid ---
model_sizes=('10m' '100m')
learning_rates=(3e-4 5e-4) # Reduced grid slightly for example brevity
# Format: "batch_size gradient_accumulation_steps"
batch_configs_10m=('64 2') # Reduced grid slightly
batch_configs_100m=('16 8' '32 4')

# Base output dir for this entire sweep job
SWEEP_NAME="a5k_hp_sweep_eval_${SLURM_JOB_ID}"
SWEEP_HOST_OUTPUT_DIR="${HOST_OUTPUT_BASE_DIR}/${SWEEP_NAME}"
SWEEP_CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}/${SWEEP_NAME}" # Relative path within mount
mkdir -p "$SWEEP_HOST_OUTPUT_DIR"
echo "Sweep Output Base (Host): $SWEEP_HOST_OUTPUT_DIR"
echo "Sweep Output Base (Container): $SWEEP_CONTAINER_OUTPUT_DIR"


# === Loop Through Configurations ===
TOTAL_RUNS=0
for ms in "${model_sizes[@]}"; do
  # Select correct batch configs based on model size
  declare -a batch_configs
  if [[ "$ms" == "10m" ]]; then
    batch_configs=("${batch_configs_10m[@]}")
    # Define eval dataset paths for 10m
    CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_10m" # Example path
  elif [[ "$ms" == "100m" ]]; then
    batch_configs=("${batch_configs_100m[@]}")
    # Define eval dataset paths for 100m
    CONTAINER_VALID_DATA_PATH="${CONTAINER_DATA_DIR}/processed/test_set_100m" # Example path
  else
    echo "WARNING: Unknown model size '$ms'. Skipping."
    continue
  fi
  # Define fixed priming path (used for both model sizes in this example)
  CONTAINER_PRIMING_PATH="${CONTAINER_PRIMING_DIR}/priming-corpuses" # Example path

  for lr in "${learning_rates[@]}"; do
    for bc in "${batch_configs[@]}"; do
      TOTAL_RUNS=$((TOTAL_RUNS + 1))
      read -r bs accum <<< "$bc"

      # --- Calculate Eval Steps ---
      eval_steps_value=$(( MAX_STEPS_PER_RUN / NUM_EVALS_PER_RUN ))
      # Ensure eval_steps is at least 1
      if [[ $eval_steps_value -lt 1 ]]; then
          eval_steps_value=1
      fi
      # Also ensure save steps are reasonable (at least match eval, maybe slightly more frequent?)
      # Let's set save_steps = eval_steps for simplicity, train.py handles saving if needed for eval anyway
      save_steps_value=$eval_steps_value
      echo "Calculating steps: MAX_STEPS=${MAX_STEPS_PER_RUN}, NUM_EVALS=${NUM_EVALS_PER_RUN} -> eval_steps=${eval_steps_value}, save_steps=${save_steps_value}"

      # --- Create unique names/paths for this specific run ---
      run_id="run${TOTAL_RUNS}_ms${ms}_lr${lr}_bs${bs}_ac${accum}"
      echo "############################################################"
      echo "### Starting Run ${TOTAL_RUNS}: ${run_id}"
      echo "############################################################"

      HOST_RUN_OUTPUT_DIR="${SWEEP_HOST_OUTPUT_DIR}/${run_id}"
      CONTAINER_RUN_OUTPUT_DIR_ARG="${SWEEP_CONTAINER_OUTPUT_DIR}/${run_id}"
      mkdir -p "$HOST_RUN_OUTPUT_DIR"

      NEPTUNE_RUN_NAME="${SWEEP_NAME}_${run_id}"
      NEPTUNE_TAGS="a5000 hp_sweep_eval $ms lr${lr} bs${bs} ac${accum}" # Updated tag

      # Set dataset path based on model size
      CONTAINER_TRAIN_DATA_PATH="${CONTAINER_DATA_DIR}/processed/training_set_${ms}"

      # --- Execute Singularity Command for this configuration ---
      singularity exec --nv \
          -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
          -B "${HOST_DATA_BASE_DIR}":"${CONTAINER_DATA_DIR}" \
          -B "${HOST_OUTPUT_BASE_DIR}":"${CONTAINER_OUTPUT_DIR}" \
          -B "${HOST_PRIMING_BASE_DIR}":"${CONTAINER_PRIMING_DIR}" \
          "${HOST_SIF_PATH}" \
          python3 "${CONTAINER_WORKSPACE}/src/train.py" \
              --model "gpt2" \
              --model_size "$ms" \
              --train_dataset_path "$CONTAINER_TRAIN_DATA_PATH" \
              --output_dir "$CONTAINER_RUN_OUTPUT_DIR_ARG" \
              \
              --max_steps "$MAX_STEPS_PER_RUN" \
              --per_device_train_batch_size "$bs" \
              --gradient_accumulation_steps "$accum" \
              \
              --learning_rate "$lr" \
              --lr_scheduler_type "cosine" \
              --num_warmup_steps 100  \
              --weight_decay 0.01 \
              --max_grad_norm 1.0 \
              \
              --use_amp \
              --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
              --seed "$SEED" \
              --logging_steps 50 \
              --save_steps "$save_steps_value" \
              \
              --neptune_project "${SINGULARITYENV_NEPTUNE_PROJECT}" \
              --neptune_run_name "${NEPTUNE_RUN_NAME}" \
              --neptune_tags ${NEPTUNE_TAGS} \
              \
              --eval_steps "$eval_steps_value" \
              --priming_eval_steps "$eval_steps_value" \
              --local_eval \
              --evaluate_script_path "${CONTAINER_WORKSPACE}/src/evaluate.py" \
              --validation_dataset_path "$CONTAINER_VALID_DATA_PATH" \
              --priming_eval_dir_path "$CONTAINER_PRIMING_PATH" \
              --trigger_standard_eval \
              --trigger_priming_eval

      echo "------------------------------------------------------------"
      echo "### Finished Run ${TOTAL_RUNS}: ${run_id}"
      echo "------------------------------------------------------------"
      sleep 10 # Small pause between runs

    done # batch configs
  done # learning rates
done # model sizes

echo "==== Hyperparameter Sweep Complete (${TOTAL_RUNS} runs) ===="