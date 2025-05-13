#!/bin/bash
# project_config.sh
# Configuration for 10m GPT-2 model, 2 epochs, local evaluation enabled.

# --- General Project Settings ---
export HOST_PROJECT_DIR="/home/AD/thmorton/nothing-project" # <<< UPDATE THIS TO YOUR ACTUAL PROJECT PATH
export SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT="python39_llm_env.sif"
export DATA_DIR_RELATIVE_TO_PROJECT_ROOT="data"
export PRIMING_DIR_RELATIVE_TO_PROJECT_ROOT="eval"
export OUTPUT_BASE_DIR_RELATIVE_TO_PROJECT_ROOT="src/.output"
export CHILD_JOB_LOGS_DIR_RELATIVE_TO_PROJECT_ROOT="logs/main_orch_children"
export CHECKPOINT_READY_SENTINEL_FILENAME="EVAL_READY.txt"
export TRAINING_COMPLETION_SENTINEL_FILENAME="TRAINING_COMPLETED.txt"

# --- main_orchestrator.sh Specifics ---
# ... (keep these relevant if you use the full orchestrator later)
export MAIN_ORCH_JOB_NAME="main_orch_10m"
export MAIN_ORCH_PARTITION="general"
export MAIN_ORCH_CPUS="1"
export MAIN_ORCH_MEM="4G"
export MAIN_ORCH_TIME="1-00:00:00" # Shorter for a 2-epoch 10m model run
export MAIN_ORCH_LOG_DIR="../logs"
export MAIN_ORCH_MAIL_TYPE="END,FAIL"
export MAIN_ORCH_MAIL_USER="thmorton@ucsd.edu" # <<< UPDATE
export SEED_FOR_STANDALONE="42"
export TRAINING_JOB_SCRIPT_RELATIVE_PATH="scripts/training_job.sh"
export EVAL_ORCH_SCRIPT_RELATIVE_PATH="scripts/eval_orchestrator.sh"
export JOB_WAIT_INTERVAL_SECONDS="60"

# --- training_job.sh Specifics ---
export TRAIN_JOB_NAME_PREFIX="train_10m"
export TRAIN_JOB_PARTITION="general_gpu_p6000" # <<< UPDATE to your single GPU partition
export TRAIN_JOB_CPUS="4" # CPUs for dataloading, etc.
export TRAIN_JOB_MEM="32G" # Memory for a 10m model can likely be less
export TRAIN_JOB_TIME="0-12:00:00" # Max time for this specific 10m training
export TRAIN_JOB_GPU_REQUEST="1" # Request 1 GPU of any available type in partition
# export TRAIN_JOB_GPU_REQUEST="specific_gpu_type:1" # Or specific type
export TRAIN_JOB_MAIL_TYPE="END,FAIL"
export TRAIN_JOB_MAIL_USER="thmorton@ucsd.edu" # <<< UPDATE

export SYSTEM_MODULES_TRAIN="singularity/4.1.1 cuda/11.8" # <<< UPDATE if needed
export NEPTUNE_CRED_FILE_PATH="$HOME/.neptune_creds"
export DEFAULT_NEPTUNE_PROJECT="thmorton/nothing-project" # <<< UPDATE
export PYTORCH_CUDA_ALLOC_CONF_TRAIN="expandable_segments:True"

export CONTAINER_WORKSPACE_PATH="/workspace"
export CONTAINER_DATA_DIR_PATH="/data"
export CONTAINER_OUTPUT_DIR_PATH="/output_train"

# train.py script arguments for 10m model
export TRAIN_PY_SCRIPT_RELATIVE_PATH="src/train.py"
export TRAIN_PY_MODEL_NAME="gpt2" # Base for tokenizer/config structure
export TRAIN_PY_MODEL_SIZE="10m"  # Crucial for creating the 10m config in train.py
export TRAIN_PY_TRAIN_DATASET_RELATIVE_PATH="processed/training_set_10m" # <<< UPDATE to your 10m dataset
export TRAIN_PY_VALID_DATASET_RELATIVE_PATH="processed/test_set_10m"   # <<< UPDATE to your 10m validation set
export TRAIN_PY_NUM_EPOCHS="2"
export TRAIN_PY_BATCH_SIZE="8"
export TRAIN_PY_GRAD_ACCUM_STEPS="16"
export TRAIN_PY_LR="5e-4"
export TRAIN_PY_LR_SCHEDULER="cosine"
export TRAIN_PY_WARMUP_STEPS="100" # Adjusted for shorter run
export TRAIN_PY_WEIGHT_DECAY="0.01"
export TRAIN_PY_MAX_GRAD_NORM="1.0"
export TRAIN_PY_USE_AMP="true"
export TRAIN_PY_LOGGING_STEPS="50" # More frequent logging for shorter run
export TRAIN_PY_SAVE_STEPS="200"   # Save every 200 steps

# Settings for LOCAL EVALUATION (triggered by train.py)
export TRAIN_PY_LOCAL_EVAL="true" # Enable local evaluation
export TRAIN_PY_EVALUATE_SCRIPT_PATH="src/evaluate.py" # Path to the new evaluate.py
export TRAIN_PY_TRIGGER_STANDARD_EVAL="true" # Trigger standard perplexity
export TRAIN_PY_TRIGGER_PRIMING_EVAL="true"  # Trigger priming eval
# Path to priming data dir for train.py to pass to evaluate.py during local eval
export TRAIN_PY_PRIMING_EVAL_DIR_CONFIG_PATH="eval/priming-corpuses" # Relative to HOST_PROJECT_DIR
export TRAIN_PY_EVAL_STEPS="200"   # Evaluate every 200 steps
export TRAIN_PY_PRIMING_EVAL_STEPS="200" # Evaluate priming every 200 steps

export NEPTUNE_TRAINING_TAGS_DEFAULT="training_phase gpt2_10m 2epoch"

# --- eval_orchestrator.sh (Submitter) Specifics ---
# ... (keep relevant if you use the full orchestrator later)
export EVAL_ORCH_SUBMITTER_TIME="0-00:05:00" # Shorter for this specific run

# --- eval_job.sh (Long-running Worker) Specifics ---
export EVAL_JOB_PARTITION="general_gpu_p6000" # <<< UPDATE to your single GPU partition for eval
export EVAL_JOB_CPUS="4"
export EVAL_JOB_MEM="32G"
export EVAL_JOB_TIME="1-00:00:00" # Time for the watcher if used with orchestrator
export EVAL_JOB_GPU_REQUEST="1"
export EVAL_JOB_MAIL_TYPE="FAIL,END"
export EVAL_JOB_MAIL_USER="your_email@example.com" # <<< UPDATE

export SYSTEM_MODULES_EVAL="singularity/4.1.1 cuda/11.8" # <<< UPDATE
export SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT_EVAL="${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT}"
export DATA_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL="${DATA_DIR_RELATIVE_TO_PROJECT_ROOT}"
export PRIMING_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL="eval"
export DEFAULT_NEPTUNE_PROJECT_EVAL="your-workspace/your-project" # <<< UPDATE
export PYTORCH_CUDA_ALLOC_CONF_EVAL="expandable_segments:True"

export CONTAINER_DATA_DIR_PATH_EVAL="/data_eval_mnt"
export CONTAINER_PRIMING_DIR_PATH_EVAL="/priming_data_eval_mnt"
export CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH="/checkpoints_to_watch"
export CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH="/eval_results_base"

# evaluate.py script arguments (for the long-running job, if orchestrated)
export EVAL_PY_SCRIPT_RELATIVE_PATH="src/evaluate.py"
export EVAL_PY_WATCH_INTERVAL_SECONDS="300"
export EVAL_PY_BASE_MODEL_NAME_OR_PATH="gpt2" # Should match TRAIN_PY_MODEL_NAME for consistency
export EVAL_PY_MODEL_CLASS_NAME="GPT2LMHeadModel"
export EVAL_PY_RUN_STANDARD_EVAL="true"
export EVAL_PY_VALID_DATASET_RELATIVE_PATH="processed/test_set_10m" # <<< UPDATE (relative to CONTAINER_DATA_DIR_PATH_EVAL)
export EVAL_PY_STD_EVAL_MAX_SAMPLES="10000" # Reduced for faster eval of 10m model
export EVAL_PY_RUN_PRIMING_EVAL="true"
export EVAL_PY_PRIMING_DIR_RELATIVE_PATH="priming-corpuses" # <<< UPDATE (relative to CONTAINER_PRIMING_DIR_PATH_EVAL)
export EVAL_PY_PRIMING_MAX_SAMPLES_PER_FILE="200" # Reduced for 10m model
export EVAL_PY_PRIMING_DELIMITER="."
export EVAL_PY_STD_BATCH_SIZE="16"
export EVAL_PY_PRIMING_BATCH_SIZE="8"
export EVAL_PY_USE_AMP="true"

echo "Project config for 10m model loaded."