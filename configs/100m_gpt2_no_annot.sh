#!/bin/bash
# 100m_gpt2_no_annot.sh

# --- General Project Settings ---
export HOST_PROJECT_DIR="~/nothing-project" # IMPORTANT: Set this to your actual project path
export SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT="python39_llm_env.sif"
export DATA_DIR_RELATIVE_TO_PROJECT_ROOT="data"
export PRIMING_DIR_RELATIVE_TO_PROJECT_ROOT="eval" # Priming CSVs base directory
export OUTPUT_BASE_DIR_RELATIVE_TO_PROJECT_ROOT="src/.output" # Base for SHARED_RUN_ID folders
export CHILD_JOB_LOGS_DIR_RELATIVE_TO_PROJECT_ROOT="logs/main_orch_children" # Base for child Slurm logs from main_orchestrator
export CHECKPOINT_READY_SENTINEL_FILENAME="EVAL_READY.txt"
export TRAINING_COMPLETION_SENTINEL_FILENAME="TRAINING_COMPLETED.txt"

# --- main_orchestrator.sh Specifics ---
export MAIN_ORCH_JOB_NAME="main_orch"
export MAIN_ORCH_PARTITION="general" # CPU partition
export MAIN_ORCH_CPUS="1"
export MAIN_ORCH_MEM="4G"
export MAIN_ORCH_TIME="7-00:00:00" # Max time for the entire pipeline run
export MAIN_ORCH_LOG_DIR="../logs" # Relative to scripts folder, so project_root/logs
export MAIN_ORCH_MAIL_TYPE="END,FAIL"
export MAIN_ORCH_MAIL_USER="thmorton@ucsd.edu"
export SEED_FOR_STANDALONE="40" # Default seed if not passed by a sweep
export TRAINING_JOB_SCRIPT_RELATIVE_PATH="scripts/training_job.sh"
export EVAL_ORCH_SCRIPT_RELATIVE_PATH="scripts/eval_orchestrator.sh"
export JOB_WAIT_INTERVAL_SECONDS="120"

# --- training_job.sh Specifics ---
export TRAIN_JOB_NAME_PREFIX="train" # Will be appended with _${SHARED_RUN_ID}
export TRAIN_JOB_PARTITION="general_gpu_a5000" # Training GPU partition
export TRAIN_JOB_CPUS="8"
export TRAIN_JOB_MEM="64G"
export TRAIN_JOB_TIME="7-00:00:00" # Max time for one training run
export TRAIN_JOB_MAIL_TYPE="END,FAIL"
export TRAIN_JOB_MAIL_USER="thmorton@ucsd.edu"
# export TRAIN_JOB_GPU_TYPE_COUNT="a100:1" # Example if using specific gres for training

export SYSTEM_MODULES_TRAIN="singularity/4.1.1 cuda/11.8"
export NEPTUNE_CRED_FILE_PATH="$HOME/.neptune_creds" # Path to Neptune credentials
export DEFAULT_NEPTUNE_PROJECT="thmorton/nothing-project" # Fallback Neptune project for training
export PYTORCH_CUDA_ALLOC_CONF_TRAIN="expandable_segments:True"

# Container paths (usually standard, but can be configured)
export CONTAINER_WORKSPACE_PATH="/workspace"
export CONTAINER_DATA_DIR_PATH="/data"
export CONTAINER_OUTPUT_DIR_PATH="/output_train" # Where train.py writes checkpoints inside container

# train.py script arguments
export TRAIN_PY_SCRIPT_RELATIVE_PATH="src/train.py"
export TRAIN_PY_MODEL_NAME="gpt2"
export TRAIN_PY_MODEL_SIZE="100m"
export TRAIN_PY_TRAIN_DATASET_RELATIVE_PATH="processed/training_set_100m" # Relative to CONTAINER_DATA_DIR_PATH
export TRAIN_PY_VALID_DATASET_RELATIVE_PATH="processed/test_set_10m"   # Relative to CONTAINER_DATA_DIR_PATH
export TRAIN_PY_NUM_EPOCHS="2"
export TRAIN_PY_BATCH_SIZE="8"
export TRAIN_PY_GRAD_ACCUM_STEPS="16"
export TRAIN_PY_LR="3e-4"
export TRAIN_PY_LR_SCHEDULER="cosine"
export TRAIN_PY_WARMUP_STEPS="1500"
export TRAIN_PY_WEIGHT_DECAY="0.01"
export TRAIN_PY_MAX_GRAD_NORM="1.0"
export TRAIN_PY_USE_AMP="true" # "true" or "false"
export TRAIN_PY_LOGGING_STEPS="100"
export TRAIN_PY_SAVE_STEPS="1500"
export NEPTUNE_TRAINING_TAGS_DEFAULT="training_phase my_model_type" # Space separated, more will be added by script

# --- eval_orchestrator.sh (Submitter) Specifics ---
export EVAL_ORCH_SUBMITTER_JOB_NAME="eval_orch_submit"
export EVAL_ORCH_SUBMITTER_PARTITION="general"
export EVAL_ORCH_SUBMITTER_CPUS="1"
export EVAL_ORCH_SUBMITTER_MEM="1G"
export EVAL_ORCH_SUBMITTER_TIME="0-00:10:00"
export EVAL_ORCH_SUBMITTER_MAIL_TYPE="FAIL"
export EVAL_ORCH_SUBMITTER_MAIL_USER="your_email@example.com"
export EVAL_JOB_SCRIPT_RELATIVE_PATH="scripts/eval_job.sh" # Path to the long-running eval_job.sh
export EVAL_JOB_SLURM_LOGS_SUBDIR_NAME="long_running_eval_slurm_logs" # Inside SHARED_OUTPUT_DIR_HOST
export EVALUATE_PY_MULTI_OUTPUT_SUBDIR_NAME="eval_results_multi" # Inside SHARED_OUTPUT_DIR_HOST, base for evaluate.py outputs
export EVAL_JOB_NAME_PREFIX="multi_eval_watch" # Will be appended with _${SHARED_RUN_ID}

# --- eval_job.sh (Long-running Worker) Specifics ---
export EVAL_JOB_PARTITION="general_gpu_p6000" # Evaluation GPU partition
export EVAL_JOB_CPUS="4"
export EVAL_JOB_MEM="32G"
export EVAL_JOB_TIME="7-00:00:00" # Max time, e.g., 7 days
export EVAL_JOB_GPU_REQUEST="1" # e.g., "1", or "p6000:1" or "gpu:1" depending on Slurm config
export EVAL_JOB_MAIL_TYPE="FAIL,END"
export EVAL_JOB_MAIL_USER="thmorton@ucsd.edu"

export SYSTEM_MODULES_EVAL="singularity/4.1.1 cuda/11.8"
export SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT_EVAL="${SIF_IMAGE_PATH_RELATIVE_TO_PROJECT_ROOT}" # Can be same as train SIF
export DATA_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL="${DATA_DIR_RELATIVE_TO_PROJECT_ROOT}"
export PRIMING_DIR_RELATIVE_TO_PROJECT_ROOT_EVAL="eval" # Path for priming CSVs relative to project root
export NEPTUNE_CRED_FILE_PATH_EVAL="$HOME/.neptune_creds_eval" # Optional: different creds for eval
export DEFAULT_NEPTUNE_PROJECT_EVAL="your-workspace/your-eval-project" # Fallback Neptune project for eval
export PYTORCH_CUDA_ALLOC_CONF_EVAL="expandable_segments:True"

# Container paths for eval_job.sh
export CONTAINER_WORKSPACE_PATH_EVAL="/workspace"
export CONTAINER_DATA_DIR_PATH_EVAL="/data_eval_mnt"
export CONTAINER_PRIMING_DIR_PATH_EVAL="/priming_data_eval_mnt"
export CONTAINER_CHECKPOINT_WATCH_DIR_MOUNT_PATH="/checkpoints_to_watch" # Where SHARED_OUTPUT_DIR_TO_WATCH is mounted
export CONTAINER_EVALUATE_PY_RESULTS_BASE_MOUNT_PATH="/eval_results_base" # Where EVALUATE_PY_OVERALL_OUTPUT_DIR_HOST is mounted

# evaluate.py script arguments (for the long-running job)
export EVAL_PY_SCRIPT_RELATIVE_PATH="src/evaluate.py"
export EVAL_PY_WATCH_INTERVAL_SECONDS="5"
export EVAL_PY_BASE_MODEL_NAME_OR_PATH="gpt2" # Or path to a base model dir in project
export EVAL_PY_MODEL_CLASS_NAME="GPT2LMHeadModel"
export EVAL_PY_RUN_STANDARD_EVAL="true"
export EVAL_PY_VALID_DATASET_RELATIVE_PATH="processed/test_set_10m" # Relative to CONTAINER_DATA_DIR_PATH_EVAL
export EVAL_PY_STD_EVAL_MAX_SAMPLES="0"
export EVAL_PY_RUN_PRIMING_EVAL="true"
export EVAL_PY_PRIMING_DIR_RELATIVE_PATH="priming-corpuses" # Relative to CONTAINER_PRIMING_DIR_PATH_EVAL
export EVAL_PY_PRIMING_MAX_SAMPLES_PER_FILE="1000"
export EVAL_PY_PRIMING_DELIMITER="."
export EVAL_PY_STD_BATCH_SIZE="16"
export EVAL_PY_PRIMING_BATCH_SIZE="8"
export EVAL_PY_USE_AMP="true"

echo "Project config loaded."