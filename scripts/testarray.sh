#!/bin/bash

#SBATCH --job-name=env_array_test
#SBATCH --partition=general_gpu_a5000  # Use the partition that was failing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=../logs/env_array_test_%A_%a.out # %A is job ID, %a is array task ID
#SBATCH --error=../logs/env_array_test_%A_%a.err
#SBATCH --array=1-2 # Run two tasks to see if they land on different nodes

# ==============================================================================
#  SLURM Environment Diagnostic Script
# ==============================================================================

echo "=== DIAGNOSTIC JOB STARTED: $(date) ==="
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node Hostname: $(hostname)"
echo "Shell: $SHELL"
echo "========================================="
echo ""

echo "--- 1. Checking for Module Init Scripts ---"
echo "Checking for /etc/profile.d/modules.sh..."
if [ -f "/etc/profile.d/modules.sh" ]; then
    echo "    FOUND: /etc/profile.d/modules.sh"
    echo "    Permissions: $(ls -l /etc/profile.d/modules.sh)"
else
    echo "    NOT FOUND: /etc/profile.d/modules.sh"
fi

echo "Checking for /usr/share/modules/init/bash..."
if [ -f "/usr/share/modules/init/bash" ]; then
    echo "    FOUND: /usr/share/modules/init/bash"
    echo "    Permissions: $(ls -l /usr/share/modules/init/bash)"
else
    echo "    NOT FOUND: /usr/share/modules/init/bash"
fi
echo ""

echo "--- 2. Attempting to Source and Load ---"
# Attempt to source the most likely candidate
MODULE_INIT_SCRIPT="/etc/profile.d/modules.sh"

if [ -f "$MODULE_INIT_SCRIPT" ]; then
    echo "Attempting to source $MODULE_INIT_SCRIPT ..."
    # Use a subshell to avoid polluting the main script's environment if it fails
    (
      source "$MODULE_INIT_SCRIPT"
      echo "    Source command executed without error."
      echo "    Attempting 'module --version':"
      module --version 2>&1 || echo "    'module --version' FAILED"
      echo ""
      echo "    Attempting 'module avail':"
      module avail 2>&1 | head -n 10 || echo "    'module avail' FAILED"
    )
else
    echo "$MODULE_INIT_SCRIPT not found, cannot test sourcing."
fi
echo ""

echo "--- 3. Checking PATH variable ---"
echo "$PATH"
echo ""

echo "--- 4. Checking for 'module' command in PATH ---"
which module || echo "'module' is not an executable in the PATH"
echo ""

echo "--- 5. Listing contents of /etc/profile.d/ ---"
ls -l /etc/profile.d/
echo ""


echo "=== DIAGNOSTIC JOB FINISHED: $(date) ==="
