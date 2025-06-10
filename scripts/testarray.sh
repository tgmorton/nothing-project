#!/bin/bash

# ==============================================================================
#  Advanced SLURM Environment & Filesystem Diagnostic Script
# ==============================================================================
#
#  Purpose: To aggressively debug inconsistent module loading issues in a
#           SLURM array job by assuming nothing about the node's environment.
#
# ==============================================================================

# === SBATCH Directives ===
# Use the partition where jobs have been failing
#SBATCH --job-name=paranoid_env_test
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=../logs/paranoid_test_%A_%a.out # JobID_TaskID
#SBATCH --error=../logs/paranoid_test_%A_%a.err
#SBATCH --array=1-3 # Run 3 tasks to maximize chances of hitting a bad node

# --- Start of Script ---
echo "================================================================"
echo "=== ADVANCED DIAGNOSTIC STARTED: $(date)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node Hostname: $(hostname)"
echo "Executing Shell: $SHELL"
echo "---"

# --- 1. Filesystem & Permissions Deep Dive ---
echo "[STEP 1] Performing deep check for module initialization scripts..."
echo "This checks existence, permissions, and resolves symbolic links."

POSSIBLE_SCRIPTS=(
    "/etc/profile.d/modules.sh"
    "/usr/share/modules/init/bash"
    "/etc/profile.d/lmod.sh"
)
MODULE_INIT_SCRIPT=""

# Loop through potential scripts to find a valid one
for script_path in "${POSSIBLE_SCRIPTS[@]}"; do
    echo " -> Checking path: ${script_path}"
    # Use ls to check existence and permissions. It's more reliable than [ -f ] for NFS.
    ls -ld "${script_path}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "    FOUND: File exists. Details: $(ls -ld "${script_path}")"

        # If it's a symlink, find its ultimate target
        if [ -L "${script_path}" ]; then
            target_path=$(readlink -f "${script_path}")
            echo "    INFO: It's a symbolic link. Target is: ${target_path}"
            echo "    INFO: Target details: $(ls -ld "${target_path}")"
        fi

        # Crucial test: Check if the file is readable by us right now
        if [ -r "${script_path}" ]; then
            echo "    SUCCESS: File is readable. Selecting this one."
            MODULE_INIT_SCRIPT="${script_path}"
            break # Exit the loop since we found a valid script
        else
            echo "    WARNING: File exists but is NOT readable. Continuing search..."
        fi
    else
        echo "    NOT FOUND."
    fi
done
echo "---"


# --- 2. Shell Execution Trace ---
# This is the most critical step. 'set -x' prints every command to stderr
# right before it's executed, showing us exactly where the failure occurs.
echo "[STEP 2] Attempting to source the found script with shell tracing enabled."
echo "Check the .err log for output prefixed with '+'."

if [ -n "$MODULE_INIT_SCRIPT" ]; then
    # Enable command tracing and pipe it to stderr
    set -x
    # THE ACTUAL TEST: Source the script we found and validated
    source "$MODULE_INIT_SCRIPT"
    # Disable command tracing
    set +x
    echo "SUCCESS: The 'source' command completed without a fatal script error."
else
    echo "FATAL: No readable module initialization script was found. Cannot proceed."
    echo "================================================================"
    echo "=== DIAGNOSTIC FAILED: $(date)"
    echo "================================================================"
    exit 1
fi
echo "---"


# --- 3. Post-Source Verification ---
echo "[STEP 3] Verifying that the 'module' command is now available."
# 'type' is a shell builtin that tells you what a command is (alias, function, file)
type module
if [ $? -eq 0 ]; then
    echo "SUCCESS: The 'module' command is now defined as a function."
    echo "Attempting to get module system version..."
    module --version
    echo "---"
    echo "Attempting to list available modules (first 15 lines)..."
    module avail 2>&1 | head -n 15
else
    echo "FAILURE: The 'module' command is still not defined after sourcing."
fi
echo "---"

# --- 4. Full Environment Dump ---
echo "[STEP 4] Dumping all environment variables for review."
env | sort
echo "---"


echo "================================================================"
echo "=== DIAGNOSTIC FINISHED: $(date)"
echo "================================================================"
