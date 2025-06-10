#!/bin/bash

# ==============================================================================
#  Targeted Node Diagnostic Script for ssrde-c-403
# ==============================================================================
#
#  Purpose: To gather definitive proof of a filesystem mounting issue on a
#           specific, problematic compute node. This script should only be
#           run after the node has been identified as faulty.
#
# ==============================================================================

# === SBATCH Directives ===
#SBATCH --job-name=node_403_test
#SBATCH --partition=general_gpu_a5000
#SBATCH --nodelist=ssrde-c-403   # CRITICAL: Force the job to run on the target node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=../logs/node_403_test_%j.out
#SBATCH --error=../logs/node_403_test_%j.err

echo "================================================================"
echo "=== TARGETED DIAGNOSTIC STARTED for ssrde-c-403: $(date)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node Hostname: $(hostname)"
echo "---"

# --- 1. Filesystem Mount Check ---
echo "[STEP 1] Checking all mounted filesystems for '/sscf/ssrde-storage'..."
mount | grep 'sscf' || echo "    RESULT: The /sscf filesystem is NOT listed in the mount table."
echo "---"

# --- 2. Direct Filesystem Access Test ---
echo "[STEP 2] Attempting to list the contents of the target directory..."
echo "Running 'ls -ld /sscf/ssrde-storage/'..."
ls -ld /sscf/ssrde-storage/
if [ $? -eq 0 ]; then
    echo "    RESULT: 'ls' command SUCCEEDED. This is unexpected."
else
    echo "    RESULT: 'ls' command FAILED as expected on a faulty node."
fi
echo "---"

# --- 3. Deeper Filesystem/Network Diagnostic ---
echo "[STEP 3] Running 'df -h' to check filesystem disk space usage..."
df -h
echo "---"

# --- 4. Kernel Message Log ---
echo "[STEP 4] Checking for recent kernel-level errors (NFS, network, etc.)..."
# Use dmesg to look for the last 20 lines containing NFS, error, or failure messages
dmesg | grep -i -E "nfs|error|failure|timeout" | tail -n 20 || echo "    No relevant kernel messages found."
echo "---"


echo "================================================================"
echo "=== TARGETED DIAGNOSTIC FINISHED: $(date)"
echo "================================================================"
