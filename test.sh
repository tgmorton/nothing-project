#!/bin/bash
#SBATCH --job-name=minimal_a5k_test
#SBATCH --partition=general_gpu_a5000
#SBATCH --gres=gpu:1   # Or try gpu:1 if the above test suggested it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # Minimal CPU
#SBATCH --mem=2G             # Minimal RAM
#SBATCH --time=00:05:00      # Short time
#SBATCH --output=logs/test_a5k_%j.out
#SBATCH --error=logs/test_a5k_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Checking GPU:"
nvidia-smi
sleep 45
echo "Minimal test finished."