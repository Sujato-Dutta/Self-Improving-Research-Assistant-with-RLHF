#!/bin/bash
# ==============================================================================
# SLURM Batch Job: Install Python Dependencies on Compute Node
# Strictly runs on compute nodes (cpu_student), NEVER on login node
# ==============================================================================
#SBATCH --job-name=setup_rlhf_env
#SBATCH --partition=cpu_student
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err

echo "=========================================================="
echo "Starting RLHF Environment Setup on Compute Node"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /dgxa_home/se23uari167/Self-Improving-Research-Assistant

# Install dependencies using user site-packages
python3 -m pip install --user --upgrade pip
python3 -m pip install --user transformers trl accelerate scikit-learn scipy tqdm mlflow sentence-transformers faiss-cpu

echo "Verifying installation on compute node:"
python3 -c "
import torch
import transformers
import trl
import sentence_transformers
import faiss
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('TRL:', trl.__version__)
print('All libraries imported successfully!')
"

echo "Setup completed at $(date)"
