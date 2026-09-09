#!/bin/bash
# ==============================================================================
# SLURM Batch Job: Full Self-Improving RLHF Pipeline on NVIDIA DGX A100
# Target Partition: gpu_student (Mahindra University DGX Cluster)
# Runs fully detached on compute nodes so you can close your laptop
# ==============================================================================
#SBATCH --job-name=RLHF_Pipeline
#SBATCH --output=rlhf_pipeline_%j.out
#SBATCH --error=rlhf_pipeline_%j.err
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=06:00:00

echo "=========================================================="
echo "Starting Full Self-Improving RLHF Pipeline on DGX A100"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

# Display allocated GPU
nvidia-smi

cd /dgxa_home/se23uari167/Self-Improving-Research-Assistant

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=.
export DEVICE=cuda
export USE_BF16=1
export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

echo "Executing run_self_improvement_loop.py..."
python3 -u scripts/run_self_improvement_loop.py

echo "=========================================================="
echo "Pipeline execution finished at $(date)"
echo "Checkpoints saved to: artifacts/checkpoints/"
echo "Evaluation metrics saved to: data/benchmark/results_summary.json"
echo "=========================================================="
