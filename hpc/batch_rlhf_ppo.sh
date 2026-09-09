#!/bin/bash
# ==============================================================================
# SLURM Batch Job: PPO RLHF Post-Training with TRL on DGX A100
# Partition: gpu_student
# ==============================================================================
#SBATCH --job-name=RLHF_PPO_Qwen
#SBATCH --output=rlhf_ppo_%j.out
#SBATCH --error=rlhf_ppo_%j.err
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=06:00:00

echo "=========================================================="
echo "Starting PPO-Based RLHF Post-Training on DGX A100"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

nvidia-smi

cd /dgxa_home/se23uari167/Self-Improving-Research-Assistant
export PYTHONPATH=.
export DEVICE=cuda
export USE_BF16=1
export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

python3 -u scripts/run_self_improvement_loop.py

echo "RLHF training completed at $(date)"
