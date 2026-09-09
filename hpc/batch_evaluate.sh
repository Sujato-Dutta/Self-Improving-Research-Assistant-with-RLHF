#!/bin/bash
# ==============================================================================
# SLURM Batch Job: Held-Out 350-Benchmark Evaluation on DGX A100
# Partition: gpu_student
# ==============================================================================
#SBATCH --job-name=Benchmark_Evaluation
#SBATCH --output=benchmark_eval_%j.out
#SBATCH --error=benchmark_eval_%j.err
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=03:00:00

echo "=========================================================="
echo "Running 350-Question Held-Out Benchmark on A100..."
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

nvidia-smi

cd /dgxa_home/se23uari167/Self-Improving-Research-Assistant
export PYTHONPATH=.
export DEVICE=cuda
export USE_BF16=1

python3 -u scripts/run_self_improvement_loop.py

echo "Benchmark evaluation completed at $(date)"
