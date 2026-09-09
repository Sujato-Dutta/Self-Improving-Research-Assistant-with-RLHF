#!/bin/bash
# Requisitions an interactive session on Mahindra University DGX A100 (gpu_student)
echo "Requisitioning 1-hour interactive shell with 1x A100 MIG slice (5GB) and 4 CPU cores..."
srun -N 1 --ntasks-per-node=4 --gres=gpu:a100_1g.5gb:1 --mem=16000 --time=01:00:00 --partition=gpu_student --pty /bin/bash
