# NVIDIA DGX A100 Training Guide (Mahindra University Supercomputer Lab)

This directory contains production SLURM job submission scripts and instructions tailored specifically for the Mahindra University NVIDIA DGX A100 cluster (`10.59.121.172`), adhering to the official HPC training documentation (`docs/DGX_A100_training_Doc.pdf`).

---

## Cluster Overview & Architecture

- **Cluster Headnode**: `10.59.121.172` (Dell PowerEdge R760xs)
- **Compute Nodes**: `dgxa`, `dgxb` (8x NVIDIA A100-SXM4-40GB per node, NVSwitch 600 GB/s)
- **Multi-Instance GPU (MIG) Slices Available**:
  - `a100_1g.5gb:1` (1 compute slice, 5GB vRAM)
  - `a100_2g.10gb:1` (2 compute slices, 10GB vRAM)
  - `a100_4g.20gb:1` (4 compute slices, 20GB vRAM)
  - `a100:1` (Full A100 40GB vRAM)

---

## SLURM Partitions & Quotas

| Partition | User Category | Max GPUs/MIGs | Max CPUs | Max RAM | Time Limit | Storage (U.2 NVMe) |
|---|---|---|---|---|---|---|
| `gpu_student` | Students | 1 MIG (`a100_1g.5gb:1`) | 32 cores | 256 GB | 7 days | 70 GB |
| `gpu_scholar` | PhD/Scholars | 2 MIGs (`a100_2g.10gb:1`) | 48 cores | 384 GB | 30 days | 100 GB |
| `gpu_faculty` | Faculty | 2 GPUs (`a100:1` or `a100_4g.20gb:1`) | 64 cores | 512 GB | 30 days | 200 GB |

---

## Step 1: Connecting to the Cluster

From the university campus network:
```bash
ssh -X <your_username>@10.59.121.172
```
*Note: Do NOT execute training jobs on the headnode (`dgx-login01`). Always submit via `sbatch` or `srun`.*

---

## Step 2: Enroot PyTorch Container Environment Setup

The Mahindra University DGX A100 uses **Enroot** containerization backed by NVIDIA NGC:

```bash
# 1. Import official PyTorch 23.01 image from NGC
enroot import --output pytorch23.sqsh 'docker://nvcr.io#nvidia/pytorch:23.01-py3'

# 2. Create Enroot container
enroot create --name pytorch_rlhf pytorch23.sqsh

# 3. Verify container is created
enroot list

# 4. Start interactive session with your home directory mounted
enroot start --mount $HOME pytorch_rlhf bash
```

Inside the container:
```bash
cd ~/Self-Improving-Research-Assistant
pip install -r requirements.txt
```

---

## Step 3: Submitting Batch Training Jobs

### 1. Train Reward Model on A100
```bash
sbatch hpc/batch_train_reward_model.sh
```

### 2. Run PPO RLHF Post-Training on A100
```bash
sbatch hpc/batch_rlhf_ppo.sh
```

### 3. Run Held-Out 350-Benchmark Evaluation
```bash
sbatch hpc/batch_evaluate.sh
```

---

## Step 4: Monitoring Jobs

```bash
# Check queue status
squeue -u $USER

# Inspect node status
sinfo

# View live output logs
tail -f rlhf_ppo_*.out

# Cancel job if needed
scancel <job_id>
```

---

## Step 5: Interactive GPU Debugging

To requisition an interactive GPU session for 1 hour on `gpu_student`:
```bash
bash hpc/interactive_gpu.sh
```
Or directly via SLURM:
```bash
srun -N 1 --ntasks-per-node=4 --gres=gpu:a100_1g.5gb:1 --mem=16000 --time=01:00:00 --partition=gpu_student --pty /bin/bash
```
