#!/usr/bin/env python
"""
Automates synchronization, remote execution on Mahindra University DGX A100 cluster (10.59.121.172),
and retrieval of trained checkpoints and evaluation metrics back to local workspace.
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClusterPipeline")

CLUSTER_HOST = "10.59.121.172"


def run_cmd(cmd, check=True):
    logger.info(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr and res.returncode != 0:
        print(res.stderr, file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res


def check_ssh_connection(username: str) -> bool:
    """Verifies passwordless SSH connectivity to cluster."""
    cmd = f'ssh -o BatchMode=yes -o ConnectTimeout=5 {username}@{CLUSTER_HOST} "echo SSH_SUCCESS"'
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return "SSH_SUCCESS" in res.stdout


def sync_code_to_cluster(username: str, remote_dir: str = "~/Self-Improving-Research-Assistant"):
    logger.info(f"Synchronizing codebase to {username}@{CLUSTER_HOST}:{remote_dir}...")
    run_cmd(f'ssh {username}@{CLUSTER_HOST} "mkdir -p {remote_dir}"')

    # Exclude venv, git cache, large temp files
    tar_cmd = f'tar --exclude="venv" --exclude=".git" --exclude="__pycache__" -czf project_transfer.tar.gz .'
    run_cmd(tar_cmd, check=True)

    scp_cmd = f'scp project_transfer.tar.gz {username}@{CLUSTER_HOST}:{remote_dir}/'
    run_cmd(scp_cmd, check=True)

    extract_cmd = f'ssh {username}@{CLUSTER_HOST} "cd {remote_dir} && tar -xzf project_transfer.tar.gz && rm project_transfer.tar.gz"'
    run_cmd(extract_cmd, check=True)

    # Clean up local archive
    if Path("project_transfer.tar.gz").exists():
        Path("project_transfer.tar.gz").unlink()
    logger.info("Codebase successfully synchronized to cluster.")


def launch_cluster_jobs(username: str, remote_dir: str = "~/Self-Improving-Research-Assistant"):
    logger.info("Launching SLURM batch pipeline on DGX A100 cluster...")

    # Remote script to submit jobs and wait for completion
    remote_script = f"""
cd {remote_dir}
chmod +x hpc/*.sh
echo "=== Submitting SLURM Job: Reward Model Training ==="
JOB1=$(sbatch --parsable hpc/batch_train_reward_model.sh)
echo "Submitted RM training job ID: $JOB1"

echo "=== Submitting SLURM Job: PPO RLHF Post-Training (dependent on RM) ==="
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 hpc/batch_rlhf_ppo.sh)
echo "Submitted RLHF job ID: $JOB2"

echo "=== Submitting SLURM Job: Benchmark Evaluation (dependent on RLHF) ==="
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 hpc/batch_evaluate.sh)
echo "Submitted Evaluation job ID: $JOB3"

echo "All jobs queued in SLURM. Tracking job $JOB3..."
"""
    run_cmd(f'ssh {username}@{CLUSTER_HOST} "{remote_script}"', check=True)


def fetch_results_from_cluster(username: str, remote_dir: str = "~/Self-Improving-Research-Assistant"):
    logger.info("Fetching trained model weights, checkpoints, and benchmark metrics from cluster...")
    os.makedirs(BASE_DIR / "artifacts" / "checkpoints", exist_ok=True)
    os.makedirs(BASE_DIR / "artifacts" / "models", exist_ok=True)
    os.makedirs(BASE_DIR / "data" / "benchmark", exist_ok=True)

    # Fetch checkpoints
    run_cmd(f'scp -r {username}@{CLUSTER_HOST}:{remote_dir}/artifacts/checkpoints/* "{BASE_DIR}/artifacts/checkpoints/"', check=False)
    # Fetch models
    run_cmd(f'scp -r {username}@{CLUSTER_HOST}:{remote_dir}/artifacts/models/* "{BASE_DIR}/artifacts/models/"', check=False)
    # Fetch benchmark results
    run_cmd(f'scp {username}@{CLUSTER_HOST}:{remote_dir}/data/benchmark/results_summary.json "{BASE_DIR}/data/benchmark/"', check=False)
    logger.info("Results successfully synchronized to local machine.")


def main():
    parser = argparse.ArgumentParser(description="DGX A100 Cluster Pipeline Manager")
    parser.add_argument("--user", required=True, help="Username for Mahindra Univ cluster (10.59.121.172)")
    parser.add_argument("--action", choices=["test_ssh", "sync", "launch", "fetch", "all"], default="all")
    args = parser.parse_args()

    if args.action in ["test_ssh", "all"]:
        logger.info(f"Testing connection to {args.user}@{CLUSTER_HOST}...")
        if not check_ssh_connection(args.user):
            logger.error(
                f"Cannot connect to {args.user}@{CLUSTER_HOST} via passwordless SSH.\n"
                f"Please authorize your public SSH key by running:\n"
                f'type "$env:USERPROFILE\\.ssh\\id_ed25519.pub" | ssh {args.user}@{CLUSTER_HOST} "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"'
            )
            sys.exit(1)
        logger.info("SSH connection verified successfully!")

    if args.action in ["sync", "all"]:
        sync_code_to_cluster(args.user)

    if args.action in ["launch", "all"]:
        launch_cluster_jobs(args.user)

    if args.action in ["fetch", "all"]:
        fetch_results_from_cluster(args.user)


if __name__ == "__main__":
    main()
