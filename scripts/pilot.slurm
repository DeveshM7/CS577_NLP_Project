#!/usr/bin/env bash
#SBATCH --job-name=cs577-pilot
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
workspace="${SCRATCH:?Scholar scratch is required}/cs577"
export HF_HOME="${HF_HOME:-$workspace/hf}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
env_dir="${UV_PROJECT_ENVIRONMENT:-$workspace/pilot-venv}"
dataset="${1:-artifacts/pilot.jsonl}"
output="${2:-$workspace/runs/$SLURM_JOB_ID}"
mkdir -p "$(dirname "$output")"
mkdir "$output"

{
    git rev-parse HEAD
    git status --short
    sha256sum scripts/pilot.slurm scripts/run_pilot.py pyproject.toml uv.lock "$dataset"
    nvidia-smi
} > "$output/provenance.txt"
"$env_dir/bin/python" -u scripts/run_pilot.py infer "$dataset" "$output" 2>&1 | tee "$output/inference.log"
"$env_dir/bin/python" -u scripts/run_pilot.py smoke "$dataset" "$output" 2>&1 | tee "$output/smoke.log"
