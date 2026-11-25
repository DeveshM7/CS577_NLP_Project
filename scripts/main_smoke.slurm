#!/usr/bin/env bash
#SBATCH --job-name=cs577-main-smoke
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
arm="${1:?Specify arm B or C}"
case "$arm" in B|C) ;; *) echo "Expected arm B or C" >&2; exit 2 ;; esac
dataset="${2:?Specify the frozen export directory}"
mode="${3:-smoke}"
case "$mode" in smoke|train) ;; *) echo "Expected mode smoke or train" >&2; exit 2 ;; esac
source_dir="${4:-$PWD}"
seed="${5:-577}"
training="$dataset/train_${arm}.jsonl"
validation="$dataset/validation_pairs.jsonl"
output="$workspace/runs/$SLURM_JOB_ID"
mkdir -p "$(dirname "$output")"
mkdir "$output"

(
    printf 'Arm %s\n' "$arm"
    printf 'Mode %s\n' "$mode"
    printf 'Training seed %s\n' "$seed"
    if [[ -f "$source_dir/git-provenance.txt" ]]; then
        cat "$source_dir/git-provenance.txt"
    else
        git rev-parse HEAD
        git status --short
    fi
    cd "$source_dir"
    sha256sum scripts/main_smoke.slurm scripts/run_main.py scripts/run_pilot.py pyproject.toml uv.lock "$training" "$validation"
    nvidia-smi
) > "$output/provenance.txt"
"$env_dir/bin/python" -u "$source_dir/scripts/run_main.py" "$mode" "$training" "$output" --seed "$seed" 2>&1 | tee "$output/$mode.log"
"$env_dir/bin/python" -u "$source_dir/scripts/run_main.py" reload "$validation" "$output" --seed "$seed" 2>&1 | tee "$output/reload.log"
