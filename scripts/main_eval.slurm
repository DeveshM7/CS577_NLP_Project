#!/usr/bin/env bash
#SBATCH --job-name=cs577-eval
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
workspace="${SCRATCH:?Scholar scratch is required}/cs577"
export HF_HOME="${HF_HOME:-$workspace/hf}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
env_dir="${UV_PROJECT_ENVIRONMENT:-$workspace/pilot-venv}"
arm="${1:?Specify arm A, B, C, or D}"
dataset="${2:?Specify the frozen test JSONL}"
source_dir="${3:?Specify the source snapshot directory}"
adapter_args=()
case "$arm" in
    A) ;;
    B|C|D) adapter_args=(--adapter "${4:?Specify the trained adapter directory}") ;;
    *) echo "Expected arm A, B, C, or D" >&2; exit 2 ;;
esac
output="$workspace/runs/$SLURM_JOB_ID"
mkdir "$output"
cd "$source_dir"

{
    printf 'Arm %s\n' "$arm"
    cat git-provenance.txt
    sha256sum scripts/main_eval.slurm scripts/run_main.py scripts/run_pilot.py pyproject.toml uv.lock "$dataset"
    if [[ ${#adapter_args[@]} -gt 0 ]]; then
        sha256sum "${adapter_args[1]}/adapter_config.json" "${adapter_args[1]}/adapter_model.safetensors"
    fi
    nvidia-smi
} > "$output/provenance.txt"
"$env_dir/bin/python" -u scripts/run_main.py test "$dataset" "$output" --arm "$arm" "${adapter_args[@]}" 2>&1 | tee "$output/inference.log"
