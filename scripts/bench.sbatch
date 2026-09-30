#!/bin/bash
# One-GPU benchmark job on a nano4 H200 compute node.
#   sbatch scripts/bench.sbatch                       # default: tests + all presets
#   BENCH_ARGS="--preset hpl --moduli 14,11" sbatch scripts/bench.sbatch
# Results land in results/<jobid>/ tagged with host and GPU.
#SBATCH --account=ACD110018
#SBATCH --job-name=ozd-bench
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --exclude=25a-hgpn011,25a-hgpn026,25a-hgpn144

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OUT=results/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

{
  echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
} | tee "$OUT/env.txt"

build/test_ozdgemm | tee "$OUT/test.txt"
build/bench ${BENCH_ARGS:---preset all --moduli 14,11,8 --breakdown} | tee "$OUT/bench.csv"
if [[ "${ADAPTIVE_SWEEP:-0}" == 1 ]]; then
  build/test_ozdgemm --adaptive | tee "$OUT/adaptive.csv"
fi
