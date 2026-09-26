#!/bin/bash
# Native vs. v0-baseline vs. current version, same node, same job.
# Needs the v0 worktree:  git worktree add baseline-v0 v0-baseline && make -C baseline-v0
#   sbatch scripts/compare.sbatch
#   PRESET=hpl MODULI=14,11 NSYS=0 sbatch scripts/compare.sbatch
#SBATCH --account=ACD110018
#SBATCH --job-name=ozd-compare
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
PRESET=${PRESET:-all}
MODULI=${MODULI:-14,11,8}
NVHPC_ROOT=${NVHPC_ROOT:-/work/envstack/apps/nvhpc/26.3/Linux_x86_64/26.3}

{
  echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
  echo "baseline=$(git -C baseline-v0 rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
} | tee "$OUT/env.txt"

build/test_ozdgemm | tee "$OUT/test.txt"
build/bench --preset "$PRESET" --moduli "$MODULI" --breakdown | tee "$OUT/bench.csv"
baseline-v0/build/bench --preset "$PRESET" --moduli 14 --no-native | tail -n +2 | tee -a "$OUT/bench.csv"

# One Nsight Systems timeline at the HPL-like shape (3 timed calls).
NSYS=$(ls -d "$NVHPC_ROOT"/compilers/bin/nsys 2>/dev/null || true)
if [[ "${NSYS_RUN:-1}" == 1 && -n "$NSYS" ]]; then
  export TMPDIR=$PWD/$OUT/tmp && mkdir -p "$TMPDIR"   # compute nodes forbid /tmp/nvidia
  "$NSYS" profile -t cuda,nvtx,cublas -o "$OUT/hpl-shape" --force-overwrite true \
    build/bench --shape 32768,32768,1024 --moduli 14 --reps 3 --warmup 1 > /dev/null
  "$NSYS" stats -r cuda_gpu_kern_sum,nvtx_sum -f csv -o "$OUT/hpl-shape" "$OUT/hpl-shape.nsys-rep" > /dev/null
fi
