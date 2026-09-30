#!/bin/bash
# A/B timing of two library builds on one node, alternating rounds so clock
# or power drift hits both equally. Needs the old build in a worktree:
#   git worktree add baseline-v1 05007ab && make -C baseline-v1 build/bench
#   OLD=baseline-v1 sbatch scripts/abtest.sbatch
#   ARGS="--moduli 14 --reps 10 --warmup 3 --breakdown" sbatch scripts/abtest.sbatch
# Both builds run under identical conditions (no native DGEMM in between:
# a native run heats the GPU and slows the emulated run after it by ~5%).
#SBATCH --account=ACD110018
#SBATCH --job-name=ozd-ab
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
OLD=${OLD:-baseline-v1}
ROUNDS=${ROUNDS:-3}
SHAPES=${SHAPES:-"4096,4096,4096 8192,8192,1024 16384,16384,1024 32768,32768,1024 16384,16384,16384"}
ARGS=${ARGS:-"--moduli 14 --reps 20 --warmup 3"}

{
  echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
  echo "old=$OLD@$(git -C "$OLD" rev-parse --short HEAD) rounds=$ROUNDS args=$ARGS"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
} | tee "$OUT/env.txt"

args=()
for s in $SHAPES; do args+=(--shape "$s"); done
build/bench "${args[@]}" --reps 1 --warmup 1 --no-native > /dev/null   # warm the GPU
first=1
for r in $(seq "$ROUNDS"); do
  for bin in "$OLD/build/bench" build/bench; do
    # shellcheck disable=SC2086
    "$bin" "${args[@]}" $ARGS --no-native |
      awk -v r="$r" -v first="$first" 'NR == 1 { if (first) print "round," $0; next } { print r "," $0 }' \
      >> "$OUT/ab.csv"
    first=0
  done
done
cat "$OUT/ab.csv"
