#!/bin/bash
# v1 vs v2 per-stage times (bench --breakdown), alternating, same node.
set -eu
cd /work/u1131128/hw2
OUT=results/$SLURM_JOB_ID; mkdir -p $OUT
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) old=baseline-v1@$(git -C baseline-v1 rev-parse --short HEAD)" | tee $OUT/env.txt
S="--shape 4096,4096,4096 --shape 8192,8192,1024 --shape 32768,32768,1024 --shape 16384,16384,16384"
first=1
for r in 1 2; do for bin in baseline-v1/build/bench build/bench; do
  $bin $S --moduli 14 --reps 10 --warmup 3 --no-native --breakdown |
    awk -v r=$r -v first=$first 'NR==1{if(first)print "round," $0; next}{print r "," $0}' >> $OUT/breakdown.csv
  first=0
done; done
cat $OUT/breakdown.csv
