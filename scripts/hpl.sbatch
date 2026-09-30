#!/bin/bash
# netlib HPL 2.3 with the ozdgemm DGEMM hook, 1 rank / 1 GPU.
#   sbatch scripts/hpl.sbatch
#   N=20000 NB=512 CONFIGS="cpu native emu:14 emu:11" sbatch scripts/hpl.sbatch
# CONFIGS entries: cpu | native | emu:<moduli> | auto:<moduli> | tol:<tol> | autotol:<tol>
# Results: results/<jobid>/hpl-<config>.out and summary.csv
#SBATCH --account=ACD110018
#SBATCH --job-name=ozd-hpl
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --exclude=25a-hgpn011,25a-hgpn026,25a-hgpn144

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
REPO=$PWD
OUT=$REPO/results/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
N=${N:-40000}
NB=${NB:-1024}
CONFIGS=${CONFIGS:-"native emu:14 emu:11 autotol:0"}
NVHPC_ROOT=${NVHPC_ROOT:-/work/envstack/apps/nvhpc/26.3/Linux_x86_64/26.3}
export OPAL_PREFIX=$NVHPC_ROOT/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

{
  echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD) N=$N NB=$NB threads=$OMP_NUM_THREADS"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader
  lscpu | grep -E "Model name|^CPU\(s\)"
} | tee "$OUT/env.txt"

echo "config,N,NB,time_s,gflops,residual,passed,gpu_calls,emulated,moduli,h2d_s,gemm_s,d2h_s,cpu_blas_s" > "$OUT/summary.csv"
for cfg in $CONFIGS; do
  mode=${cfg%%:*}; arg=${cfg#*:}; [[ "$arg" == "$cfg" ]] && arg=""
  unset HPL_OZD_MODULI HPL_OZD_TOL
  case $mode in
    cpu|native) export HPL_OZD_MODE=$mode ;;
    emu|auto)   export HPL_OZD_MODE=$mode HPL_OZD_MODULI=$arg ;;
    tol)        export HPL_OZD_MODE=emu HPL_OZD_TOL=$arg ;;
    autotol)    export HPL_OZD_MODE=auto HPL_OZD_TOL=$arg ;;
    *) echo "bad config $cfg"; exit 1 ;;
  esac
  RUN=$OUT/run-${cfg//:/-}
  mkdir -p "$RUN" && cd "$RUN"
  cat > HPL.dat <<DAT
HPLinpack benchmark input file
ozdgemm HW2
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$N           Ns
1            # of NBs
$NB          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
DAT
  log=$OUT/hpl-${cfg//:/-}.out
  "$OPAL_PREFIX/bin/mpirun" -np 1 --bind-to none "$REPO/build/xhpl" > "$log" 2>&1 || echo "xhpl exited $?" >> "$log"
  cd "$REPO"
  # T/V line: WRxxxxx N NB P Q Time Gflops
  read -r t gf < <(awk '/^WR/ {print $6, $7}' "$log" | tail -1) || true
  res=$(awk -F'=' '/\|\|Ax-b\|\|_oo/ {split($2,a," "); print a[1]}' "$log" | tail -1)
  pass=$(grep -c "PASSED" "$log" || true)
  stats=$(awk '/^OZD_HPL mode/ {for(i=1;i<=NF;i++) if($i ~ /^gpu=/) g=substr($i,5); if (match($0,/emulated [0-9]+/)) e=substr($0,RSTART+9,RLENGTH-9); if (match($0,/moduli [0-9]+\.\.[0-9]+/)) m=substr($0,RSTART+7,RLENGTH-7)}
                /^OZD_HPL gflop/ {for(i=1;i<=NF;i++){ if($i ~ /^h2d=/) h=substr($i,5); if($i ~ /^gemm=/) gm=substr($i,6); if($i ~ /^d2h=/) d=substr($i,5); if($i ~ /^cpu_blas=/) c=substr($i,10)}}
                END {print g","e","m","h","gm","d","c}' "$log")
  echo "$cfg,$N,$NB,${t:-},${gf:-},${res:-},$pass,$stats" | tee -a "$OUT/summary.csv"
done
