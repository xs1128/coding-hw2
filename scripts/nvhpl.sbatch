#!/bin/bash
# NVIDIA HPL (HPC-Benchmarks container) vs. our netlib HPL + ozdgemm shim,
# same job, same node, 1 rank / 1 H200.
#   SIF=/path/hpc-benchmarks-26.02.01.sif sbatch scripts/nvhpl.sbatch
# NV_RUNS entries: <name>:<N>:<NB>[:<emulation mode>]  (mode 2 = Ozaki-II)
# OZD_CONFIGS: as CONFIGS in hpl.sbatch, run at N=$OZD_N with our build/xhpl
# (empty: NVIDIA runs only). The reported comparison uses H200.
# Results: results/<jobid>/nv-<name>.out, hpl-<config>.out, summary.csv
#SBATCH --account=ACD110018
#SBATCH --job-name=nv-hpl
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --exclude=25a-hgpn011,25a-hgpn026,25a-hgpn144

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
REPO=$PWD
OUT=$REPO/results/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
SIF=${SIF:-/work/u1131128/hw2-sif/hpc-benchmarks-26.02.01.sif}
NV_RUNS=${NV_RUNS:-"n50k:50000:1024 n50k-emu2:50000:1024:2 n131k:131072:1024"}
OZD_N=${OZD_N:-50000}
OZD_CONFIGS=${OZD_CONFIGS-"native emu:14"}

{
  echo "host=$(hostname) date=$(date -Is) commit=$(git rev-parse --short HEAD) cpus=${SLURM_CPUS_PER_TASK:-?}"
  echo "sif=$SIF"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,memory.total --format=csv,noheader
  lscpu | grep -E "Model name|^CPU\(s\)"
} | tee "$OUT/nv-env.txt"

nv_dat() {  # N NB -> HPL.dat on stdout (NVIDIA's HPL-H200-1GPU.dat with N, NB replaced)
  cat <<DAT
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
$1           Ns
1            # of NBs
$2           NBs
1            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
2 8          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
3 2          BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1 0          DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
192          swapping threshold
1            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
DAT
}

for run in $NV_RUNS; do
  IFS=: read -r name n nb emu <<< "$run"
  RUN=$OUT/nvrun-$name
  mkdir -p "$RUN"
  nv_dat "$n" "$nb" > "$RUN/HPL.dat"
  envs=()
  [[ -n "${emu:-}" ]] && envs=(--env "HPL_EMULATE_DOUBLE_PRECISION=$emu")
  log=$OUT/nv-$name.out
  echo "== $run" | tee "$log"
  apptainer exec --nv "${envs[@]}" -B "$RUN:/dat" "$SIF" \
    mpirun -np 1 --bind-to none /workspace/hpl.sh --dat /dat/HPL.dat --no-multinode \
    >> "$log" 2>&1 || echo "nv xhpl exited $?" >> "$log"
  grep -E "^WR|WARNING|PASSED|FAILED" "$log" || true
done

# Our build on the same node (OZD_CONFIGS="" skips it).
if [[ -n "$OZD_CONFIGS" ]]; then
  N=$OZD_N CONFIGS="$OZD_CONFIGS" SLURM_SUBMIT_DIR=$REPO bash "$REPO/scripts/hpl.sbatch"
fi
python3 "$REPO/scripts/nvhpl_summary.py" "$OUT"
