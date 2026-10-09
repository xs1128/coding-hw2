#!/bin/bash
# Build netlib HPL 2.3 from third_party/hpl-2.3.tar.gz with the ozdgemm hook.
# Result: build/xhpl. Runs on the login node (no GPU needed).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NVHPC_ROOT=${NVHPC_ROOT:-/work/envstack/apps/nvhpc/26.3/Linux_x86_64/26.3}
export OPAL_PREFIX=$NVHPC_ROOT/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export OMPI_CC=gcc   # HPC-X mpicc defaults to nvc

echo "sha256 check"
echo "32c5c17d22330e6f2337b681aded51637fb6008d3f0eb7c277b163fadd612830  $REPO/third_party/hpl-2.3.tar.gz" | sha256sum -c -

make -C "$REPO" -j hpl-shim

HPL=$REPO/third_party/hpl-2.3
rm -rf "$HPL"
tar xzf "$REPO/third_party/hpl-2.3.tar.gz" -C "$REPO/third_party"
patch -d "$HPL" -p1 < "$REPO/hpl/hpl-ozd.patch"
sed -e "s|@HPLDIR@|$HPL|" -e "s|@REPO@|$REPO|g" -e "s|@NVHPC_ROOT@|$NVHPC_ROOT|" \
    "$REPO/hpl/Make.ozd.in" > "$HPL/Make.ozd"
make -C "$HPL" arch=ozd > "$REPO/build/hpl-build.log" 2>&1 || { tail -30 "$REPO/build/hpl-build.log"; exit 1; }
cp "$HPL/bin/ozd/xhpl" "$REPO/build/xhpl"
nm "$REPO/build/xhpl" | grep -w "T cblas_dgemm" && echo "built build/xhpl (cblas_dgemm from ozd_shim)"
