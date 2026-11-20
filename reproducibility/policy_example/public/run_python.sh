#!/usr/bin/env bash
set -e
study_dir="$(cd "$(dirname "$0")" && pwd)"
export MUJOCO_PY_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES="${AE_PUBLIC_GPU:-2}"
export PATH="/tmp/ae-public-policy-20261111-env/bin:$PATH"
export LD_LIBRARY_PATH="$study_dir/native-dev/usr/lib/x86_64-linux-gnu:/tmp/ae-public-policy-20261111-env/lib/python3.11/site-packages/mujoco_py/binaries/linux/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
exec /tmp/ae-public-policy-20261111-env/bin/python "$@"
