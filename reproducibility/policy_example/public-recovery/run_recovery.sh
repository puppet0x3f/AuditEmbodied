#!/usr/bin/env bash
set -e
native_dir="$(cd "$(dirname "$0")" && pwd)"
native_python="${AE_PUBLIC_PYTHON:-python}"
export MUJOCO_PY_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES="${AE_PUBLIC_GPU:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
native_bundle="$($native_python -c "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('mujoco_py').origin).parent/'binaries/linux/mujoco210/bin')")"
export LD_LIBRARY_PATH="$native_bundle:${LD_LIBRARY_PATH:-}"
exec "$native_python" "$native_dir/run_recovery.py" "$@"
