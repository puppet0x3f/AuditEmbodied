# Shallow-container position-holding extension

This is the frozen runner for the additional 240 physical executions. It needs the
simulation dependencies in `../requirements-simulation.txt` and OSMesa. It is separate
from the NumPy-only `recompute_mechanism_geometry.py` command.

From this directory, with a Python 3.10 simulation environment and OSMesa available:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 LP_NUM_THREADS=1 python campaign.py --phase development --workers 4

MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 LP_NUM_THREADS=1 python campaign.py --phase formal --workers 16
```

Use a fresh copy: the runner refuses to replace existing execution folders. The
formal matrix uses the original geometry seeds 69000–69019, two original executors,
two waits and left/right loss plus healthy sham. Only initial-wait EEF position
references change; wrists, physics, supplied CAD, grippers, recovery code and scoring
thresholds remain fixed. Every actual visual service is physically charged. The
original registered baseline is reused for analysis and is not executed by this matrix.
Runtime-dependent services can change subsequent outcomes; exact recorded results are
reproduced by the standalone numerical entry, not promised by a new policy execution.
