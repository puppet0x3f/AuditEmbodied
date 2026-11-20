# Execute a shared-prefix holding condition

Install the simulator requirements in the parent policy example. From this folder:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 python run_example.py --out /tmp/ae-paired-example
```

Use a new output directory. The default is the disclosed healthy development seed 90,
landmark visual recovery, measured-position holding and a one-second initial wait.
`--method closed_loop` selects the frozen CAD feedback executor. `--hold registered`,
`measured_rotation` or `measured_pose` selects the other position/rotation combinations;
`--steps 1` selects 0.05 seconds. `--scenario left_loss` or `right_loss` executes a real
grasp fault. The seed, holding target and duration change the experimental condition;
all gains, models and audit thresholds stay fixed.

The source and `protocol.json` are the new shared runner used for the formal paired
comparison. Both executors receive RGB and robot measurements. Their original perception
and recovery code is byte-identical to the preceding frozen implementations. Full costs
include measured visual service and every retry. Evaluator-only target diagnostics are
appended after execution and never enter a policy observation port.

`--method fixed_touchdown` selects the predeclared placement probe. It changes the CAD
executor's two initial lowering targets to 0.09 m strokes at 30 steps each, while retaining
its actual RGB services and later control. It is a mechanism probe, with formal execution
conditional on the gate specified in `protocol.json`. Its new executions are counted
separately from the original paired baseline.

The optional example requires the simulator and adds no formal policy sample. The main
NumPy-only reproduction command independently evaluates recorded trajectories without
running this example.
