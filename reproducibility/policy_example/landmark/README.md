# Execute independent landmark visual recovery

Install the simulator requirements in the parent policy example. From this folder:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 python run_example.py --out /tmp/ae-landmark-example
```

Use a new output directory. The default is the disclosed healthy development
condition at seed 90. `--scenario left_loss` or `right_loss` executes a real
unilateral grasp fault; `--hold registered --steps 20` selects the other holding
condition. Results and passive 500 Hz recording appear in `OUT/independent/`.
This executable example adds no formal test sample.

`visual_servo.py` receives calibrated RGB color centroids and measured robot
reference grasps. It triangulates landmarks and aligns the handle line, without
the CAD optimizer or the old recovery executor. Its own approach, table leveling,
recenter and lift stages command ordinary physical actions. Initial reference RGB
is task preparation before the fault. Every subsequent visual service and pickup
retry is physically charged. The simulator's object/contact audit never reaches
control. This is a classical visual-control implementation on the same robot
plant, rather than an external pretrained foundation policy.

`protocol.json`, `independent-protocol.json` and the source freezes document the
480-execution position/rotation factorial and separate 240-execution visual
validation. All originals, thresholds and prior results remain unchanged.
