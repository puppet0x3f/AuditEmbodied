# Execute the prospective holding/geometry policy

Install the simulator requirements in the parent policy example. From this folder:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 python run_example.py --geometry tray --hold registered --steps 1 --out /tmp/ae-prospective-example
```

Use a new output directory. The default runs a healthy development condition with
RGB feedback; `--scenario left_loss` or `right_loss` runs an actual fault. The
command produces the complete physical trace and passive 500 Hz recording. This
example is separate from the NumPy-only evaluation and adds no independent test
sample to the paper.

`campaign.py` supplies the registered object, fault, pose-holding intervention and
physical execution. `closed_loop.py` receives only RGB fits, robot kinematics and
registered references. `geometry.py` binds either the original pot CAD or the
shallow rectangular container CAD; no simulator state is supplied to fitting.
Both bodies weigh 1.818 kg. Native contacts and object motion remain outside the
controller. Existing gains, guards, gripper force and outcome thresholds are used.

The complete formal campaigns and their seed ranges are described in
`protocol.json`. Their raw research records remain in the workspace; the review
package independently recomputes projected records without running a simulator.
