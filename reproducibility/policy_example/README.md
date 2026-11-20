# Run a new recovery policy in the physical simulator

This optional example executes the independently implemented RGB feedback
recovery policy. The main package's `recompute.py` remains NumPy-only; this
example separately needs the simulator and CPU offscreen rendering.

Install into a Python 3.10 environment:

```bash
python -m pip install -r policy_example/requirements-simulation.txt
```

On Linux, OSMesa must be available (`libosmesa6`). Place the matching
`warmup-port.json` from the reproduction bundle in this example directory, then
run from `reproducibility/`:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 LP_NUM_THREADS=1 \
python policy_example/campaign.py --seed 63 --scenario left_loss \
  --load density150 --method closed_loop --window actual \
  --out /tmp/ae-policy-example
```

Use a new output directory. Seed 63 is a disclosed development example, not an
additional independent evaluation. `--method retarget` runs the frozen position
target reference; `--window 1000ms` executes the registered initial hold. A
successful example adds no new result to the paper's test denominator.

The output contains the full control trace, `uniform_audit` with V/C/P/I/Y and
the verdict, legacy classification for comparison, measured service waits,
native 500 Hz contacts/motion and actual input RGB. It advances the same physical
environment through every stage; it never resets after closing or teleports an
object. Installed robosuite supplies the robot/object assets. No original
workspace, GPU, checkpoint download or external service is used.

## The policy interface

`closed_loop.py:recover(port)` is the new-policy example. It receives:

| Port | Available information or operation |
| --- | --- |
| `robot()` | End-effector positions/rotations, finger encoders and registered robot references |
| `observe()` | Calibrated RGB silhouettes, known CAD fitting inputs and resulting pose estimate |
| `move`, `phase`, `set_rotations` | Physical Cartesian motion/holding and controller orientation targets |
| `table_targets`, `registered`, `registered_rotations` | References saved before the fault |
| `time()`, `gripper_commands` | Elapsed simulator time and the preceding gripper commands |

Contact truth, object ground-truth pose, seed, fault name, load and future
observations are absent from this control port. `native.py` records evaluator
measurements separately. Acquiring/fitting RGB is measured and followed by
actual holding steps; all retries and final verification remain in the trace.

To test a different recovery strategy, replace the body of `recover(port)` and
retain explicit observations/service-step accounting. Return its observation
records and internal pickup-attempt intervals, as in the example. The existing
global executor hook connects it to the unchanged initialization and audit.
Internal pickup retries remain inside one global recovery invocation.

## Task bindings and the default protocol

| Binding | This cooperative-lifting example |
| --- | --- |
| Observation | Four 320-pixel RGB cameras, known calibration/CAD/colors, robot kinematics and encoders |
| Skills/stages | Global lowering, table load/release/settle, clear, hover/descend/close and lift |
| Admission | The realized global call has no keeper prerequisite; other skills must supply their own entry predicate |
| Roles/interface | Intentional table handoff is permitted; the registered table/regrasp cycle is required |
| Terminal test | Two native pad grasps with bounded relative motion, clearance/tilt/height limits and a one-second hold |
| Waiting | Initial measured or prescribed hold, then each measured observation/fit rounded up to 50 ms of physical holding |

The new cohort puts raw terminal completion in C and the required interface in
P, keeping Y identical to the legacy accepted-and-protected result. Original
cohorts retain their historical definitions. P measures declared support/skill
commitments, not a drop or harm certificate. The primary guard uses 20 Hz
samples; the accompanying 500 Hz recording exposes sampling dependence using
the same physical thresholds.

A different task supplies its observations, commands/stages, entry conditions,
role/handoff rules and terminal predicate. The paired initialization, physical
waiting, cumulative audit and cost accounting are the reused procedure; robot
controllers and geometric/task parameters are explicit task bindings.

The later `prospective/README.md` example additionally exposes the supplied CAD,
initial holding target and duration used in the new prospective cohorts. Its
500 Hz guard view is primary; the original example and its historical protocol
remain unchanged.

The `handover/README.md` example executes the later cross-task position-reference
study with the exact original local/retry and global skills. It offers registered,
entry-measured and smooth position holding while retaining one common wrist reference.
Its separately installed simulator command writes a new physical episode; it is
distinct from numerical recomputation of recorded formal trajectories.

`smooth/run_example.py` runs the lifting registered/immediate/smooth reference
comparison with the frozen CAD recovery. `public/README.md` separately documents
the released pretrained Transport policies, their pinned older simulator runtime
and checkpoint download provenance. Native policy completion and measured cost
components remain separate from the lifting protection and full physical costs.
