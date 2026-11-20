# Execute the handover holding comparison

This optional example executes physical control with the separately installed
simulator dependencies in `../requirements-simulation.txt`. It does not read the
original workspace, train a model, or replay a saved test outcome. The exact
earlier recovery skills are included under `frozen-source/`; only the first
decision wait's position reference and duration change. Subsequent decisions
retain measured-pose holding and the original rule and global-retry logic.

From the extracted package root, using the simulator environment:

```bash
python policy_example/handover/run_example.py --out /tmp/ae-handover-example
```

The default uses the disclosed development seed 91, the rotation fault,
one-second waiting, smooth position reanchoring and the local/retry policy.
The output folder must be new. The command writes the complete physical trace,
native contacts, reference/torque/sensor diagnostics and independently scored
terminal completion, sampled protection, native-guard protection and full cost.
This example adds no formal test samples.

`--method rule|R3`, `--hold registered|measured_position|smooth_position`,
`--wait-seconds 0.05|1.0` and `--scenario nominal|object_sham|object_rotation`
select other supplied conditions. All holding alternatives use the same
measured wrist reference, gains and preceding gripper signs. Smooth reanchoring
linearly transitions from the last command target to entry-measured position
over a fixed 0.25 seconds; no parameter search is performed.

Passive native audit overhead is measured separately and excluded from the
controller clock. Actual policy/controller service and other original blocking
overruns remain physically executed and charged. Numerical recomputation of the
formal 720 executions is provided by the package root's NumPy-only command;
this optional command generates a new physical episode instead.
