# AuditEmbodied reproduction guide

Use the simulator examples to execute new episodes. Use the NumPy evaluators to score recorded paths and regenerate the paper tables when the matching reproduction inputs are available.

## Execute new episodes

Install the simulator dependencies with `python -m pip install -e .` from the repository root, or install `policy_example/requirements-simulation.txt` from this directory in a Python 3.10 environment. The examples use headless OSMesa on Linux. Write each run to a fresh output directory.

| Experiment | Entry point |
| --- | --- |
| Handover holding intervention | [Handover example](policy_example/handover/README.md) |
| RGB feedback and CAD-guided lifting | [Lifting example](policy_example/README.md) |
| Holding duration and geometry | [Prospective example](policy_example/prospective/README.md) |
| Shared-prefix CAD and landmark recovery | [Paired example](policy_example/paired/README.md) and [landmark example](policy_example/landmark/README.md) |
| Smooth position holding | [`smooth/run_example.py`](policy_example/smooth/run_example.py) |
| Budgeted CAD recovery | [Arrival-adaptive example](policy_example/budgeted/README.md) |
| Shallow-container holding | [Geometry extension](policy_example/geometry-position/README.md) |
| Pretrained Transport policies | [Native Transport](policy_example/public/README.md) and [grasp-loss recovery](policy_example/public-recovery/README.md) |

The handover example runs from this checkout. CAD-based examples also need the matching `warmup-port.json` in the respective example directory. Formal campaigns may need the original run inputs specified by their protocols. The Transport adapters use a separate runtime and download their published checkpoints separately.

## Recompute recorded results

Place the matching reproduction bundle's `data/`, `models/` and `reference/` directories beside these scripts. Use Python 3.10–3.12 and install only the requirements below. Run from `reproducibility/`. Choose new output directories for each command.

```bash
python -m pip install -r requirements.txt
python -I recompute.py --out results/base
python -I recompute_followup.py --base-results results/base --out results/followup
python -I recompute_mechanism_geometry.py --out results/mechanism-geometry
```

The base evaluator recomputes the recorded-trajectory audit, CAD and landmark fits, and the base tables. The follow-up evaluator adds the budgeted CAD and public-policy recovery comparisons. The mechanism and geometry evaluator adds the joint-tracking diagnostics and shallow-container extension. Each entry writes verification results with the regenerated tables. The data and reference tables must match the evaluator revision.
