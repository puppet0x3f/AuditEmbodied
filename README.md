# AuditEmbodied

AuditEmbodied evaluates recovery from physical faults in simulated bimanual manipulation. It compares recovery methods from matched fault states, advances the robot and object through the prescribed wait, and measures the full trajectory from the first decision through retries and final verification. The benchmark covers bimanual handover and cooperative lifting in MuJoCo and robosuite.

The holding controller is part of the recovery protocol. In the cooperative-lifting experiments, changing the holding reference at the same waiting duration reversed the ranking of frozen CAD and landmark recovery methods on protected completion. The audit reports terminal completion, support throughout the execution, and recovery cost separately.

## Install and test

Use Python 3.10 on Linux for the validated simulator setup. Install the OSMesa runtime (`libosmesa6` on Debian or Ubuntu) for headless RGB rendering.

```bash
git clone https://github.com/puppet0x3f/AuditEmbodied.git
cd AuditEmbodied
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest

ae-smoke --output-dir outputs/smoke
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -m pytest -q
```

Add `--render` to `ae-smoke` to check the four-camera RGB observations. Tests that use PyTorch, NVIDIA EGL, or historical run files skip when their prerequisites are absent.

## Run a recovery trial

The handover example runs a rotation fault with a one-second wait and smooth position holding. It writes a new control trace, native contact samples and `example-audit.json` with completion, protection and elapsed simulated time.

```bash
python reproducibility/policy_example/handover/run_example.py \
  --out "$PWD/outputs/handover-example"
```

Use a fresh output directory on each run. Pass `--hold registered`, `--hold measured_position` or `--hold smooth_position` to select the position reference. `--wait-seconds` accepts `0.05` and `1.0`.

For cooperative lifting, run a nominal physical rollout with `python -m auditembodied.l4_lift --output outputs/lifting-nominal.json`. The executable recovery examples and their prerequisites are listed in the [reproduction guide](reproducibility/README.md).

## Source layout

| Path | Contents |
| --- | --- |
| [`auditembodied/`](auditembodied/) | Simulation, fault injection, recovery policies, training and trajectory audits |
| [`tests/`](tests/) | Tests for physical control, policies, scoring and trace replay |
| [`experiments/`](experiments/) | Original campaign, diagnostic and analysis scripts grouped by run |
| [`reproducibility/`](reproducibility/) | Recorded-trajectory evaluators, frozen simulator examples and public-policy adapters |

Additional installed commands include `ae-l0` for the handover benchmark, `ae-trajectory` for trajectory collection, `ae-l1` and `ae-l1-calibrate` for visual verifier training and calibration, and `ae-l2` and `ae-l2-ablation` for temporal evaluation. Each accepts `--help`. The optional VLM scripts require their own PyTorch, Transformers and PEFT environment and a local checkpoint specified with `--model-path`.

## Experiment data

This repository ships source and experiment configurations. Recorded trajectories, saved checkpoints, reference tables, and generated artifacts are stored separately. The smoke test and handover example above generate their own observations. Recomputing recorded results requires the matching inputs under `reproducibility/data/`, `reproducibility/models/`, and `reproducibility/reference/`. Historical scripts in `experiments/` expect the inputs from their original run, often alongside the script or under `outputs/`.

The AgentChord-derived graph runtime retains its original attribution and Apache-2.0 terms. See [AgentChord notice](third_party/AgentChord-NOTICE.md) and [license](third_party/AgentChord-LICENSE).
