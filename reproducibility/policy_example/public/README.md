# Released native Transport policies

This separate Linux/CUDA example physically executes the original released
Diffusion Policy CNN or LSTM-GMM Transport checkpoint. It is independent of the
main package's NumPy-only recorded-trajectory evaluation and of the robosuite
1.5.2 lifting examples. No training or checkpoint substitution occurs.

The official policy source is Diffusion Policy revision
`5ba07ac6661db573af695b419a7947ecb704690f`. From this example directory, obtain the
pinned source from upstream:

```bash
mkdir -p official-source/diffusion_policy-main
curl -fL https://codeload.github.com/real-stanford/diffusion_policy/tar.gz/5ba07ac6661db573af695b419a7947ecb704690f \
  | tar -xz --strip-components=1 -C official-source/diffusion_policy-main
```

`formal-freeze.json` records source provenance, checkpoint URLs, sizes and SHA256
digests. The checkpoint downloader retrieves the approximately 5.3 GB diffusion
and 413 MB LSTM weights separately.

Use a separate Python 3.11 environment. The recorded deployment uses the versions
in `requirements-native.txt`, the released robosuite fork at
`277ab9588ad7a4f4b55cf75508b44aa67ec171f0`, and PyTorch3D 0.7.0's Python transforms.
Linux needs OSMesa and OpenGL development headers/runtime libraries. After creating
and activating that environment:

```bash
python -m pip install -r requirements-native.txt
python -m pip install 'robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz'
PYTORCH3D_NO_EXTENSION=1 python -m pip install 'pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.0.zip'
python -m pip install --no-deps --ignore-requires-python free-mujoco-py==2.1.6
python download_checkpoints.py
```

Install the bundled free MuJoCo binding last; it supplies the native MuJoCo 2.1
runtime used by this older benchmark. The original released fork should not be
replaced with modern robosuite, which changes the task/controller implementation.

From a **fresh copy** of this example directory:

```bash
AE_PUBLIC_PYTHON=python AE_PUBLIC_GPU=0 bash run_native.sh --policy diffusion_policy_cnn
AE_PUBLIC_PYTHON=python AE_PUBLIC_GPU=0 bash run_native.sh --policy lstm_gmm
```

Each command writes 180 new raw files to `formal/<policy>/` and refuses to
overwrite existing outcomes. It runs the predeclared 20-seed native unpaused
baseline plus the eight boundary/reference/wait alternatives. Reducing the batch
size is a new example protocol, rather than an exact rerun of the recorded cohort.
The 700 policy actions remain intact; waits add physical control steps. Native
completion is payload and trash in their required bins, without lifting guards.

`run_native.py` retains the original executed roll-in and committed chunk,
including float32 absolute OSC commands for diffusion. The frozen author-side
`campaign.py`, `resume_diffusion.py`, source digests and original environment-specific
`run_python.sh` remain unchanged. Use the portable `run_native.sh` above, rather
than that historical author wrapper. The entry checks complete prefix physics,
RGB histories and policy RNG/recurrent state against each policy's own baseline.
Cross-policy state equality is not required.

Historical interruption logs and the prefix timing note are stored with the
original runs. The legacy observable-dictionary timer includes operations
besides camera acquisition.
Report physical control and actual batch-query times as separate components;
these synchronous native costs are not complete physical deployment latency.

Installing this optional runtime is unnecessary for independently recomputing all
paper tables. Native input histories, predicates, actual actions, timing components
and prefix state checks are in the main offline data projection.
