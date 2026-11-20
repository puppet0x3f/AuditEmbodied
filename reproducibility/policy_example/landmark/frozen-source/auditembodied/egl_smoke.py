"""Verify NVIDIA EGL and the four robot RGB cameras in an isolated process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


def verify_egl(output: str | Path, device: int = 0):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    from .l0 import CAMERAS, L0Config, TwoArmL0Env
    from OpenGL import GL

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"schema": "auditembodied.egl_smoke.v1", "host": socket.gethostname(),
              "python": sys.executable, "device": int(device), "passed": False}
    env = None
    try:
        env = TwoArmL0Env(0, L0Config(render=True, expert_mode="continuous", horizon=100))
        env.reset()
        env.bootstrap_handover()
        env.obs = env.raw._get_observations(force_update=True)
        env.sim._render_context_offscreen.gl_ctx.make_current()
        report["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
        report["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        report["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
        report["context_class"] = type(env.sim._render_context_offscreen.gl_ctx).__name__
        report["backend"] = os.environ["MUJOCO_GL"]
        frames = env.camera_frames()
        report["cameras"] = {}
        from imageio.v2 import imwrite

        for camera in CAMERAS:
            frame = frames[camera]
            frame_path = output.parent / f"egl-{camera}.png"
            imwrite(frame_path, frame)
            report["cameras"][camera] = {
                "shape": list(frame.shape), "dtype": str(frame.dtype),
                "std": float(np.std(frame)), "path": str(frame_path),
            }
        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "mujoco", "robosuite", "h5py", "pytest")
        }
        report["passed"] = bool(
            "NVIDIA" in report["gl_vendor"].upper()
            and report["context_class"] == "EGLGLContext"
            and all(value["shape"] == [64, 64, 3] and value["std"] > 1.0
                    for value in report["cameras"].values())
        )
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/egl-smoke.json")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    result = verify_egl(args.output, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
