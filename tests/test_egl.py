"""Run explicitly on zh-server with AUDITEMBODIED_TEST_EGL=1."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from auditembodied.continuous import TrajectoryConfig, _collect_job


@pytest.mark.skipif(os.environ.get("AUDITEMBODIED_TEST_EGL") != "1", reason="requires NVIDIA EGL")
def test_egl_images_survive_context_replacement_and_worker_reuse():
    cfg = TrajectoryConfig()
    jobs = [(0, "nominal", cfg)] * 4
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        episodes = list(pool.map(_collect_job, jobs))
    reference = episodes[0]
    for episode in episodes:
        assert episode.evaluator_metadata["success"]
        np.testing.assert_array_equal(episode.rgb, reference.rgb)
        np.testing.assert_array_equal(episode.proprio, reference.proprio)
    # Keep native camera previews available for this graphics regression.
    from imageio.v2 import imwrite
    from pathlib import Path
    output = Path("outputs/egl-context-replay")
    output.mkdir(parents=True, exist_ok=True)
    for view, frame in enumerate(reference.rgb[37]):
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 128
        imwrite(output / f"camera-{view}.png", frame)
