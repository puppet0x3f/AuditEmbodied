"""Render an entry-time RGB snapshot without blocking the control simulation."""
from concurrent.futures import ThreadPoolExecutor
import copy
import time

import numpy as np

from .l0 import CAMERAS


class EntryRenderer:
    def __init__(self, env):
        self.pool = ThreadPoolExecutor(max_workers=1)
        model = copy.copy(env.sim.model._model)
        options = copy.copy(env.sim._render_context_offscreen.vopt)
        flags = env.sim._render_context_offscreen.scn.flags.copy()
        self.pool.submit(self._setup, model, options, flags).result()

    def _setup(self, model, options, flags):
        from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
        self.sim = MjSim(model)
        self.context = MjRenderContextOffscreen(self.sim, device_id=-1)
        self.context.vopt = options
        self.context.scn.flags[:] = flags

    def submit(self, env):
        # This is an observation-only copy. No state is written to the live sim.
        started = time.perf_counter()
        snapshot = copy.copy(env.sim.data._data)
        return self.pool.submit(self._render, snapshot, started)

    def _render(self, snapshot, started):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.sim.data._data = snapshot
        # Do not forward(): use the exact derived camera/geometry state copied
        # at acquisition, including its native integration timestamp.
        rgb = np.stack([self.sim.render(camera_name=c, width=64, height=64)[::convention].copy() for c in CAMERAS])
        return rgb, {'snapshot_sim_time': float(snapshot.time),
            'render_wall_seconds': time.perf_counter() - started}

    def close(self):
        def release():
            self.context.__del__()
            self.sim._render_context_offscreen = None
        self.pool.submit(release).result()
        self.pool.shutdown()
