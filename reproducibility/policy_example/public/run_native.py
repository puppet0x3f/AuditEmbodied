"""Portable entry for the original matrix, retaining native float32 absolute actions.

Run in a fresh copy of this directory after installing the separate native runtime.
The frozen author-side files remain unchanged. Output is written to formal/<policy>/.
"""
import numpy as np
import campaign
from controlled_transport import ControlledImageEnv, ControlledSteps


class NativeRollinEnv(ControlledImageEnv):
    def step(self, action):
        index = len(self.actions)
        if self.prefix_actions is not None and index < self.pause_step + self.chunk_steps:
            action = np.asarray(self.prefix_actions[index])
        return super().step(action)


class NativeRollinSteps(ControlledSteps):
    def configure(self, hold, wait, pause_step, prefix_actions):
        if prefix_actions is not None:
            dtype = np.float32 if self.env.absolute_actions else np.float64
            prefix_actions = np.asarray(prefix_actions, dtype=dtype)
        return super().configure(hold, wait, pause_step, prefix_actions)


def factory(seed, cfg, render=True):
    env = NativeRollinEnv(seed, cfg, render)
    env.absolute_actions = bool(cfg.task.abs_action)
    return NativeRollinSteps(env, n_obs_steps=int(cfg.n_obs_steps),
                            n_action_steps=int(cfg.n_action_steps),
                            max_episode_steps=int(cfg.task.env_runner.max_steps))


if __name__ == '__main__':
    campaign.factory = factory
    campaign.main()
