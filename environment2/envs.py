import numpy as np
from environment2.cocel_sip_v1 import CoCELSIPENV

class action_space:

    def __init__(self, env) -> None:
        self.shape = (1,)
        self.low = np.array([-env.action_max])
        self.high = np.array([env.action_max])
    
    def sample(self):
        return np.random.uniform(-1., 1., 1) * self.high


class ENV(object):
    def __init__(self, name, lookat, distance, elevation, azimuth, n_history, render_mode="human"):
        self.task_name = name
        if name == "CoCELSIP":
            self.env = CoCELSIPENV(lookat, distance, elevation, azimuth)
        else:
            raise ValueError("[Configuration Error] Choose a valid task name.")
        self.render_mode = render_mode
        self.n_history = n_history
        self.state_dim = self.env.state_dim * self.n_history

        # Create dummy
        self.action_space = action_space(self.env)
        self._max_episode_steps = self.max_step()

        self.action_dim = self.env.action_dim
        self.action_max = self.env.action_max
        eqi_idx = np.array([self.env.eqi_idx] * n_history)
        eqi_idx += self.env.state_dim * np.arange(n_history).reshape([-1, 1])
        self.eqi_idx = eqi_idx.reshape([-1]).tolist()
        reg_idx = np.array([self.env.reg_idx] * n_history)
        reg_idx += self.env.state_dim * np.arange(n_history).reshape([-1, 1])
        self.reg_idx = reg_idx.reshape([-1]).tolist()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state[self.env.state_dim:] = self.state[:-self.env.state_dim]
        self.state[:self.env.state_dim] = next_state
        return next_state, reward, terminated, truncated, info # next_state, reward[0], terminated, truncated, info <-- original

    def reset(self):
        init_obs, _ = self.env.reset()
        self.state = np.concatenate([init_obs]*self.n_history)
        return self.state.reshape([-1]).astype(np.float32).copy(), {}

    def max_step(self):
        return 1000

    def render(self, lookat, distance, elevation, azimuth):
        if self.task_name in ["CoCELSIP", "CoCELQIP"]:
            img = self.env.render(lookat, distance, elevation, azimuth)        
            return img        
        else:
            pass