import mujoco
import numpy as np
from environment2.cocel_sip.cocel_sip import SIPAsset
from environment2.cocel_sip.viewer import Viewer
from copy import deepcopy
from math import sin, cos
import mujoco_viewer

class CoCELSIPENV(SIPAsset):
    def __init__(self, lookat, distance, elevation, azimuth):
        super(CoCELSIPENV, self).__init__()
        self.state_dim = 5
        self.action_dim = 1
        self.action_max = 20.0
        self.pos_max = 0.85
        self.model = mujoco.MjModel.from_xml_path('./environment2/mujoco/cocel_sip.xml')
        self.data = mujoco.MjData(self.model)
        self.viewer = None 

        self.eqi_idx = [0, 1, 2, 4]
        self.reg_idx = [3]
        self.viewer = mujoco_viewer.MujocoViewer(model=self.model, data=self.data,
                                width=640, height=480,
                                title='CoCEL_SIP',
                                mode='offscreen',
                                hide_menus=True)
        self.viewer.cam.distance = distance
        self.viewer.cam.lookat = lookat
        self.viewer.cam.elevation = elevation
        self.viewer.cam.azimuth = azimuth  

    def reset(self):
        self.local_step = 1
        self.total_reward = 0
        q = np.zeros(2)
        q[0] = 0.01 * np.random.randn()
        q[1] = np.pi + .01 * np.random.randn()
        qd = .01 * np.random.randn(2)
        self.state = np.concatenate([q, qd])
        self.prev_state = np.concatenate([q, qd])
        obs = self._get_obs()

        return obs, {}

    def step(self, action):

        # RL policy controller
        self.prev_state = self.state.copy()
        self.action_tanh = action / self.action_max

        qpos = np.array([self.state[0], -self.state[1]])
        qvel = np.array([self.state[2], -self.state[3]])
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.userdata[0] = deepcopy(self.action_tanh)
        self.data.userdata[1] = deepcopy(self.total_reward)

        self._do_simulation(action)
        new_obs = self._get_obs()
        reward, terminated = self._get_reward(new_obs, self.action_tanh)
        info = {}
        self.total_reward += reward
        return new_obs, reward, terminated, False, info

    def _get_reward(self, obs, act):
        pos, cos_th, th_dot = obs[0], obs[3], obs[4]
        notdone = np.isfinite(obs).all() and (np.abs(pos) <= self.pos_max)
        notdone = notdone and np.abs(th_dot) < 27.
        r_pos = 0.5 + 0.5 * np.exp(-0.7 * pos ** 2)
        r_act = 0.8 + 0.2 * np.maximum(1 - (act ** 2), 0.0)
        r_angle = 0.5 - 0.5 * cos_th
        r_vel = 0.5 + 0.5 * np.exp(-0.2 * th_dot ** 2)
        reward = r_pos * r_act * r_angle * r_vel
        done = not notdone
        return reward.squeeze(-1), done

    def _get_obs(self):
        cart_vel = (self.state[0] - self.prev_state[0]) / self.sample_time
        ang_vel = (self.state[1] - self.prev_state[1]) / self.sample_time
        return np.array([self.state[0], cart_vel, sin(self.state[1] + np.pi),
                         cos(self.state[1] + np.pi), ang_vel])

    def render(self, lookat, distance, elevation, azimuth):  

        self.viewer.cam.distance = distance
        self.viewer.cam.lookat = lookat
        self.viewer.cam.elevation = elevation
        self.viewer.cam.azimuth = azimuth                   
        mujoco.mj_forward(self.model, self.data)
        img = self.viewer.read_pixels()
        return img

    def _viewer_reset(self):
        self.viewer = mujoco_viewer.MujocoViewer(model=self.model, data=self.data,
                                width=640, height=480,
                                title='CoCEL_SIP',
                                mode='offscreen',
                                hide_menus=True)
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] += 0.3
        # self.viewer.cam.elevation += 35
        # self.viewer.cam.azimuth = 205      

        # mujoco.mj_step(self.model, self.data)



        