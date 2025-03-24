import numpy as np
from math import sin, cos

class SIPAsset():
    def __init__(self):

        # Physical parameters
        self.g = 9.81
        self.l = 0.38168421 #0.56
        self.m = 0.715
        self.a = 0.147
        self.I = 0.052081613928
        self.c = 0.00154756
        self.v_max = 1.3
        self.sample_time = 0.02
        self.frame_skip = 10
        self.step_size = self.sample_time / self.frame_skip
        assert self.sample_time % self.step_size == 0

    def _do_simulation(self, a):
        # RK4
        for i in range(self.frame_skip):
            xd1 = self._get_state_dot(self.state, a)
            xd2 = self._get_state_dot(self.state + (self.step_size / 2) * xd1, a)
            xd3 = self._get_state_dot(self.state + (self.step_size / 2) * xd2, a)
            xd4 = self._get_state_dot(self.state + self.step_size * xd3, a)
            xd = (xd1 + 2 * xd2 + 2 * xd3 + xd4) / 6
            self.state += self.step_size * xd

    def set_state(self, qpos, qvel):
        self.state = np.concatenate((qpos, qvel))

    def _get_state_dot(self, state, a):
        pos, th, pos_d, th_d = state
        if pos_d >= self.v_max and a >= 0:
            pos_dd = 0.0
        elif pos_d <= -self.v_max and a <= 0:
            pos_dd = 0.0
        else:
            pos_dd = a[0]
        th_dd = (self.m * self.g * self.l * sin(th) - self.c * th_d -
                 self.m * self.l * cos(th) * pos_dd) / (self.I + self.m * self.l ** 2)
        x_d = np.array([pos_d, th_d, pos_dd, th_dd])
        return x_d