import numpy as np


class Model:
    def __init__(self, ts=0):
        self.ts = ts

        # JFK Model
        self.x = 0
        self.x_next = 0
        self.xc = 0
        self.xc_next = 0
        self.u = 0.1  # Default update rate (not precise)
        self.dx = 0
        self.dxc = 0
        # JFK Parameters
        self.mu = 0.13
        self.q = 1 / 3
        self.tau_x = 24.2
        self.k = 0.55
        self.kc = 0.4

        # ProcessS Model
        self.beta = 0
        self.beta_prev = 0
        self.beta_calc = 0
        self.H = 0
        self.H_next = 0
        self.dH = 0
        self.B = 0
        # ProcessS Parameters
        self.Hm = 0.67
        self.Lm = 0.17
        self.tau_r = 18.2
        self.tau_d = 4.2
        self.Ac = 0.1333

        # Intertia Model
        self.W = 0
        self.subjective_alertness = 0
        self.sleepness = 0
        self.W_next = 0
        self.dW = 0
        self.A = 0
        # Intertia Parameters
        self.tau_w = 0.622

        self.status = 0
        self.awake = 'awake'

    def calculate_jfk(self):
        self.dx = np.pi / 12 * (self.xc + self.mu * (self.x / 3 + 4 * self.x ** 3 / 3 - 256 * self.x ** 7 / 105)
                                + (1 - 0.4 * self.x) * (1 - self.kc * self.xc) * self.u)
        self.dxc = np.pi / 12 * (self.q * self.xc * (1 - 0.4 * self.x) * (1 - self.kc * self.xc) * self.u
                                 - (24 / (0.99727 * self.tau_x)) ** 2 * self.x - self.k * self.x * (1 - 0.4 * self.x)
                                 * (1 - self.kc * self.xc) * self.u)
        self.x_next = self.x + self.dx
        self.xc_next = self.xc + self.dxc

    def calculate_ps(self):
        self.B = self.H - self.Ac * self.x

        if self.B >= self.Hm:
            self.beta_calc = 1
        elif self.B <= self.Lm:
            self.beta_calc = 0
            self.beta = 0
        else:
            self.beta_calc = self.beta_prev

        if self.beta == 1:
            self.dH = - self.H / self.tau_d
        elif self.beta == 0:
            self.dH = (1 - self.H) / self.tau_r

        self.H_next = self.H + self.dH

    def calculate_inertia(self):
        self.dW = -(1 - self.beta) * self.W / self.tau_w
        self.A = (1 - self.beta) * (1 + self.Ac - self.H - self.W)
        self.subjective_alertness = 56.18 * (self.A - 0.26) / 0.85 + 45.07
        self.sleepness = 3.83 * (self.B - 0.62) / 0.61 + 2.83
        self.W_next = self.W + self.dW

    def calculate(self):
        self.calculate_jfk()
        self.calculate_ps()
        self.calculate_inertia()
        if self.B == 0:
            self.status = 0
            self.awake = 'awake'
        else:
            self.status = 1
            self.awake = 'sleep'

    def __str__(self):
        ret = 'Sleep State at timestamp {}:\n'.format(self.ts)
        ret += str(self.subjective_alertness)
        return ret

    def time(self):
        day = self.ts // 24
        hour = self.ts % 24
        return {"D": day, "H": hour}
