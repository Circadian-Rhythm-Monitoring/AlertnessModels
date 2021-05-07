import math
import numpy as np


class Model:
    def __init__(self, ts=0):
        self.ts = ts

        self.x = 0
        self.x_next = 0
        self.xc = 0
        self.xc_next = 0
        self.u = 0.1  # Default update rate (not precise)

        self.dx = 0
        self.dxc = 0

        # Parameters
        self.mu = 0.13
        self.q = 1/3
        self.tau_x = 24.2
        self.k = 0.55
        self.kc = 0.4

    def set_parameters(self, mu=None, q=None, tau_x=None, k=None, kc=None):
        if mu is not None:
            self.mu = mu
        if q is not None:
            self.q = q
        if tau_x is not None:
            self.tau_x = tau_x
        if k is not None:
            self.k = k
        if kc is not None:
            self.kc = kc

    def do_work(self):
        self.dx = np.pi/12 * (self.xc + self.mu * (self.x/3 + 4*self.x**3/3 - 256*self.x**7/105)
                              + (1 - 0.4*self.x) * (1 - self.kc * self.xc) * self.u)
        self.dxc = np.pi/12 * (self.q * self.xc * (1 - 0.4*self.x) * (1 - self.kc * self.xc) * self.u
                               - (24/(0.99727*self.tau_x))**2 * self.x - self.k * self.x * (1 - 0.4*self.x)
                               * (1 - self.kc * self.xc) * self.u)
        self.x_next = self.x + self.dx
        self.xc_next = self.xc + self.dxc

    def __str__(self):
        ret = 'JFK Model:\n'
        ret += '\tX: {}\n\txc: {}\n'.format(self.x, self.xc)
        return ret
