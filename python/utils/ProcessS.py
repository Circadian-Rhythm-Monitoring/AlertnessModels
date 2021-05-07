class Model:
    def __init__(self, ts=0, abnormal=False):
        self.ts = ts
        self.beta = 0
        self.beta_prev = 0
        self.H = 0
        self.x = 0

        self.H_next = 0

        self.dH = 0
        self.B = 0

        # Parameters
        self.Hm = 0.67
        self.Lm = 0.17
        self.tau_r = 18.2
        self.tau_d = 4.2
        self.Ac = 0.1333

        self.abnormal = abnormal
        self.abnormal_count = 0

    def set_parameters(self, Hm=None, Lm=None, tau_r=None, tau_d=None, Ac=None):
        if Hm is not None:
            self.Hm = Hm
        if Lm is not None:
            self.Lm = Lm
        if tau_r is not None:
            self.tau_r = tau_r
        if tau_d is not None:
            self.tau_d = tau_d
        if Ac is not None:
            self.Ac = Ac

    def do_work(self):
        self.B = self.H - self.Ac * self.x

        if self.B >= self.Hm:
            if self.abnormal:
                self.abnormal_count += 1
            else:
                self.beta = 1
        elif self.B <= self.Lm:
            if self.abnormal:
                self.abnormal_count += 1
            else:
                self.beta = 0
        else:
            # print('3', self.ts)
            self.beta = self.beta_prev

        if self.beta == 1:
            self.dH = - self.H / self.tau_d
        elif self.beta == 0:
            self.dH = (1 - self.H) / self.tau_r

        self.H_next = self.H + self.dH

    def __str__(self):
        ret = 'ProcessS Model:\n'
        ret += '\tbeta: {}\n\tH: {}\n\tB: {}\n'.format(self.beta, self.H, self.B)
        return ret
