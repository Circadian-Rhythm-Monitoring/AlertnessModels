class Model:
    def __init__(self, ts=0):
        self.ts = ts

        self.W = 0
        self.beta = 0
        self.H = 0
        self.B = 0
        self.subjective_alertness = 0
        self.sleepness = 0

        self.W_next = 0

        self.dW = 0
        self.A = 0

        # Parameters
        self.tau_w = 0.622
        self.Ac = 0.1333

    def set_parameters(self, tau_w=None, Ac=None):
        if tau_w is not None:
            self.tau_w = tau_w
        if Ac is not None:
            self.Ac = Ac

    def do_work(self):
        self.dW = -(1 - self.beta) * self.W / self.tau_w
        self.A = (1 - self.beta) * (1 + self.Ac - self.H - self.W)
        self.subjective_alertness = 56.18 * (self.A - 0.26) / 0.85 + 45.07
        self.sleepness = 3.83 * (self.B - 0.62) / 0.61 + 2.83

        self.W_next = self.W + self.dW

    def __str__(self):
        ret = 'Inertia Model:\n'
        ret += '\tW: {}\n\tsubjective alertness: {}\n\tsleepness: {}\n'.format(self.W, self.subjective_alertness,
                                                                               self.sleepness)
        return ret