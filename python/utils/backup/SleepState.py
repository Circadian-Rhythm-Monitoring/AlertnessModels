from utils import JFK
from utils import ProcessS
from utils import SleepInertia


class SleepState:
    def __init__(self, ts=0, abnormal=False, status=None):
        self.ts = ts
        self.JFK_model = JFK.Model(ts)
        self.ProcessS_model = ProcessS.Model(ts, abnormal)
        self.Inertia_model = SleepInertia.Model(ts)
        self.status = 0
        self.awake = 'awake'

    def init_parameters(self):
        self.ProcessS_model.x = self.JFK_model.x
        self.Inertia_model.B = self.ProcessS_model.B
        self.Inertia_model.H = self.ProcessS_model.H
        self.Inertia_model.beta = self.ProcessS_model.beta

    def calculate(self):
        self.JFK_model.do_work()
        self.ProcessS_model.x = self.JFK_model.x
        self.ProcessS_model.do_work()
        self.Inertia_model.B = self.ProcessS_model.B
        self.Inertia_model.H = self.ProcessS_model.H
        self.Inertia_model.beta = self.ProcessS_model.beta
        self.Inertia_model.do_work()
        if self.ProcessS_model.B == 0:
            self.status = 0
            self.awake = 'awake'
        else:
            self.status = 1
            self.awake = 'sleep'

    def __str__(self):
        ret = 'Sleep State at timestamp {}:\n'.format(self.ts)
        ret += str(self.JFK_model)
        ret += str(self.ProcessS_model)
        ret += str(self.Inertia_model)
        return ret

    def time(self):
        day = self.ts // 24
        hour = self.ts % 24
        return {"D": day, "H": hour}
