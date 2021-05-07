from utils import SleepState
import random
import pandas as pd


class Person:
    def __init__(self, name=''):
        self.states = dict()
        self.name = str(name)

    def add_state(self, ts, abnormal=False):
        state = SleepState.SleepState(ts, abnormal)

        prev_ts = ts - 1
        if len(self.states.keys()) == 0:
            # state.ProcessS_model.beta_prev = int(random.random() * 1000) % 2
            state.ProcessS_model.beta_prev = 0
            state.ProcessS_model.B = random.random() / 2 + 0.17
            state.ProcessS_model.H = random.random() * 1
            state.Inertia_model.W = random.random() * 1
            state.JFK_model.xc = random.random() * 3 - 1.5
            state.JFK_model.x = random.random() * 3 - 1.5
            state.calculate()
            self.states[ts] = state
            return

        if prev_ts not in self.states.keys():
            print('TS {}: Cannot find previous state with timestamp {}'.format(ts, prev_ts))
        else:
            prev_state = self.states[prev_ts]
            # print(prev_state)
            state.ProcessS_model.beta_prev = prev_state.ProcessS_model.beta
            state.ProcessS_model.H = prev_state.ProcessS_model.H_next
            state.Inertia_model.W = prev_state.Inertia_model.W_next
            state.JFK_model.xc = prev_state.JFK_model.xc_next
            state.JFK_model.x = prev_state.JFK_model.x_next

            state.calculate()
            self.states[ts] = state

    def to_DataFrame(self):
        all_ts = list(self.states.keys())
        all_ts.sort()

        x = []
        xc = []
        beta = []
        H = []
        dH = []
        B = []
        W = []
        dW = []
        subjective_alertness = []
        sleepness = []

        for ts in all_ts:
            state = self.states[ts]
            x.append(state.JFK_model.x)
            xc.append(state.JFK_model.xc)
            beta.append(state.ProcessS_model.beta)
            H.append(state.ProcessS_model.H)
            dH.append(state.ProcessS_model.dH)
            B.append(state.ProcessS_model.B)
            W.append(state.Inertia_model.W)
            dW.append(state.Inertia_model.dW)
            subjective_alertness.append(state.Inertia_model.subjective_alertness)
            sleepness.append(state.Inertia_model.sleepness)

        df = pd.DataFrame({'ts': all_ts, 'x': x, 'xc': xc, 'beta': beta, 'H': H, 'dH': dH, 'B': B, 'W': W, 'dW': dW,
                           'subjective_alertness': subjective_alertness, 'sleepness': sleepness})
        return df
