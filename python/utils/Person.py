from utils import SleepState
import random
import pandas as pd


class Person:
    def __init__(self, name=''):
        self.states = dict()
        self.name = str(name)

    def add_state(self, ts, current_state=None):
        state = SleepState.Model(ts)

        prev_ts = ts - 1
        if len(self.states.keys()) == 0:
            # state.ProcessS_model.beta_prev = int(random.random() * 1000) % 2
            state.beta_prev = 0
            if current_state is not None:
                state.beta = current_state
            state.B = random.random() / 2 + 0.17
            state.H = random.random() * 1
            state.W = random.random() * 1
            state.xc = random.random() * 3 - 1.5
            state.x = random.random() * 3 - 1.5
            state.calculate()
            self.states[ts] = state
            return

        if prev_ts not in self.states.keys():
            print('TS {}: Cannot find previous state with timestamp {}'.format(ts, prev_ts))
        else:
            prev_state = self.states[prev_ts]
            # print(prev_state)
            state.beta_prev = prev_state.beta
            if current_state is not None:
                state.beta = current_state
            state.H = prev_state.H_next
            state.W = prev_state.W_next
            state.xc = prev_state.xc_next
            state.x = prev_state.x_next

            state.calculate()
            self.states[ts] = state

    def get_state(self, ts):
        if ts not in self.states.keys():
            return None
        else:
            return self.states[ts]

    def to_dataframe(self):
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
            x.append(state.x)
            xc.append(state.xc)
            beta.append(state.beta)
            H.append(state.H)
            dH.append(state.dH)
            B.append(state.B)
            W.append(state.W)
            dW.append(state.dW)
            subjective_alertness.append(state.subjective_alertness)
            sleepness.append(state.sleepness)

        df = pd.DataFrame({'ts': all_ts, 'x': x, 'xc': xc, 'beta': beta, 'H': H, 'dH': dH, 'B': B, 'W': W, 'dW': dW,
                           'subjective_alertness': subjective_alertness, 'sleepness': sleepness})
        return df
