from utils import Person
import matplotlib.pyplot as plt
import random
import os
import numpy as np


class DaySchedule:
    def __init__(self, day_, awake, sleep):
        self.day = day_
        self.awake = awake
        self.sleep = sleep


def random_schedule(n_days):
    min_awake = 14
    max_awake = 20
    min_sleep = 4
    max_sleep = 10

    schedule = []
    day_count = 0
    while day_count <= n_days:
        day_count += 1
        awake = random.randint(min_awake, max_awake)
        sleep = random.randint(min_sleep, max_sleep)
        day_schedule = DaySchedule(day_count, awake, sleep)
        schedule.append(day_schedule)
    return schedule


def random_state(beta, B, count):
    hm = 0.67
    lm = 0.17

    if beta == 0:
        diff = hm - B
        rd = random.random()
        if rd <= (0.5 - diff)**2 and (count is None or count >= 5):
            return 1
        else:
            return 0
    elif beta == 1:
        diff = B - lm
        rd = random.random()
        if diff <= 0:
            return 0
        elif rd >= np.sqrt(diff + 0.5) and (count is None or count >= 4):
            return 0
        else:
            return 1


if __name__ == '__main__':
    start_ts = 0
    end_ts = 200

    ignore_starting_days = 2

    p1 = Person.Person('test_subject')

    prev_B = None
    prev_beta = None
    state_count = None
    for ts in range(start_ts, end_ts):
        day = ts // 24
        hour = ts % 24

        if prev_B is not None and prev_beta is not None:
            rd_beta = random_state(prev_beta, prev_B, state_count)
            if rd_beta == prev_beta:
                if state_count is None:
                    state_count = 1
                else:
                    state_count += 1
            else:
                state_count = 0
            p1.add_state(ts, current_state=rd_beta)
        else:
            p1.add_state(ts)
        prev_B = p1.get_state(ts).B
        prev_beta = p1.get_state(ts).beta

    df = p1.to_dataframe()
    df.to_csv('data/data.csv', index=False)
    print(df.head(10))

    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    vars = ['x', 'beta', 'H', 'B', 'W', 'sleepness', 'subjective_alertness']

    for var in vars:
        plt.figure()
        plt.plot(df['ts'], df[var], label=var)
        plt.legend()
        plt.savefig('plots/' + var + '.jpg')
        plt.close()

    plt.figure()
    plt.plot(df['ts'], df['beta'], label='beta')
    plt.plot(df['ts'], df['B'], label='B')
    plt.legend()
    plt.show()
