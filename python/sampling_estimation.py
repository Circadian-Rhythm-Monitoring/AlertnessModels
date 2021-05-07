import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class DataType:
    def __init__(self, x, xc, B, H, W, beta):
        self.x = x
        self.xc = xc
        self.B = B
        self.H = H
        self.W = W
        self.beta = beta

    def copy(self):
        return DataType(x=self.x, xc=self.xc, B=self.B, W=self.W, H=self.H, beta=self.beta)


class ParameterType:
    def __init__(self, mu=0.13, q=1/3, tau_x=24.2, k=0.55, kc=0.4,
                 Hm=0.67, Lm=0.17, tau_r=18.2, tau_d=4.2,
                 Ac=0.1333, tau_w=0.622):
        self.mu = mu
        self.q = q
        self.tau_x = tau_x
        self.k = k
        self.kc = kc
        self.Hm = Hm
        self.Lm = Lm
        self.tau_r = tau_r
        self.tau_d = tau_d
        self.Ac = Ac
        self.tau_w = tau_w

    def random_parameter(self):
        random.seed(1234)

        # JFK Parameters
        # self.mu = random.uniform(0.13*0.5, 0.13*1.5)
        # self.q = random.uniform(1/3*0.5, 1/3*1.5)
        self.tau_x = random.uniform(24.5*0.9, 24.5*1.1)
        # self.k = random.uniform(0.55*0.5, 0.55*1.5)
        # self.kc = random.uniform(0.4*0.5, 0.4*1.5)
        # ProcessS Parameters
        # self.Hm = random.uniform(0.67*0.5, 0.67*1.5)
        # self.Lm = random.uniform(0.17*0.5, 0.17*1.5)
        self.tau_r = random.uniform(18.2*0.5, 18.2*1.5)
        self.tau_d = random.uniform(4.2*0.5, 4.2*1.5)
        self.Ac = random.uniform(0.13333*0.5, 0.13333*1.5)
        # Intertia Parameters
        self.tau_w = random.uniform(0.622*0.5, 0.622*1.5)

    def get_new(self, var_range=0.1):
        # mu = self.mu * (1 + (random.random() - 0.5) * var_range)
        # q = self.q * (1 + (random.random() - 0.5) * var_range)

        mu = self.mu
        q = self.q
        # tau_x = self.tau_x * (1 + (random.random() - 0.5) * (var_range/2))
        tau_x = random.random()*5 - 2.5 + 24.5
        k = self.k
        kc = self.kc
        Hm = self.Hm
        Lm = self.Lm
        # k = self.k * (1 + (random.random() - 0.5) * var_range)
        # kc = self.kc * (1 + (random.random() - 0.5) * var_range)
        # Hm = self.Hm * (1 + (random.random() - 0.5) * var_range)
        # Lm = self.Lm * (1 + (random.random() - 0.5) * var_range)
        tau_r = self.tau_r * (1 + (random.random() - 0.5) * var_range)
        tau_d = self.tau_d * (1 + (random.random() - 0.5) * var_range)
        Ac = self.Ac * (1 + (random.random() - 0.5) * var_range)
        tau_w = self.tau_w * (1 + (random.random() - 0.5) * var_range)
        return ParameterType(mu=mu, q=q, tau_x=tau_x, k=k, kc=kc, Hm=Hm, Lm=Lm,
                             tau_r=tau_r, tau_d=tau_d, Ac=Ac, tau_w=tau_w)

    def load_file(self, file_path):
        file = open(file_path, 'r').read()
        data = file.split('\n')
        for line in data:
            line_data = line.split(": ")
            if len(line_data) < 2:
                continue
            var_name = line_data[0]
            var_value = float(line_data[1])
            if var_name == 'mu':
                self.mu = float(var_value)
            if var_name == 'q':
                self.q = float(var_value)
            if var_name == 'tau_x':
                self.tau_x = float(var_value)
            if var_name == 'k':
                self.k = float(var_value)
            if var_name == 'kc':
                self.kc = float(var_value)
            if var_name == 'Hm':
                self.Hm = float(var_value)
            if var_name == 'Lm':
                self.Lm = float(var_value)
            if var_name == 'tau_r':
                self.tau_r = float(var_value)
            if var_name == 'tau_d':
                self.tau_d = float(var_value)
            if var_name == 'tau_w':
                self.tau_w = float(var_value)
            if var_name == 'Ac':
                self.Ac = float(var_value)

    def save_to_file(self):
        config_str = ""
        config_str += "mu: {}\n".format(self.mu)
        config_str += "q: {}\n".format(self.q)
        config_str += "tau_x: {}\n".format(self.tau_x)
        config_str += "k: {}\n".format(self.k)
        config_str += "kc: {}\n".format(self.kc)
        config_str += "Hm: {}\n".format(self.Hm)
        config_str += "Lm: {}\n".format(self.Lm)
        config_str += "tau_r: {}\n".format(self.tau_r)
        config_str += "tau_d: {}\n".format(self.tau_d)
        config_str += "tau_w: {}\n".format(self.tau_w)
        config_str += "Ac: {}\n".format(self.Ac)

        file_writer = open('parameters', 'w')
        file_writer.write(config_str)
        file_writer.close()


class Model:
    def __init__(self, random_init=False, config_file=None):
        # JFK Parameters
        self.mu = 0.13
        self.q = 1 / 3
        self.tau_x = 24.2
        self.k = 0.55
        self.kc = 0.4
        # ProcessS Parameters
        self.Hm = 0.67
        self.Lm = 0.17
        self.tau_r = 18.2
        self.tau_d = 4.2
        # Intertia Parameters
        self.tau_w = 0.622
        self.Ac = 0.1333

        self.parameters = ParameterType()

        if random_init:
            self.random_parameter()
            self.parameters.random_parameter()
        if config_file is not None:
            self.parameters.load_file(config_file)

        # self.X_train = None
        # self.y_train = None
        # self.X_test = None
        # self.y_test = None

        self.best_parameter = self.parameters
        self.min_error = float('inf')

    def random_parameter(self):
        random.seed(1234)

        # JFK Parameters
        self.mu = random.uniform(0.13*0.5, 0.13*1.5)
        self.q = random.uniform(1/3*0.5, 1/3*1.5)
        self.tau_x = random.uniform(24.5*0.5, 24.5*1.5)
        self.k = random.uniform(0.55*0.5, 0.55*1.5)
        self.kc = random.uniform(0.4*0.5, 0.4*1.5)
        # ProcessS Parameters
        self.Hm = random.uniform(0.67*0.5, 0.67*1.5)
        self.Lm = random.uniform(0.17*0.5, 0.17*1.5)
        self.tau_r = random.uniform(18.2*0.5, 18.2*1.5)
        self.tau_d = random.uniform(4.2*0.5, 4.2*1.5)
        self.Ac = random.uniform(0.13333*0.5, 0.13333*1.5)
        # Intertia Parameters
        self.tau_w = random.uniform(0.622*0.5, 0.622*1.5)

    def train(self, x_train=None, y_train=None, var_range=0.1, n_epoch=100, n_sample=1000,
              save_result=False):
        # self.X_train = x_train
        # self.y_train = y_train
        # self.learning_rate = lr

        min_error = float('inf')
        # best_parameter = ParameterType(mu=self.mu, q=self.q, tau_x=self.tau_x, k=self.k, kc=self.kc,
        #                                Hm=self.Hm, Lm=self.Lm, tau_r=self.tau_r, tau_d=self.tau_d, Ac=self.Ac,
        #                                tau_w=self.tau_w)
        best_parameter = self.parameters
        timer = time.time()

        record = {"tau_x": [0] * n_epoch,
                  "tau_r": [0] * n_epoch,
                  "tau_d": [0] * n_epoch,
                  "Ac": [0] * n_epoch,
                  "tau_w": [0] * n_epoch,
                  "error": [0] * n_epoch}
        print("=== Training Process START ===")
        for epoch in range(n_epoch):
            if epoch > 0:
                epoch_time = time.time() - timer
                timer = time.time()
                print("Epoch {} / {} - Error: {} - {}s".format(epoch, n_epoch, min_error/len(x_train), epoch_time))
            cur_parameter = best_parameter
            for sample in range(n_sample):
                if sample > 0 and (sample+1) % 100 == 0:
                    print("\t{} / {}".format(sample+1, n_sample))
                sample_parameter = cur_parameter.get_new(var_range=var_range)
                mu = sample_parameter.mu
                q = sample_parameter.q
                tau_x = sample_parameter.tau_x
                k = sample_parameter.k
                kc = sample_parameter.kc
                Hm = sample_parameter.Hm
                Lm = sample_parameter.Lm
                tau_r = sample_parameter.tau_r
                tau_d = sample_parameter.tau_d
                Ac = sample_parameter.Ac
                tau_w = sample_parameter.tau_w

                total_error = 0
                for i, train_data in enumerate(x_train):
                    # x = train_data.x
                    beta = train_data.beta
                    y = y_train[i]
                    n_days = len(beta)

                    # JFK
                    x = [0] * n_days
                    dx = [0] * n_days
                    xc = [0] * n_days
                    dxc = [0] * n_days
                    u = [0.1] * n_days
                    # ProcessS
                    H = [0] * n_days
                    dH = [0] * n_days
                    B = [0] * n_days
                    # Inertia
                    W = [0] * n_days
                    dW = [0] * n_days
                    A = [0] * n_days
                    alertness = [0] * n_days

                    x[0] = train_data.x
                    xc[0] = train_data.xc
                    B[0] = train_data.B
                    H[0] = train_data.H
                    W[0] = train_data.W

                    error = [0] * n_days
                    for day in range(n_days):
                        # JFK
                        dx[day] = np.pi / 12 * (
                                    xc[day] + mu * (x[day] / 3 + 4 * x[day] ** 3 / 3 - 256 * x[day] ** 7 / 105)
                                    + (1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day])
                        dxc[day] = np.pi / 12 * (q * xc[day] * (1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day]
                                                 - (24 / (0.99727 * tau_x)) ** 2 * x[day] - k * x[day] * (
                                                         1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day])
                        if not day == n_days-1:
                            x[day+1] = x[day] + dx[day]
                            xc[day+1] = xc[day] + dxc[day+1]

                        # PS
                        B[day] = H[day] - Ac * x[day]

                        if beta[day] == 1:
                            dH[day] = - H[day] / tau_d
                        elif beta[day] == 0:
                            dH[day] = (1 - H[day]) / tau_r
                        if not day == n_days - 1:
                            H[day+1] = H[day] + dH[day]

                        dW[day] = -(1 - beta[day]) * W[day] / tau_w
                        A[day] = (1 - beta[day]) * (1 + Ac - H[day] - W[day])
                        alertness[day] = 56.18 * (A[day] - 0.26) / 0.85 + 45.07

                        if not day == n_days-1:
                            W[day+1] = W[day] + dW[day]

                        error[day] = abs(alertness[day] - y[day])
                    total_error += sum(error) / len(error)

                if total_error <= min_error:
                    min_error = total_error
                    best_parameter = sample_parameter
            record['tau_x'][epoch] = best_parameter.tau_x
            record['tau_r'][epoch] = best_parameter.tau_r
            record['tau_d'][epoch] = best_parameter.tau_d
            record['tau_w'][epoch] = best_parameter.tau_w
            record['Ac'][epoch] = best_parameter.Ac
            record['error'][epoch] = min_error/len(x_train)
        # return best_parameter, min_error/len(x_train)
        print("=== Training Process END ===")
        self.best_parameter = best_parameter
        self.min_error = min_error / len(x_train)
        if save_result:
            self.best_parameter.save_to_file()
        return record

    def predict(self, x_test=None, y_test=None, plot=False, plot_path='result'):
        beta = x_test.beta
        y = y_test
        n_days = len(beta)

        mu = self.best_parameter.mu
        q = self.best_parameter.q
        tau_x = self.best_parameter.tau_x
        k = self.best_parameter.k
        kc = self.best_parameter.kc
        Hm = self.best_parameter.Hm
        Lm = self.best_parameter.Lm
        tau_r = self.best_parameter.tau_r
        tau_d = self.best_parameter.tau_d
        tau_w = self.best_parameter.tau_w
        Ac = self.best_parameter.Ac

        # JFK
        x = [0] * n_days
        dx = [0] * n_days
        xc = [0] * n_days
        dxc = [0] * n_days
        u = [0.1] * n_days
        # ProcessS
        H = [0] * n_days
        dH = [0] * n_days
        B = [0] * n_days
        # Inertia
        W = [0] * n_days
        dW = [0] * n_days
        A = [0] * n_days
        alertness = [0] * n_days

        x[0] = x_test.x
        xc[0] = x_test.xc
        B[0] = x_test.B
        H[0] = x_test.H
        W[0] = x_test.W

        error = [0] * n_days
        for day in range(n_days):
            # JFK
            dx[day] = np.pi / 12 * (
                    xc[day] + mu * (x[day] / 3 + 4 * x[day] ** 3 / 3 - 256 * x[day] ** 7 / 105)
                    + (1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day])
            dxc[day] = np.pi / 12 * (q * xc[day] * (1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day]
                                     - (24 / (0.99727 * tau_x)) ** 2 * x[day] - k * x[day] * (
                                             1 - 0.4 * x[day]) * (1 - kc * xc[day]) * u[day])
            if not day == n_days - 1:
                x[day + 1] = x[day] + dx[day]
                xc[day + 1] = xc[day] + dxc[day + 1]

            # PS
            B[day] = H[day] - Ac * x[day]

            if beta[day] == 1:
                dH[day] = - H[day] / tau_d
            elif beta[day] == 0:
                dH[day] = (1 - H[day]) / tau_r
            if not day == n_days - 1:
                H[day + 1] = H[day] + dH[day]

            dW[day] = -(1 - beta[day]) * W[day] / tau_w
            A[day] = (1 - beta[day]) * (1 + Ac - H[day] - W[day])
            alertness[day] = 56.18 * (A[day] - 0.26) / 0.85 + 45.07

            if not day == n_days - 1:
                W[day + 1] = W[day] + dW[day]

            error[day] = abs(alertness[day] - y[day])
        total_error = sum(error) / len(error)

        if plot:
            n_day = len(y_test)
            days = [i for i in range(n_day)]
            plt.plot(days, alertness, color='b', label='Predicted', linewidth=3)
            plt.plot(days, y_test, color='r', label='Observed')
            plt.legend()
            plt.title(plot_path)
            plt.xlabel('time(h)')
            plt.ylabel('alertness')
            plt.savefig(plot_path+'.jpg')
            plt.show()

        return self.best_parameter, total_error


def from_csv(file_path, n_days=5):
    df = pd.read_csv(file_path)
    day_indexes = df.day.unique()

    test_idx = random.choice(day_indexes[0:len(day_indexes)-5])

    x_train = []
    y_train = []
    x_test = None
    y_test = pd.DataFrame()
    for day_idx in day_indexes:
        if day_idx >= len(day_indexes) - n_days:
            break
        day_df_all = pd.DataFrame()
        for d_idx in range(day_idx, day_idx+n_days):
            day_df = df.loc[df['day'] == d_idx]
            day_df_all = pd.concat([day_df_all, day_df])
        day_df = day_df_all.reset_index()
        # day_df = df.loc[df['day'] == day_idx]
        # day_df = day_df.reset_index()
        x = day_df.loc[0, 'x']
        xc = day_df.loc[0, 'xc']
        B = day_df.loc[0, 'B']
        H = day_df.loc[0, 'H']
        W = day_df.loc[0, 'W']
        beta = day_df['beta'].values
        train_data = DataType(x=x, xc=xc, B=B, H=H, W=W, beta=beta)
        x_train.append(train_data)
        y_train.append(day_df['subjective_alertness'].values)

        if test_idx == day_idx:
            x_test = train_data.copy()
            # y_test = pd.concat([y_test, day_df], ignore_index=True)
            y_test = day_df['subjective_alertness'].values
    # x_test.beta = y_test['beta'].values
    # y_test = y_test['subjective_alertness'].values
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = from_csv("data/train.csv")
    config = 'parameters'
    load_config = False
    train_model = True

    n_epoch = 15
    target_parameter = {"tau_x": [24.2] * n_epoch,
                        "tau_r": [18.2] * n_epoch,
                        "tau_d": [4.2] * n_epoch,
                        "tau_w": [0.622] * n_epoch,
                        "Ac": [0.1333] * n_epoch}

    if load_config:
        model = Model(random_init=True, config_file=config)
    else:
        model = Model(random_init=True)
    model.predict(x_test=x_test, y_test=y_test, plot=True, plot_path='initial')
    if train_model:
        rec = model.train(x_train=x_train, y_train=y_train, n_epoch=n_epoch, n_sample=100, var_range=0.2, save_result=True)
        x = np.linspace(1, n_epoch, n_epoch)
        for key in rec.keys():
            if key == 'error':
                continue
            plt.figure()
            v = rec[key]
            t = target_parameter[key]
            plt.plot(x, v, color='b', label=str(key))
            plt.plot(x, t, color='r', label='target')
            plt.legend()
            plt.xlabel('iteration')
            plt.ylabel(str(key))
            plt.title(str(key))
            plt.savefig('plots/{}.jpg'.format(key))
            plt.close()
    model.predict(x_test=x_test, y_test=y_test, plot=True)








