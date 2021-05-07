from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')

    train_variables = ['day_ts', 'beta', 'x']
    predict_variables = ['subjective_alertness']

    X = data[train_variables].values
    y = data[predict_variables].values

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    print(X_train.shape, y_train.shape)

    model = svm.SVR()
    model.fit(X_train, y_train.ravel())

    X_test = sc_X.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)

    diff_sum = 0
    for i in range(len(y_pred)):
        diff = abs(y_pred[i] - y_test[i][0]) / y_test[i][0]
        diff_sum += diff
    print("Variation: {}".format(diff_sum / len(y_pred)))
