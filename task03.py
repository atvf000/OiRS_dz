import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

types = ['Bream',
         'Roach',
         'Whitefish',
         'Parkki',
         'Perch',
         'Pike']


def species_to_int(data):
    newdata = data.copy()
    for i in types:
        newdata[i] = data['Species'].apply(lambda x: 1 if x == i else 0)
    return newdata


def prepare(data):
    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=100)

    X = teach_data[["Length1", "Height", "Width"] + types].values
    y = teach_data['Weight'].values.reshape(-1, 1)

    X_test = test_data[["Length1", "Height", "Width"] + types].values
    y_test = test_data['Weight'].values.reshape(-1, 1)

    return X, y, X_test, y_test


class SGDLinearRegression:
    def __init__(self, epsilon=0.001):
        self.w: np.ndarray = None
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y, iterations=500):
        X = np.concatenate(
            (np.ones((X.shape[0], 1)), X),
            axis=1
        )
        self.w = np.random.randn(X.shape[1])
        features = X.shape[1]
        samples = X.shape[0]

        iter = 0
        while iter < iterations:
            for i, x in enumerate(X):
                h = self.w.dot(x)
                delta = h - y[i]
                self.w -= 2 * self.epsilon * delta * x / samples / features
            iter += 1

    def predict(self, X: np.ndarray):
        X = np.concatenate(
            (np.ones((X.shape[0], 1)), X),
            axis=1
        )
        return [self.w.dot(x) for x in X]

    def get_params(self, deep):
        return dict(epsilon=self.epsilon)

    def set_params(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self


def start():
    print("Read data from csv")
    data = pd.read_csv("Fish.csv")

    print("Prepare data")
    data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    data = species_to_int(data)
    X, y, X_test, y_test = prepare(data)

    print("-" * 20, "\nStart grid search")
    ms_scorer = skl_m.make_scorer(skl_m.mean_squared_error, greater_is_better=False)
    pipeline_scaler_regression = make_pipeline(StandardScaler(), SGDLinearRegression())

    pipeline = GridSearchCV(
        pipeline_scaler_regression,
        dict(sgdlinearregression__epsilon=np.geomspace(0.0001, 1, num=13)),
        scoring=ms_scorer
    )

    pipeline.fit(X, y)

    print("Start predict")
    print('Result: ',
          math.sqrt(
              skl_m.mean_squared_error(
                  pipeline.predict(X_test),
                  y_test
              )
          )
    )

    print("Create graphic")

    epsilons = []
    results = []
    for eps in [0.0001, 0.001, 0.01, 0.1, 1]:
        pipeline_scaler_regression = make_pipeline(StandardScaler(), SGDLinearRegression())

        pipeline = GridSearchCV(
            pipeline_scaler_regression,
            dict(sgdlinearregression__epsilon=[eps]),
            scoring=ms_scorer
        )

        pipeline.fit(X, y)

        epsilons.append(eps)
        results.append(
            math.sqrt(
                skl_m.mean_squared_error(
                    pipeline.predict(X_test),
                    y_test
                )
            )
        )

    plt.plot(epsilons, results)
    plt.show()

    pipline_reg = make_pipeline(StandardScaler(), SGDRegressor(alpha=0))
    pipline_reg.fit(X, y)

    print("Start predict")
    print('Result: ',
          math.sqrt(
              skl_m.mean_squared_error(
                  pipline_reg.predict(X_test),
                  y_test
              )
          )
    )


if __name__ == '__main__':
    start()
