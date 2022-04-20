import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging as log
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sm
from sklearn.pipeline import make_pipeline
import sklearn.model_selection as sms
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
BASE_COLUMN = 'Y house price of unit area'
BUILDING_GRAPHS = False
PRINT_DATAFRAME = False
data_frames = {}


class SGDLinearRegression:
    def __init__(self, epsilon=0.001):
        self.w = None
        self.bias = None
        self.epsilon = epsilon
        self._costs = []
        self._iterations = []
        self._attrs_to_get = { 'epsilon' }
        np.random.seed(np.random.randint(100))

    def get_params(self, deep):
        return {
            k: getattr(self, k)
            for k in self._attrs_to_get
        }

    def set_params(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        return self

    def fit(self, X, y, iterations=500):
        m, length = X.shape
        self.w = np.random.randn(length)
        self.bias = 0

        current_iter = 0
        while current_iter < iterations:
            h = np.dot(X, self.w) + self.bias
            loss = np.square(h - y)
            cost = np.sum(loss) / (2 * m)

            h = np.dot(X, self.w) + self.bias
            delta_w = np.dot(X.T, (h - y)) / m / length
            db = np.sum(h - y) / m / length
            self.w = self.w - self.epsilon * delta_w
            self.bias = self.bias - self.epsilon * db

            current_iter = current_iter + 1

            self._costs.append(cost)
            self._iterations.append(current_iter)

    def predict(self, X):
        return np.dot(X, self.w) + self.bias

    def plot(self, figsize=(7, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self._iterations,self._costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title("Iterations vs Cost")
        plt.show()

    def score(self, X, y):
        return 1 - (np.sum(((y - self.predict(X))**2))/np.sum((y - np.mean(y))**2))


def process():
    data = pd.read_csv('real_estate.csv')
    x_data = data.drop(BASE_COLUMN, axis=1)
    y_data = data[BASE_COLUMN]
    x_train, x_test, y_train, y_test = sms.train_test_split(x_data, y_data, test_size=0.2, random_state=101)

    reg = make_pipeline(StandardScaler(), SGDLinearRegression())
    logic_cv = GridSearchCV(
        reg,
        dict(sgdlinearregression__epsilon=np.geomspace(0.0001, 1, num=13))
    )

    log.warning(f'Training for SGDLinearRegression regression...')

    logic_cv.fit(x_train, y_train)
    log.warning(f'Predicting for SGDLinearRegression regression...')

    predicted_cv = logic_cv.predict(x_test)

    mse = sm.mean_squared_error(y_test, predicted_cv)
    rmse = np.sqrt(mse)

    return rmse


def start():
    print(f'RMSE: {process()}')

if __name__ == '__main__':
    start()