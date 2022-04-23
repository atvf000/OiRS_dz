import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_m
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

types = ['Bream',
         'Roach',
         'Whitefish',
         'Parkki',
         'Perch',
         'Pike',
         'Smelt']

def species_to_int(data):
    newdata = data.copy()
    for i in types:
        newdata[i] = data['Species'].apply(lambda x: 1 if x == i else 0)
    return newdata

def prepare(data):
    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=5)

    X = teach_data[["Length1", "Height", "Width"] + types].values
    y = teach_data['Weight']

    X_test = test_data[["Length1", "Height", "Width"] + types].values
    y_test = test_data['Weight']

    return X, y, X_test, y_test


class SGDLinearRegression:
    def __init__(self, epsilon=0.001):
        self.w : np.ndarray = None
        self.epsilon = epsilon # на семинаре решили, что это скорость

    @staticmethod
    def __get_size(X: np.ndarray):
        return X.shape[0] # чтобы получать размер выборки или число фич

    @staticmethod
    def __extend_with_ones(X: np.ndarray):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # добавить 1... массово

    def fit(self, X: np.ndarray, y, iterations=500):
        X = SGDLinearRegression.__extend_with_ones(X)
        self.w = X[0].copy() # вроде можно получать shape, но мы всё-таки в питоне, а тут лишнюю функцию вызовешь, потом полдня дебажить
        self.w.fill(0.28147) # 28147-89 - это ГОСТ на симметричное шифрование в СССР, рандомное число из моей головы
        coord_size = SGDLinearRegression.__get_size(X[0]) # количество фич
        sample_amount = SGDLinearRegression.__get_size(X) # количество 
        while iterations > 0:
            iterations -= 1
            for i, x in enumerate(X):
                delta = self.__predict_single_extended(x) - y[i]
                self.w -= 2./coord_size * self.epsilon * delta * x / sample_amount

    def __predict_single_extended(self, X_single: np.ndarray):
        return self.w.dot(X_single)

    def predict(self, X: np.ndarray):
        X = SGDLinearRegression.__extend_with_ones(X)
        return [self.__predict_single_extended(x) for x in X]

    def get_params(self, deep):
        return dict(epsilon=self.epsilon)

    def set_params(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self


def start():
    data = pd.read_csv("Fish.csv")
    data = species_to_int(data)
    X, y, X_test, y_test = prepare(data)

    fitted_regression = make_pipeline(
        StandardScaler(),
        SGDLinearRegression()
    )

    fitted_regression.fit(X, y)
    print('Result: ', math.sqrt(
        skl_m.mean_squared_error(
            y_test,
            fitted_regression.predict(X_test))
        )
    )

    ms_scorer = skl_m.make_scorer(skl_m.mean_squared_error, greater_is_better=False)

    fitted_regression = skl_m.GridSearchCV(
        SGDLinearRegression(),
        dict(sgdlinearregression__epsilon=np.geomspace(0.0001, 1, num=13)),
        scoring=ms_scorer)

    fitted_regression.fit(X, y)


    print('Result: ', math.sqrt(
        skl_m.mean_squared_error(
            y_test,
            fitted_regression.predict(X_test))
        )
    )

    plt.plot([i['sgdlinearregression__epsilon'] for i in fitted_regression.cv_results_['params']],
                   # в задании эпсилонов чуть меньше, но ведь больше=лучше, верно?
                   fitted_regression.cv_results_['mean_test_score'])
    plt.show()

if __name__ == '__main__':
    start()
