import numpy as np
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_m
import sklearn.feature_extraction.text as skl_text


class SGDLinearRegression:
    def __init__(self, epsilon=0.001):
        self.w = None
        self.bias = None
        self.epsilon = epsilon

    @staticmethod
    def __get_size(X: np.ndarray):
        return X.shape[0]  # чтобы получать размер выборки или число фич

    @staticmethod
    def __extend_with_ones(X: np.ndarray):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # добавить 1... массово

    def fit(self, X: np.ndarray, y, iterations=500):
        X = SGDLinearRegression.__extend_with_ones(X)
        self.w = X[0].copy()
        self.w.fill(0.28147)
        coord_size = SGDLinearRegression.__get_size(X[0])
        sample_amount = SGDLinearRegression.__get_size(X)
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
