import numpy as np
import pandas as pd
import sklearn.model_selection as skl_ms
import matplotlib.pyplot as plt
import sklearn.metrics as skl_m
import sklearn.feature_extraction.text as skl_text
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare(data):
    data.pop("severe_toxic")
    data.pop("obscene")
    data.pop("threat")
    data.pop("insult")
    data.pop("identity_hate")

    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=666)

    vectorizer = skl_text.CountVectorizer(lowercase=True, ngram_range=(1, 1), strip_accents='unicode',
                                          stop_words={'english'}, analyzer='word')
    vectorizer.fit(data['comment_text'])

    X = vectorizer.transform(teach_data["comment_text"])
    y = teach_data['toxic']

    X_test = vectorizer.transform(test_data["comment_text"])
    y_test = test_data['toxic']

    return X, y, X_test, y_test


class SGDLogisticRegression:
    def __init__(self, epsilon=0.001):
        self.w = None
        self.epsilon = epsilon
        pass

    def sigmoid(self, x, w):
        return 1 / (1 + np.exp(-1 * np.dot(x, w)))

    def fit(self, X, y, iterations=500):
        X = np.concatenate(
            (np.ones((X.shape[0], 1)), X),
            axis=1
        )
        self.w = X[0].copy()
        self.w.fill(0.28147)
        for i in range(iterations):
            sigma = self.sigmoid(X, self.w)
            self.w -= self.epsilon * np.dot(X.T, (sigma - y)) / y.shape[0]

    def predict(self, X):
        X = np.concatenate(
            (np.ones((X.shape[0], 1)), X),
            axis=1
        )
        return [i >= 0.5 for i in self.sigmoid(X, self.w)]

    def get_params(self, deep):
        return dict(epsilon=self.epsilon)

    def set_params(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return self


def start():
    print("Read data from csv")
    data = pd.read_csv("train.csv")
    data = data.iloc[0:1000]

    print("Prepare data")
    X, y, X_test, y_test = prepare(data)

    print("-" * 20, "\nStart grid search")
    f1_scorer = make_scorer(f1_score, average='macro')
    pipeline_scaler_regression = make_pipeline(SGDLogisticRegression())

    pipeline = GridSearchCV(
        pipeline_scaler_regression,
        dict(sgdlogisticregression__epsilon=np.geomspace(0.0001, 1, num=13)),
        # dict(sgdlogisticregression__epsilon=[1]),
        scoring=f1_scorer,
        error_score='raise'
    )

    pipeline.fit(X.toarray(), y.values)

    print("Start predict")
    y_predict = pipeline.predict(X_test.toarray())
    f1 = skl_m.f1_score(y_test.values, y_predict, average='macro')

    print("f1:", f1)

    print("Create graphic")

    epsilons = []
    results = []
    for eps in [0.0001, 0.001, 0.01, 0.1, 1]:
        pipeline_scaler_regression = make_pipeline(StandardScaler(), SGDLogisticRegression())

        pipeline = GridSearchCV(
            pipeline_scaler_regression,
            dict(sgdlinearregression__epsilon=[eps]),
            scoring=f1_scorer
        )

        pipeline.fit(X, y)

        epsilons.append(eps)
        results.append(
            skl_m.f1_score(y_test, y_predict, average='macro')
        )

    plt.plot(epsilons, results)
    plt.show()

    pipeline_sqdclass = SGDClassifier(loss='log',
                                      penalty='l1',
                                      epsilon=53,
                                      random_state=42,
                                      tol=None
                                      )

    pipeline = GridSearchCV(
        pipeline_scaler_regression,
        dict(sgdlinearregression__epsilon=[eps]),
        scoring=f1_scorer
    )

    pipeline.fit(X, y)

    print("SGD classifier:\t", skl_m.f1_score(y_test, y_predict, average='macro'))


if __name__ == '__main__':
    start()
