import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as skl_ms
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_m
import math

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


def print_pairplot(data):
    sns.pairplot(data[["Weight", "Length1", "Length2", "Length3", "Height", "Width"]])
    plt.show()


def prepare(data):
    teach_data, test_data = skl_ms.train_test_split(data, test_size=0.2, random_state=5)

    X = teach_data[["Length1", "Height", "Width"] + types].values
    y = teach_data['Weight']
    return X, y


def linear_regression(X, y):
    fitted_regression = skl_lm.LinearRegression().fit(X, y)
    return math.sqrt(skl_m.mean_squared_error(y, fitted_regression.predict(X)))


def ridge(X, y, a=1.):
    fitted_regression = skl_lm.Ridge(alpha=a).fit(X, y)
    return math.sqrt(skl_m.mean_squared_error(y, fitted_regression.predict(X)))


def lasso(X, y):
    fitted_regression = skl_lm.Lasso().fit(X, y)
    return math.sqrt(skl_m.mean_squared_error(y, fitted_regression.predict(X)))


def print_plot(x, y):
    plt.plot(x, y)
    plt.show()


def start():
    data = pd.read_csv("Fish.csv")
    data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    # print_pairplot(data)
    data = species_to_int(data)

    X, y = prepare(data)
    print("LR   \tRMSE: \t", linear_regression(X, y))

    selected_data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    X, y = prepare(selected_data)
    print("LR s \tRMSE: \t", linear_regression(X, y))

    X, y = prepare(data)
    print("Ridge \tRMSE: \t", ridge(X, y))

    X, y = prepare(data)
    print("Lasso \tRMSE: \t", lasso(X, y))

    alpha = 0.
    results = []
    alphas = []
    while alpha <= 1.:
        results.append(ridge(X, y, alpha))
        alphas.append(alpha)
        alpha += 0.05
    print_plot(alphas, results)


if __name__ == '__main__':
    start()
