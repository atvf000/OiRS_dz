import pandas as LibPandas
import matplotlib.pyplot as LibPyplot
import sklearn.model_selection as LibSelection
import sklearn.linear_model as LibLinear
import sklearn.metrics as LibMetrics
from math import sqrt
from decimal import *

getcontext().prec = 2

type_to_int_converter = {
    'Bream': 0,
    'Roach': 1,
    'Whitefish': 2,
    'Parkki': 3,
    'Perch': 4,
    'Pike': 5,
    'Smelt': 6
}

LinearRegression = LibLinear.LinearRegression
Ridge = LibLinear.Ridge
Lasso = LibLinear.Lasso
CHOSEN_PARAM_BASE = ["Length1", 'Height', 'Width']

def declassify_to_vector(daframe: LibPandas.DataFrame):
    types_of_fish = set()
    for row in daframe['Species']:
        types_of_fish.add(row)
    types_of_fish.pop()
    newdaframe = daframe.copy()
    for type in types_of_fish:
        newdaframe[type] = daframe['Species'].apply(lambda row: 1 if row == type else 0)
    columns = list(newdaframe.columns)
    columns.remove('Species')
    newdaframe = newdaframe[columns]
    return types_of_fish, newdaframe


def unwrap(x, y):
    if y is not None:
        return x(alpha=y)
    return x()



def task1_attempt2(RegType, a=None):
    data = LibPandas.read_csv("Fish.csv")
    data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    types, data = declassify_to_vector(data)
    # LibSeaborn.pairplot(data) # 1.2
    # LibPyplot.show()
    teach_data, test_data = LibSelection.train_test_split(data, test_size=0.2, random_state=2636936)

    CHOSEN_PARAM = CHOSEN_PARAM_BASE.copy()
    CHOSEN_PARAM.extend(types)

    X = teach_data[CHOSEN_PARAM].values
    y = teach_data['Weight']
    fitted_regression: RegType = unwrap(RegType, a).fit(X, y)
    X = test_data[CHOSEN_PARAM].values
    y = test_data['Weight']
    return sqrt(LibMetrics.mean_squared_error(y, fitted_regression.predict(X)))


def task1_cycle():
    print('x  RMSE: ', task1_attempt2(LinearRegression))
    print('2  RMSE: ', task1_attempt2(Ridge))
    print('1  RMSE: ', task1_attempt2(Lasso))
    alpha = Decimal(0)
    oofive = Decimal(0.05)
    values = []
    alphas = []
    while alpha <= Decimal(1):
        values.append(task1_attempt2(Ridge, alpha))
        alphas.append(alpha)
        alpha += oofive
    print('2_ RMSE: ', min(values))
    print('2^ RMSE: ', max(values))
    LibPyplot.plot(alphas, values)
    LibPyplot.show()


def minecraft():
    task1_cycle()


if __name__ == '__main__':
    minecraft()
