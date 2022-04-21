import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.linear_model as slm
import sklearn.metrics as sm
import sklearn.model_selection as sms
from decimal import *
import logging as log
from time import sleep

BASE_COLUMN = 'Y house price of unit area'
BUILDING_GRAPHS = False
PRINT_DATAFRAME = False
data_frames = {}


models = {
    'Linear': slm.LinearRegression(),
    'Ridge': slm.Ridge(),
    'Lasso': slm.Lasso(),
}


def build_graph():
    data = pd.read_csv('real_estate.csv')
    sb.pairplot(data)
    plt.show()


def process(model_str, q=False, alpha=None, logging=True):
    if logging:
        log.warning(f'Starting processing for {model_str} regression...')
    model = models.get(model_str)

    if alpha:
        model.set_params(alpha=alpha)

    data = pd.read_csv('real_estate.csv')
    data.drop(labels='No', axis=1, inplace=True)

    if q:
        data = data[data[BASE_COLUMN] <= data[BASE_COLUMN].quantile(q=0.95)]
    x_data = data.drop(BASE_COLUMN, axis=1)
    y_data = data[BASE_COLUMN]

    if logging:
        log.warning(f'Splitting data for {model_str} regression...')
    x_train, x_test, y_train, y_test = sms.train_test_split(x_data, y_data, test_size=0.2, random_state=101)

    if logging:
        log.warning(f'Training for {model_str} regression...')
    model.fit(x_train, y_train)

    if PRINT_DATAFRAME:
        print(pd.DataFrame(model.coef_, x_data.columns, columns=['coeficient']))

    if logging:
        log.warning(f'Predicting for {model_str} regression...')
    y_pred = model.predict(x_test)

    if PRINT_DATAFRAME:
        print(pd.DataFrame({'Y_test':y_test, 'Y_pred':y_pred}).head())

    mse = sm.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    if BUILDING_GRAPHS:
        test_residuals = y_test - y_pred
        sb.scatterplot(x=y_test, y=y_pred, color='#20d489')
        plt.axhline(y=30, color='#105c3c', ls='--')
        sb.displot(test_residuals, bins=25, kde=True, color='#6f00ff')

    if logging: log.warning(f'Finished job for {model_str} regression...')
    return rmse


def process_oridge(model_str, q=False):
    alpha, oofive = Decimal(0), Decimal(0.05)
    values, alphas = [], []
    log.warning(f'Starting processing for Optimized Ridge regression...')
    while alpha <= Decimal(1):
        values.append(process(model_str=model_str, alpha=alpha, q=q, logging=False))
        alphas.append(alpha)
        alpha += oofive
    log.warning(f'Finished job for Optimized Ridge regression...')
    sleep(0.5)
    return min(values), max(values)


def start():
    if BUILDING_GRAPHS:
        build_graph()

    running = {}

    for model in models:
        running[model] = [process(model_str=model), process(model_str=model, q=True)]

    running['Optimized Ridge'] = [process_oridge(model_str='Ridge'), process_oridge(model_str='Ridge', q=True)]

    for regr in running:
        print(f'|-- {regr} regression\n'
              f'|-- RMSE default: {running.get(regr)[0]}\n'
              f'|-- RMSE q < 0.95: {running.get(regr)[1]}\n'
              f'\n')


if __name__ == '__main__':
    start()
