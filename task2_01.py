import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import sklearn.model_selection
import math

types = ['Bream',
         'Roach',
         'Whitefish',
         'Parkki',
         'Perch',
         'Pike',
         'Smelt']

func_act_names = {
    0: "Sigmoid",
    1: "Tanh",
    2: "ReLU",
    3: "ELU"
}

optim_names = {
    0: "SGD",
    1: "SGD with momentum",
    2: "RMSprop",
    3: "Adam"
}


def kill_random():
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def species_to_int(data):
    newdata = data.copy()
    for i in types:
        newdata[i] = data['Species'].apply(lambda x: 1 if x == i else 0)
    return newdata


def prepare(data):
    teach_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=5)

    X = teach_data[["Length1", "Height", "Width"] + types].values
    y = teach_data['Weight'].values

    X = torch.tensor(X).float()
    y = torch.tensor(y.reshape((-1, 1))).float()

    X_test = test_data[["Length1", "Height", "Width"] + types].values
    y_test = test_data['Weight'].values

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test.reshape((-1, 1))).float()

    return X, y, X_test, y_test


def create_model(layers=1, func=2, optim_type=2, has_batch=False, dropout_prob=None):
    func_activ = {
        0: nn.Sigmoid(),
        1: nn.Tanh(),
        2: nn.ReLU(),
        3: nn.ELU()
    }
    model = nn.Sequential()

    if has_batch:
        model.add_module('b1', nn.BatchNorm1d(10, True))

    for i in range(1, layers):
        model.add_module(f'l{i}', nn.Linear(in_features=10, out_features=10))
        model.add_module(f'a{i}', func_activ.get(func))

    if dropout_prob is not None:
        model.add_module('d1', nn.Dropout(p=dropout_prob))

    model.add_module(f'l{layers}', nn.Linear(in_features=10, out_features=1))

    optims = {
        0: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0),
        1: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5),
        2: torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5, momentum=0),
        3: torch.optim.Adam(model.parameters())
    }

    return model, optims.get(optim_type)


def training(X, y, model, optim):
    loss_f = F.mse_loss

    for epoch in range(1000):
        optim.zero_grad()
        model.train()

        y_pred = model.forward(X)

        loss = loss_f(y_pred, y)
        loss.backward()

        optim.step()

    return math.sqrt(loss_f(model.forward(X), y).item())


def predicting(X, y, model):
    loss_f = F.mse_loss
    return math.sqrt(loss_f(model.forward(X), y).item())


def print_graph(loss, loss_train):
    plt.plot(loss, 'ob', label="train")
    plt.plot(loss_train, 'or', label="test")
    plt.legend(loc="upper left")
    plt.show()


def start():
    print("Read data from csv")
    data = pd.read_csv("Fish.csv")

    print("Prepare data")
    data = species_to_int(data)
    data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    X, y, X_test, y_test = prepare(data)

    loss = []
    loss_train = []

    # ----------------------------------------------------------------------
    print("Task 1.3")
    loss_train.clear()
    loss.clear()

    for i in range(1, 5):
        print(f'amount of layers: {i}')
        model, optim = create_model(layers=i)
        training(X, y, model, optim)

        loss.append(
            training(X, y, model, optim)
        )

        loss_train.append(
            predicting(X_test, y_test, model)
        )

        print(loss_train[-1])

    print(loss_train)
    print_graph(loss, loss_train)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.4")
    loss_train.clear()
    loss.clear()

    for i in range(4):
        print(f'type of activation function: {func_act_names.get(i)}')
        model, optim = create_model(layers=4, func=i)

        loss.append(
            training(X, y, model, optim)
        )

        loss_train.append(
            predicting(X_test, y_test, model)
        )
        print(loss_train[-1])

    print(loss_train)
    print_graph(loss, loss_train)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.5")
    loss_train.clear()
    loss.clear()

    for i in range(4):
        print(f'type of optimizer: {optim_names.get(i)}')
        model, optim = create_model(layers=4, func=2, optim_type=i)

        loss.append(
            training(X, y, model, optim)
        )

        loss_train.append(
            predicting(X_test, y_test, model)
        )
        print(loss_train[-1])

    print(loss_train)
    print_graph(loss, loss_train)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.6")
    loss_train.clear()
    loss.clear()

    for i in [None, 0.2, 0.5]:
        for j in [False, True]:
            print(f'has batchNorm: {j}, probability of dropout: {i}')
            model, optim = create_model(layers=4, func=2, optim_type=2, has_batch=j, dropout_prob=i)

            loss.append(
                training(X, y, model, optim)
            )

            loss_train.append(
                predicting(X_test, y_test, model)
            )
            print(loss_train[-1])

    print(loss_train)
    print_graph(loss, loss_train)


if __name__ == '__main__':
    start()
