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


def create_model(layers=1, func=2, optim_type=2):
    func_activ = {
        0: nn.Sigmoid(),
        1: nn.Tanh(),
        2: nn.ReLU(),
        3: nn.ELU()
    }
    model = nn.Sequential()

    for i in range(1, layers):
        model.add_module(f'l{i}', nn.Linear(in_features=10, out_features=10))
        model.add_module(f'a{i}', func_activ.get(func))

    model.add_module(f'l{layers}', nn.Linear(in_features=10, out_features=1))

    optims = {
        0: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0),
        1: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5),
        2: torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5, momentum=0),
        3: torch.optim.Adam(model.parameters(), lr=0.01)
    }

    return model, optims.get(optim_type)


def training(X, y, model, optim):
    loss_f = F.mse_loss
    loss_history = []

    for epoch in range(1000):
        optim.zero_grad()
        model.train()

        y_pred = model.forward(X)

        loss = loss_f(y_pred, y)
        loss.backward()

        optim.step()

        loss_history.append(math.sqrt(loss.item()))

    plt.plot(loss_history)
    plt.show()
    print(loss_history[-1])


def start():
    print("Read data from csv")
    data = pd.read_csv("Fish.csv")

    print("Prepare data")
    data = species_to_int(data)
    data = data[data['Weight'] <= data["Weight"].quantile(q=0.95)]
    X, y, X_test, y_test = prepare(data)

    loss = []
    b_amount_layers = 0

    print("Task 1.3")
    loss.clear()
    for i in range(1, 5):
        print(f'amount of layers: {i}')
        model, optim = create_model(layers=i)
        training(X, y, model, optim)
        # TODO add loss append

    print("Task 1.4")
    loss.clear()
    for i in range(4):



if __name__ == '__main__':
    start()
