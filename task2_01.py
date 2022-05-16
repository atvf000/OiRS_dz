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


def training(X, y):
    model = nn.Sequential()
    model.add_module('l1', nn.Linear(in_features=10, out_features=1))
    model.add_module('a1', nn.LeakyReLU())

    optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5, momentum=0)
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

    print("Start nn\n")
    training(X, y)


if __name__ == '__main__':
    start()