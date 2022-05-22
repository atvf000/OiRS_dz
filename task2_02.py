import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import sklearn.model_selection
from sklearn.feature_extraction import text
from sklearn.metrics import f1_score

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


def accuracy(y_pred, y):
    return (torch.round(y_pred) == y).float().sum() / len(y_pred)


def f1(y_pred, y):
    f = lambda x: x.detach().numpy()
    return f1_score(f(y), f(y_pred), average='macro')


def kill_random():
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def prepare(data):
    data.pop("severe_toxic")
    data.pop("obscene")
    data.pop("threat")
    data.pop("insult")
    data.pop("identity_hate")

    teach_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=5)

    vectorizer = text.CountVectorizer(lowercase=True, ngram_range=(1, 1), strip_accents='unicode',
                                      stop_words={'english'}, analyzer='word')
    vectorizer.fit(data['comment_text'])

    X = vectorizer.transform(teach_data["comment_text"]).toarray()
    y = teach_data['toxic'].to_numpy()

    X_test = vectorizer.transform(test_data["comment_text"]).toarray()
    y_test = test_data['toxic'].to_numpy()

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
        model.add_module(f'l{i}', nn.Linear(in_features=9379, out_features=9379))
        model.add_module(f'a{i}', func_activ.get(func))

    if dropout_prob is not None:
        model.add_module('d1', nn.Dropout(p=dropout_prob))

    model.add_module(f'l{layers}', nn.Linear(in_features=9379, out_features=1))

    optims = {
        0: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0),
        1: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5),
        2: torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5, momentum=0),
        3: torch.optim.Adam(model.parameters())
    }

    return model, optims.get(optim_type)


def training(X, y, model, optim):
    loss = nn.BCELoss()

    for epoch in range(1000):
        optim.zero_grad()
        model.train()

        y_pred = model.forward(torch.tensor(X).float())
        y_res = torch.tensor(y).float()
        y_res = torch.Tensor(np.array([np.array([y]) for y in y_res]))

        loss = loss(y_pred, y_res)
        loss.backward()

        optim.step()

    return loss(model.forward(torch.tensor(X).float()), y).item(), \
           accuracy(model.forward(torch.tensor(X).float()), y), \
           f1(model.forward(torch.tensor(X).float()), y)


def predicting(X, y, model):
    loss = nn.BCELoss()
    return loss(model.forward(torch.tensor(X).float()), y).item(), \
           accuracy(model.forward(torch.tensor(X).float()), y), \
           f1(model.forward(torch.tensor(X).float()), y)


def print_graph(loss, loss_test, acc, acc_test):
    plt.plot(loss, 'ob', label="train")
    plt.plot(loss_test, 'or', label="test")
    plt.legend(loc="upper left")

    plt.plot(acc, 'ob', label="train")
    plt.plot(acc_test, 'or', label="test")
    plt.legend(loc="upper left")
    plt.show()


def start():
    print("Read data from csv")
    data = pd.read_csv("train.csv")
    data = data.iloc[0:1000]

    print("Prepare data")
    X, y, X_test, y_test = prepare(data)

    loss = []
    loss_test = []
    acc = []
    acc_test = []
    f1_test = []

    # ----------------------------------------------------------------------
    print("Task 1.3")
    loss_test.clear()
    loss.clear()
    acc_test.clear()
    acc.clear()
    f1_test.clear()

    kill_random()

    for i in range(1, 5):
        print(f'amount of layers: {i}')
        model, optim = create_model(layers=i)
        loss_t, acc_t, f1_t = training(X, y, model, optim)
        loss_p, acc_p, f1_p = predicting(X_test, y_test, model)

        loss.append(loss_t)
        acc.append(acc_t)

        loss_test.append(loss_p)
        acc_test.append(acc_p)
        f1_test.append(f1_p)

        print(f1_p)

    print(f1_test)
    print_graph(loss, loss_test, acc, acc_test)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.4")
    loss_test.clear()
    loss.clear()
    acc_test.clear()
    acc.clear()
    f1_test.clear()

    kill_random()

    for i in range(4):
        print(f'type of activation function: {func_act_names.get(i)}')
        model, optim = create_model(layers=4, func=i)
        loss_t, acc_t, f1_t = training(X, y, model, optim)
        loss_p, acc_p, f1_p = predicting(X_test, y_test, model)

        loss.append(loss_t)
        acc.append(acc_t)

        loss_test.append(loss_p)
        acc_test.append(acc_p)
        f1_test.append(f1_p)

        print(f1_p)

    print(f1_test)
    print_graph(loss, loss_test, acc, acc_test)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.5")
    loss_test.clear()
    loss.clear()
    acc_test.clear()
    acc.clear()
    f1_test.clear()

    kill_random()

    for i in range(4):
        print(f'type of optimizer: {optim_names.get(i)}')
        model, optim = create_model(layers=4, func=2, optim_type=i)
        loss_t, acc_t, f1_t = training(X, y, model, optim)
        loss_p, acc_p, f1_p = predicting(X_test, y_test, model)

        loss.append(loss_t)
        acc.append(acc_t)

        loss_test.append(loss_p)
        acc_test.append(acc_p)
        f1_test.append(f1_p)

        print(f1_p)

    print(f1_test)
    print_graph(loss, loss_test, acc, acc_test)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.6")
    loss_test.clear()
    loss.clear()
    acc_test.clear()
    acc.clear()
    f1_test.clear()

    kill_random()

    for i in [None, 0.2, 0.5]:
        for j in [False, True]:
            print(f'has batchNorm: {j}, probability of dropout: {i}')
            model, optim = create_model(layers=4, func=2, optim_type=2, has_batch=j, dropout_prob=i)
            loss_t, acc_t, f1_t = training(X, y, model, optim)
            loss_p, acc_p, f1_p = predicting(X_test, y_test, model)

            loss.append(loss_t)
            acc.append(acc_t)

            loss_test.append(loss_p)
            acc_test.append(acc_p)
            f1_test.append(f1_p)

            print(f1_p)

    print(f1_test)
    print_graph(loss, loss_test, acc, acc_test)

    # ----------------------------------------------------------------------
    print('\n', '-' * 20)
    print("Task 1.7")
    loss_test.clear()
    loss.clear()
    acc_test.clear()
    acc.clear()
    f1_test.clear()

    kill_random()

    model, optim = create_model(layers=4, func=2, optim_type=2, has_batch=False, dropout_prob=None)
    loss_t, acc_t, f1_t = training(X, y, model, optim)
    loss_p, acc_p, f1_p = predicting(X_test, y_test, model)

    print(f1_p)


if __name__ == '__main__':
    start()
