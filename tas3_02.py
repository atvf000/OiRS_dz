import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torch import optim
from torch.autograd import Variable
import re
import os

def get_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read()
    return data

def prepare_data(data):
    replaces = [
        (re.compile(r'[,;:")(«»^]+'), ''),
        (re.compile(r'[?!.…]+'), ''),
        (re.compile(r'–|-|—'), ''),
        (re.compile(r'\t'), ''),
        (re.compile(r'\n\n+'), '\n'),
        (re.compile(r'[\[\]]+'), ''),
        (re.compile(r'[XLVI]+'), ''),
    ]

    for regex, rep in replaces:
        data = regex.sub(rep, data)

    data = data.lower()
    return data


def create_model(hidden_size, output_size):
    model = nn.Sequential()

    model.add_module('bn1', nn.BatchNorm1d(hidden_size))
    model.add_module('l1', nn.Linear(hidden_size, 256))
    model.add_module('bn2', nn.BatchNorm1d(256))
    model.add_module('l2', nn.Linear(256, output_size))
    model.add_module('a1', nn.ReLU())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


def training(data, model, optimizer, epoch):
    loss_func = nn.BCELoss(reduction='sum')
    result = []

    for ep in range(epoch):
        for i, (images, labels) in enumerate(data):
            optimizer.zero_grad()

            labels = idx_to_Tensor(labels, data)

            b_x = Variable(images)
            b_y = Variable(labels)
            output = model.forward(b_x)
            loss = loss_func(output, b_y)

            loss.backward()
            optimizer.step()
            result.append(loss.item())

            if (i + 1) % 1000 == 0:
                print(f'Epoch {ep + 1}; Step {i + 1}; Loss {loss.item():.4f}')

    return result


def testing(data, model):
    correct = 0
    total = len(data)
    for images, labels in data:
        test_output = model.forward(images)
        pred_y = torch.max(test_output, 0)[1]
        correct += (pred_y == labels).sum().item()

    accuracy = float(correct) / total
    print('Test Accuracy of the model: %.2f' % accuracy)


def graph(result):
    plt.plot(np.arange(len(result)), result)
    plt.show()


def start():
    print("Read data")

    print("-" * 25)
    print("-" * 5, "augmentation:", "-" * 5)

    transform_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(196),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.ToTensor()
    ])
    data_train_aug = torchvision.datasets.FashionMNIST(root='FashionMNIST/raw/train-images-idx3-ubyte',
                                                       train=True, download=True, transform=transform_aug)
    data_test_aug = torchvision.datasets.FashionMNIST(root='FashionMNIST/raw/train-images-idx3-ubyte',
                                                      train=False, download=True, transform=transform_aug)

    print("Create model")
    model, optimizer = create_model(has_batch=True, has_dropout=True)

    print("Training")
    aug_result = training(data_train_aug, model, optimizer, 1)

    print("Testing")
    testing(data_test_aug, model)

    print("Graph")
    graph(aug_result)


if __name__ == '__main__':
    start()

    ##!wget https://gitlab.toliak.ru/Toliak/oirs-datasets/-/raw/master/K_onegin.txt
