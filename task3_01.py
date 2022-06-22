import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torch import optim
from torch.autograd import Variable


def idx_to_Tensor(ind, data):
    charact_vec = [0] * len(data.classes)
    charact_vec[ind] = 1
    return torch.Tensor(np.array(charact_vec))


def create_model(has_batch=False, has_dropout=False):
    model = nn.Sequential()
    size = 1
    if has_batch:
        model.add_module('b1', nn.BatchNorm1d(196))

    for i in range(1, 3):
        model.add_module(f'c{i}', nn.Conv2d(
            in_channels=size,
            out_channels=(size << 2),
            kernel_size=5,
            stride=1,
            padding=2,
        )
                         )
        size = size << 2

        model.add_module(f'a{i}', nn.ReLU())
        model.add_module(f'm{i}', nn.MaxPool2d(kernel_size=2))

    if has_dropout:
        model.add_module('dr1', nn.Dropout2d(p=0.5))

    model.add_module('fl3', torch.nn.Flatten(0, -1))
    model.add_module('l3', nn.Linear(38416, 10))
    model.add_module('s3', nn.Softmax(dim=-1))

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
