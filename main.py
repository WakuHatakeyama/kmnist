import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from model import ConvNet, weight_init, resnet18
from data import KMNIST
from train import train

with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

TEST_SIZE = config['data']['test_size']
RANDOM_STATE = config['data']['random_state']

NUM_EPOCHS = config['num_epoch']
BATCH_SIZE = config['batch_size']

NAMES = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

if __name__ == '__main__':
    train_images = np.load(config['data']['train']['image'])['arr_0']
    train_labels = np.load(config['data']['train']['label'])['arr_0']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_set = KMNIST(X_train, y_train, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_set = KMNIST(X_val, y_val, transform=transform)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = resnet18()
    # model = ConvNet()
    # model.apply(weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train(train_loader, val_loader, model, criterion, optimizer, n_epochs=NUM_EPOCHS)