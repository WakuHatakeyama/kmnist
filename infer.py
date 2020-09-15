import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import ConvNet, weight_init
from data import KMNIST
from train import train

with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

BATCH_SIZE = config['batch_size']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    test_images = np.load(config['data']['test']['image'])['arr_0']
    test_labels = np.load(config['data']['test']['label'])['arr_0']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = KMNIST(test_images, test_labels, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ConvNet()
    model.load_state_dict(torch.load(config['pre_trained_model']))
    model.to(device)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test accuracy: {:.2f} %'.format(100*float(correct/total)))
