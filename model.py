import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2, padding=2, padding_mode='replicate')
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=2, padding_mode='replicate')
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=2, padding_mode='replicate')
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(64, 128)
        self.dropout3 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout1(self.bn3(self.pool(F.relu(self.conv3(x)))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.dropout2(self.pool(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvNet()
    model = model.to(device)
    summary(model, (1, 28, 28))
