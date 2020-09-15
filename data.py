import torch
from torch.utils.data import Dataset

class KMNIST(Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label