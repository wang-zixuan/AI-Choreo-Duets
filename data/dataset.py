import torch


class DancerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, index):
        return {
            'dancer1': self.data[index // 2],
            'dancer2': self.data[index // 2 + 1]
        }
