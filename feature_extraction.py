import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import csv

class FallDataset(Dataset):
    def __init__(self, file_name, window_size=100):
        raw_data = open(file_name, 'r')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        data = list(reader)
        data = np.array(data)
        data[data[:, 8] == "non_fall", 8] = 1
        data[data[:, 8] == "fall", 8] = -1
        print(data)
        self.samples = torch.tensor(data[:, 2:8].transpose().astype(float))
        self.labels = torch.tensor(data[:, 8].astype(int))
        print(self.samples)
        print(self.labels)
        self.window_size = window_size
        self.step = window_size // 2
        
    def __getitem__(self, index):
        frame = self.samples[:, index * self.step:index * self.step + self.window_size]
        label = self.labels[index * self.window_size]
        return (frame, label)

    def __len__(self):
        return (self.labels.shape[0] // self.step - 1)

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 36, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=1),
            nn.Conv1d(36, 72, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

########################################################################
# Set the device to the first cuda device if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

net = Net()
net = net.to(device) # Send to GPU

########################################################################
# Load data and apply normalization

batch_size = 64
dataset = FallDataset('./../data/mica/dataset_har_2/train.csv')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

for data in dataloader:
    frames, labels = data
    print(frames.shape)
    break
########################################################################
# Define a loss function and optimizer

import torch.optim as optim

# lr = float(input('Enter learning rate: '))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)