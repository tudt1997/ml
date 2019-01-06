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
        self.samples = torch.tensor(data[:, :6].transpose().astype(float))
        self.labels = torch.tensor(data[:, 6].astype(int))
        print(self.samples)
        self.window_size = window_size
        self.step = window_size // 2
        
    def __getitem__(self, index):
        frame = self.samples[:, index * self.step:index * self.step + self.window_size]
        # label = self.labels[index * self.step]
        return frame

    def __len__(self):
        return (self.labels.shape[0] // self.step - 1)

class TripletDataset(Dataset):
    def __init__(self, pos_dataset, neg_dataset):
        if pos_dataset.__len__() >= 2 * neg_dataset.__len__():
            self.size = neg_dataset.__len__()
            self.data_anchor, self.data_pos, _ = torch.utils.data.random_split(pos_dataset, [self.size, self.size, pos_dataset.__len__() - 2 * self.size])
            self.data_neg = neg_dataset
        # else:
            # TO DO

    def __len__(self):
        return self.size 

    def __getitem__(self, index):
        return self.data_anchor.__getitem__(index), self.data_pos.__getitem__(index), self.data_neg.__getitem__(index)

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 96, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(96, 192, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(192, 192, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192 * 44, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 20),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

batch_size = 128

pos_dataset = FallDataset('data/positive.csv')
neg_dataset = FallDataset('data/negative.csv')
triplet_dataset = TripletDataset(pos_dataset, neg_dataset)
size = triplet_dataset.__len__()
print('Size: %d' % (size))
triplet_dataloader = torch.utils.data.DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

test_pos_dataset = FallDataset('data/test_positive.csv')
test_neg_dataset = FallDataset('data/test_negative.csv')
test_triplet_dataset = TripletDataset(test_pos_dataset, test_neg_dataset)
test_triplet_dataloader = torch.utils.data.DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
size = test_triplet_dataset.__len__()
print('Size: %d' % (size))

########################################################################
# Define a loss function and optimizer

import torch.optim as optim

lr = float(input('Enter learning rate: '))

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-3)

########################################################################
# Train the network

epoch_num = int(input('Enter number of epoch: '))

print('Learning rate: %f' % (lr))
print('Number of epoch: %f' % (epoch_num))

for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(triplet_dataloader, 0):
        # get the inputs
        inputs_anchor, inputs_pos, inputs_neg = data
        inputs_anchor, inputs_pos, inputs_neg = inputs_anchor.to(device, dtype=torch.float), inputs_pos.to(device, dtype=torch.float), inputs_neg.to(device, dtype=torch.float) # Send to GPU
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_anchor, outputs_pos, outputs_neg = net(inputs_anchor), net(inputs_pos), net(inputs_neg)
        if i == 0 and epoch == 0:
            print(outputs_anchor.shape)
        loss = triplet_loss(outputs_anchor, outputs_pos, outputs_neg)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('Epoch %d loss: %.3f' %
            (epoch + 1, running_loss))

print('Finished Training')
########################################################################
# Save extracted features
save = input('Extract features? [Y/n]: ')
if save == '' or save == 'y':
    open('extracted/test_anchor.csv', 'w')
    open('extracted/test_positive.csv', 'w')
    open('extracted/test_negative.csv', 'w')
    with torch.no_grad():
        for data in test_triplet_dataloader:
            # get the inputs
            inputs_anchor, inputs_pos, inputs_neg = data
            inputs_anchor, inputs_pos, inputs_neg = inputs_anchor.to(device, dtype=torch.float), inputs_pos.to(device, dtype=torch.float), inputs_neg.to(device, dtype=torch.float) # Send to GPU

            outputs_anchor, outputs_pos, outputs_neg = net(inputs_anchor), net(inputs_pos), net(inputs_neg)

            with open('extracted/test_anchor.csv', 'a') as f:
                for i in range(len(outputs_anchor)):
                    f.write(",".join(map(str, outputs_anchor[i, :].cpu().numpy())) + '\n')

            with open('extracted/test_positive.csv', 'a') as f:
                for i in range(len(outputs_pos)):
                    f.write(",".join(map(str, outputs_pos[i, :].cpu().numpy())) + '\n')

            with open('extracted/test_negative.csv', 'a') as f:
                for i in range(len(outputs_neg)):
                    f.write(",".join(map(str, outputs_neg[i, :].cpu().numpy())) + '\n')
    print('Features extracted')

########################################################################
# Calculate accuracy

# correct_train = 0
# correct_test = 0
# total_train = 0
# total_test = 0
# with torch.no_grad():
#     for data in trainloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device) # Send to GPU
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device) # Send to GPU
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total_test += labels.size(0)
#         correct_test += (predicted == labels).sum().item()

# print('Train accuracy: %d %%' % (100 * correct_train / total_train))