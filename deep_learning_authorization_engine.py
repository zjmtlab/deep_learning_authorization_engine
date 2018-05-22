#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

# Hyper Parameters 
input_size = 106
hidden_size = 50
num_classes = 2
num_epochs = 10
batch_size = 10
learning_rate = 0.1

resources = [0, 1, 2, 3]
train_data_num = 1000
test_data_num= 40

"""
The policies in training data:
   group[0] can access resource[0]
   group[1] can access resource[1]
   odd user id can access resource[2]
   even user id can access resource[3]
"""

def getRow(resource, fromIndex, toIndex, isTrain):
    uId = random.randint(fromIndex, toIndex)
    userOneHot = torch.zeros(1, 100)
    userOneHot[0][uId] = 1
    group = random.randint(0, 1)
    permission = 1

    if (resource == resources[0] and group == 0) \
        or (resource == resources[1] and group == 1) \
        or (resource == resources[2] and uId%2 == 0) \
        or (resource == resources[3] and uId%2 == 1):
        permission = 0

    groupOneHot = torch.zeros(1, 2)
    groupOneHot[0][group] =1
    resourceOneHot = torch.zeros(1, len(resources))
    resourceOneHot[0][resource] = 1
    permissionOneHot = torch.LongTensor([permission])

    features = torch.cat((userOneHot, groupOneHot, resourceOneHot), 1)
    return features, permissionOneHot


def getData(userCount, fromIndex, toIndex, isTrain):
    global userIndex
    x = None
    y = None
    for i in range(0, userCount):
        for r in range(0, len(resources)):
            row, target = getRow(resources[r], fromIndex, toIndex, isTrain)
            if i == 0 and r == 0:
                x = row
                y = target
            else:
                x = torch.cat((x, row), 0)
                y = torch.cat((y, target), 0)

    return x, y

x_train, y_train = getData(train_data_num, 0, 99, True)

x_test, y_test = getData(test_data_num, 71, 99, False)

class MyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs = sequences
        self.labels = labels

    def __getitem__(self, index):
        seq, target = self.seqs[index], self.labels[index]
        return seq, target

    def __len__(self):
        return len(self.seqs)

train_dataloader = DataLoader(MyDataset(x_train, y_train), batch_size=batch_size)
test_dataloader = DataLoader(MyDataset(x_test, y_test), batch_size=batch_size)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)
    
net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.NLLLoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for data, labels in train_dataloader:  
        # Convert torch tensor to Variable
        data = Variable(data)
        labels = Variable(labels)
        
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(data)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
         
        print ('Epoch [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.item()))

# Test the Model
correct = 0
total = 0
for data, labels in test_dataloader:
    data = Variable(data)
    outputs = net(data)
    _, result = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (result == labels).sum()

print('Accuracy of the network on the 30 test data: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
