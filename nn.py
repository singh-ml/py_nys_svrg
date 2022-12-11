#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 01:16:32 2021

@author: dineshsingh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# import pandas as pd


#set hyper parameters
epochs = 10
batch_size_train = 64
batch_size_test = 128
learning_rate = 0.01
momentum = 0.5
log_interval=10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#load pytirch MNIST  dataset which is indexable (Not dataloader)
dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

#split training data into training and validation set
train_size = int(0.8 * len(dataset_train))
val_size = len(dataset_train) - train_size
train_data, validation_data = torch.utils.data.random_split(dataset_train, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)

#model bulding
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

#model Optimization with SGD
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# training the model
validation_acc=[]
validation_losses=[]
train_losses = []
network.train()
for epoch in range(epochs):
    batch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):##put the trainloader data here
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: ## tuning of log_interval may be needed
            loss_epoch=loss.item()
        batch_loss.append(loss_epoch)
    loss_avg = sum(batch_loss)/len(batch_loss)
    train_losses.append(loss_avg)    
    network.eval() ### model validation 
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader: ## put the validationloader data here
            output = network(data)
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            arr_correct=correct.numpy()
    val_loss /= len(val_loader.dataset) 
    validation_losses.append(val_loss)
    print('Train Loss: %0.6f -- Val Loss: %0.6f',loss_avg, val_loss)
    acc_val = 100.00 * arr_correct.item() / len(val_loader.dataset) ### accuracy= (total currect prediction/ total samples)
    validation_acc.append(acc_val)
    model_saved=torch.save(network.state_dict(), './model.pth')
    optimizer_saved=torch.save(optimizer.state_dict(), './optimizer.pth')

#test the model
network.eval()
test_loss = 0
correct1 = 0
with torch.no_grad():
  for data, target in test_loader:## put testloader data 
    output = network(data)
    test_loss += F.nll_loss(output, target, size_average=False).item()
    pred = output.data.max(1, keepdim=True)[1]
    correct1 += pred.eq(target.data.view_as(pred)).sum()
    arr_correct1=correct1.numpy()
test_loss /= len(test_loader.dataset) 
acc_test = 100.00 * arr_correct1.item() / len(test_loader.dataset)
print(acc_test)  

# plot validation and train loss vs epoch
plt.figure()
plt.plot(range(len(train_losses)), train_losses, "r")
plt.plot(range(len(validation_losses)), validation_losses, "g")
plt.xlabel('epochs')
plt.ylabel(' loss')
plt.savefig('./nn.png')
##set path here according to folder path

#lis={train_losses, validation_losses, validation_acc  }
df=pd.DataFrame(list(zip(train_losses, validation_losses, validation_acc)),
              columns=['train_losses','validation_losses', 'validation_acc'])
#%%
df.to_csv('./file1.csv')

####train loss, validation loss and validation accuracy for all epochs will be saved in the csv file automatically
