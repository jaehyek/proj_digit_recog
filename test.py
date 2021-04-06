from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms, datasets
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time
import sys

is_cuda = None
optimizer = None

def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)

def makeDigitRecogModel(dir_train, dropout=0.5):
    global is_cuda, optimizer
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    
    simple_transform = transforms.Compose([transforms.Resize((128, 128))
                                              , transforms.RandomRotation(0.2)
                                              , transforms.ToTensor()
                                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
    train = ImageFolder(dir_train, simple_transform)

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)

    # sample_data = next(iter(train_data_loader))
    # imshow(sample_data[0][2])
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(16820, 500)
            self.fc2 = nn.Linear(500, 50)
            self.fc3 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, training=self.training)
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)
    
    
    model = Net()
    if is_cuda:
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    
    def fit(epoch, model, data_loader, phase='training', volatile=False):
        if phase == 'training':
            model.train()
        if phase == 'validation':
            model.eval()
            volatile = True
        running_loss = 0.0
        running_correct = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile), Variable(target)
            if phase == 'training':
                optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            running_loss += F.cross_entropy(output, target, reduction='sum').data
            preds = output.data.max(dim=1, keepdim=True)[1]
            running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
            if phase == 'training':
                loss.backward()
                optimizer.step()
        
        loss = running_loss / len(data_loader.dataset)
        accuracy = 100. * running_correct / len(data_loader.dataset)
        
        print(
            f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
        return loss, accuracy
    
    
    train_losses, train_accuracy = [], []

    for epoch in range(1, 200):
        print(f'epoch:{epoch}')
        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

    
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
    plt.legend()
    plt.show()

def get_svhn(dir_dataset, split):
    '''
    
    :param dir_dataset: =>  r'data/svhn'
    :param split: 'train', 'test', 'extra
    :return:
    '''
    svhn_dataset = datasets.SVHN(root=dir_dataset,
                                 split=split,
                                 download=True)

if __name__ == '__main__' :
    # makeDigitRecogModel(r'.\digit_class')
    get_svhn( r'data/svhn',  'train')
    get_svhn(r'data/svhn', 'test')
    get_svhn(r'data/svhn', 'extra')