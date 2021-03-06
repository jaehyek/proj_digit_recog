from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
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


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class LayerActivations():
    features = None
    
    def __init__(self, model, layer_num):
        # print(model[layer_num])
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        print(f'module:{module}, input_shape:{input[0].shape}, output shape:{output.shape}')
        # self.features = output.cpu().data.numpy()
    
    def remove(self):
        self.hook.remove()
        

is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

train_transform = transforms.Compose([transforms.Resize((224, 224))
                                         , transforms.RandomHorizontalFlip()
                                         , transforms.RandomRotation(0.2)
                                         , transforms.ToTensor()
                                         , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

train = ImageFolder('digit_class_train/', train_transform)
valid = ImageFolder('digit_class_valid/', train_transform)

train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=16, shuffle=True)

vgg = models.vgg16(pretrained=True)
vgg = vgg.cuda()

vgg.classifier[6].out_features = 10
for param in vgg.features.parameters():
    param.requires_grad = False  # features ??? grad??? update?????? ????????? ?????????.

for layer in vgg.classifier.children():
    if (type(layer) == nn.Dropout):
        layer.p = 0.2

features = vgg.features


def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    
    for data in dataset:
        inputs, labels = data
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    
    return (conv_features, labels_list)

print(f'execute the preconvfeat')
conv_feat_train, labels_train = preconvfeat(train_data_loader, features)
conv_feat_val, labels_val = preconvfeat(valid_data_loader, features)
print(f'finish the preconvfeat')

class My_dataset(Dataset):
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels
    
    def __len__(self):
        return len(self.conv_feat)
    
    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]


train_feat_dataset = My_dataset(conv_feat_train, labels_train)
val_feat_dataset = My_dataset(conv_feat_val, labels_val)

train_feat_loader = DataLoader(train_feat_dataset, batch_size=64, shuffle=True)
val_feat_loader = DataLoader(val_feat_dataset, batch_size=64, shuffle=True)


# def data_gen(conv_feat, labels, batch_size=64, shuffle=True):
#     labels = np.array(labels)
#     if shuffle:
#         index = np.random.permutation(len(conv_feat))
#         conv_feat = conv_feat[index]
#         labels = labels[index]
#     for idx in range(0, len(conv_feat), batch_size):
#         yield (conv_feat[idx:idx + batch_size], labels[idx:idx + batch_size])
#
#
# train_batches = data_gen(conv_feat_train, labels_train)
# val_batches = data_gen(conv_feat_val, labels_val)

optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)




def fit_numpy(epoch, model, data_loader, phase='training', volatile=False):
    
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
        data = data.view(data.size(0), -1)
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
val_losses, val_accuracy = [], []
for epoch in range(1, 30):
    print(f'epoch:{epoch}')
    epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit_numpy(epoch, vgg.classifier, val_feat_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)



plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
plt.legend()
plt.show()

plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
plt.legend()
plt.show()
exit()



