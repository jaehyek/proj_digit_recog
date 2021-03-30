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
    plt.show()
    plt.close()


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
        
def preconvfeat(dataset, model):
    global is_cuda
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


class My_dataset(Dataset):
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels
    
    def __len__(self):
        return len(self.conv_feat)
    
    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]


def fit_numpy(epoch, model, data_loader, phase='training', volatile=False):
    global is_cuda, optimizer
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
        f'epoch:{epoch}, {phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

def DigitRecogModel(dir_train, dropout=0.5, phase='training', model_path_load=None, model_path_save=r'./model.pt'):
    global is_cuda, optimizer
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    
    train_transform = transforms.Compose([transforms.Resize((224, 224))
                                             , transforms.RandomRotation(0.2)
                                             , transforms.ToTensor()
                                             , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    train = ImageFolder(dir_train, train_transform)

    # train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, num_workers= 1 ) # debug에서는 num_workers을 사용하지 않는다.
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    
    # img, label = next(iter(train_data_loader))
    # imshow(img[0])
    
    if phase == 'training'  and model_path_load == None:
        vgg = models.vgg16(pretrained=True)
        vgg.classifier[6].out_features = 10
        # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
        if is_cuda == True:
            vgg = vgg.cuda()
        optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
        
    else:   # phase == 'validation'
        if model_path_load == None:
            print('model_path must not be None.')
            raise Exception("model_path must not be None.")
            
        vgg = models.vgg16(pretrained=False)
        vgg.classifier[6].out_features = 10
        # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
        if is_cuda == True:
            vgg = vgg.cuda()
        optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)

        checkpoint = torch.load(model_path_load)
        vgg.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for layer in vgg.classifier.children():
        if (type(layer) == nn.Dropout):
            layer.p = dropout

    print(f'___ execute the preconvfeat ___')
    
    for param in vgg.features.parameters():
        param.requires_grad = False  # features 는 grad가 update되지 못하게 막는다.

    features = vgg.features
    conv_feat_train, labels_train = preconvfeat(train_data_loader, features)

    print(f'___ finish the preconvfeat ___')

    train_feat_dataset = My_dataset(conv_feat_train, labels_train)
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=16, shuffle=True)

    

    train_losses, train_accuracy = [], []
    epoch_loop = 200 if phase=='training' else 1
    for epoch in range(epoch_loop):
        epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase=phase)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)


    # plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='loss')
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='accuracy')
    # plt.legend()
    # plt.show()

    if model_path_save != None:
        torch.save({
            'model_state_dict': vgg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path_save)
    
    return train_losses[-1], train_accuracy[-1]

if __name__ == '__main__' :
    # 처음 학습시킬때.
    # loss, acc = DigitRecogModel(r'.\digit_class_train', dropout=0.2, phase='training', model_path_load = None, model_path_save=r'./model.pt')
    # print(f'Model  Traning  loss:{loss}, accuracy:{acc}')
    
    # 학습된 것을 이어 받아서  학습할 때,
    loss, acc = DigitRecogModel(r'.\digit_class_valid', dropout=0.2, phase='training', model_path_load=r'./model.pt', model_path_save=r'./model.pt')
    print(f'Model  validation  loss:{loss}, accuracy:{acc}')
    
    # 학습을 확인하고자 할 때,
    loss, acc = DigitRecogModel(r'.\digit_class_valid', dropout=0.2, phase='validation', model_path_load=r'./model.pt',model_path_save=None)
    print(f'Model  validation  loss:{loss}, accuracy:{acc}')
    