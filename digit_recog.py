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

    # conv_out0 = LayerActivations(model, 0 )
    # conv_out1 = LayerActivations(model, 1 )
    # conv_out2 = LayerActivations(model, 2 )
    # conv_out3 = LayerActivations(model, 3 )
    # conv_out4 = LayerActivations(model, 4 )
    # conv_out5 = LayerActivations(model, 5 )
    # conv_out6 = LayerActivations(model, 6 )
    # conv_out7 = LayerActivations(model, 7 )
    # conv_out8 = LayerActivations(model, 8 )
    # conv_out9 = LayerActivations(model, 9 )
    # conv_out10= LayerActivations(model, 10)
    # conv_out11= LayerActivations(model, 11)
    # conv_out12= LayerActivations(model, 12)
    # conv_out13= LayerActivations(model, 13)
    # conv_out14= LayerActivations(model, 14)
    # conv_out15= LayerActivations(model, 15)
    # conv_out16= LayerActivations(model, 16)
    # conv_out17= LayerActivations(model, 17)
    # conv_out18= LayerActivations(model, 18)
    # conv_out19= LayerActivations(model, 19)
    # conv_out20= LayerActivations(model, 20)
    # conv_out21= LayerActivations(model, 21)
    # conv_out22= LayerActivations(model, 22)
    # conv_out23= LayerActivations(model, 23)
    # conv_out24= LayerActivations(model, 24)
    # conv_out25= LayerActivations(model, 25)
    # conv_out26= LayerActivations(model, 26)
    # conv_out27= LayerActivations(model, 27)
    # conv_out28= LayerActivations(model, 28)
    # conv_out29= LayerActivations(model, 29)
    # conv_out30 = LayerActivations(model, 30)
    

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
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

def makeDigitRecogModel(dir_train, dropout=0.5, phase='training', model_path=r'./model.pth'):
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

    vgg = models.vgg16(pretrained=True).cuda()

    for layer in vgg.classifier.children():
        if (type(layer) == nn.Dropout):
            layer.p = dropout
    
    # if is_cuda == True :
    #     vgg.classifier[0] = nn.Linear(in_features=8192, out_features=4096, bias=True).cuda()
    # else:
    #     vgg.classifier[0] = nn.Linear(in_features=8192, out_features=4096, bias=True)
    vgg.classifier[6].out_features = 10
    for param in vgg.features.parameters():
        param.requires_grad = False  # features 는 grad가 update되지 못하게 막는다.
    
    features = vgg.features


    print(f'execute the preconvfeat')
    conv_feat_train, labels_train = preconvfeat(train_data_loader, features)
    print(f'finish the preconvfeat')

    train_feat_dataset = My_dataset(conv_feat_train, labels_train)
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=16, shuffle=True)

    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)

    train_losses, train_accuracy = [], []
    for epoch in range(1, 200):
        print(f'epoch:{epoch}')
        epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase=phase)
       
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)


    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__' :
    makeDigitRecogModel(r'.\digit_class_train', phase='training', model_path=r'./model.pt')
    # makeDigitRecogModel(r'.\digit_class_train', phase='validation')