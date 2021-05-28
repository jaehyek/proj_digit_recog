from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
from torchvision import transforms
import torchvision.transforms.functional as TF
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
from makeImageFolder import get_Image_Value_List_from_json
import random
from PIL import Image

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


def imsave(inp, filename):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    matplotlib.image.imsave(filename, inp)


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


train_transform = transforms.Compose([transforms.Resize((224, 224))
                                         # , transforms.RandomRotation(0.2)
                                         # , transforms.ColorJitter(brightness=0.1)
                                         , transforms.ToTensor()
                                         , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])


def DigitRecogModel(dir_train, dropout=0.5, phase='training', model_path_load=None, model_path_save=r'./model.pt'):
    global is_cuda, optimizer, train_transform
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    
    # setting parameter
    batch_size = 5
    epoch_loop = 150 if phase == 'training' else 1
    
    train = ImageFolder(dir_train, train_transform)
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    # train_data_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    # for i, (data, target) in enumerate( train_data_loader) :
    #     imsave(data[0], os.path.join('test', str(target[0].tolist()) + '_' + str(i) + '.jpg'))
    #
    # exit()
    
    if phase == 'training' and model_path_load == None:
        print('vgg16(pretrained=True)')
        vgg = models.vgg16(pretrained=True)
    else:
        print('vgg16(pretrained=False)')
        vgg = models.vgg16(pretrained=False)
    
    vgg.classifier[6].out_features = 10
    
    if is_cuda == True:
        vgg = vgg.cuda()
    
    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
    
    if model_path_load != None:
        if is_cuda == True:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
            
        print(f'load from {model_path_load}')
        checkpoint = torch.load(model_path_load, map_location=dev)
        vgg.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # check point
    if phase == 'validation' and model_path_load == None:
        print('model_path must not be None.')
        raise Exception("model_path must not be None.")
    
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
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses, train_accuracy = [], []
    
    for epoch in range(epoch_loop):
        epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase=phase)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
    
    if model_path_save != None:
        torch.save({
            'model_state_dict': vgg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path_save)
    
    return train_losses[-1], train_accuracy[-1]


class DigitGaugeDataset(Dataset):
    def __init__(self, file_json, transform):
        list_image, list_value, dict_json_info = get_Image_Value_List_from_json(file_json)
        
        self.list_image = list_image
        self.list_label = list_value
        self.dict_json_info = dict_json_info
        self.transform = transform
    
    def __len__(self):
        # return size of dataset
        return len(self.list_image)
    
    def __getitem__(self, idx):
        image = self.list_image[idx]
        image = self.transform(image)
        return image, self.list_label[idx]


def getValueFromJson(file_json, model_path_load):
    global is_cuda, optimizer, train_transform
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    
    gaugedataset = DigitGaugeDataset(file_json, train_transform)
    len_digit = len(gaugedataset)
    
    gauge_data_loader = torch.utils.data.DataLoader(gaugedataset, batch_size=len_digit, shuffle=False)
    data, target = next(iter(gauge_data_loader))
    
    vgg = models.vgg16(pretrained=False)
    vgg.classifier[6].out_features = 10
    if is_cuda == True:
        vgg = vgg.cuda()
    
    if is_cuda == True:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
        
    checkpoint = torch.load(model_path_load, map_location=dev)
    vgg.load_state_dict(checkpoint['model_state_dict'])
    
    for layer in vgg.classifier.children():
        if (type(layer) == nn.Dropout):
            layer.train(False)
            layer.p = 0
    
    vgg.eval()
    # vgg.train()
    volatile = True
    
    if is_cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile), Variable(target)
    
    output = vgg(data)
    preds = output.data.max(dim=1, keepdim=True)[1]
    count_correct = preds.eq(target.data.view_as(preds)).cpu().sum()
    
    list_preds = preds.view(-1).tolist()
    print(f'target:{target.view(-1).tolist()}, preds:{list_preds}, accuracy :{count_correct / len_digit}')
    npred = int(''.join([str(aa) for aa in list_preds]))
    
    if gaugedataset.dict_json_info['digitFractNo'] > 0:
        npred = npred / (10 ** gaugedataset.dict_json_info['digitFractNo'])
    return npred


if __name__ == '__main__':
    time_start = time.time()
    # 처음 학습시킬때.
    loss, acc = DigitRecogModel(r'.\digit_class_aug', dropout=0.2, phase='training', model_path_load=None,
                                model_path_save=r'./model.pt')
    print(f'Model  Traning  loss:{loss}, accuracy:{acc}')

    # 기존학습에 추가 학습시킬때.
    # loss, acc = DigitRecogModel(r'.\digit_class_aug', dropout=0.2, phase='training', model_path_load=r'./model.pt',
    #                             model_path_save=r'./model.pt')
    # print(f'Model  Traning  loss:{loss}, accuracy:{acc}')
    
    # test할 json을 받아서  예측한 값을 확인.
    print('-------------------------------------------------------------')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10050-56359.json', model_path_load=r'./model.pt')
    print('~~~~~~~~~~~~~~~')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56385.json', model_path_load=r'./model.pt')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56394.json', model_path_load=r'./model.pt')
    print('~~~~~~~~~~~~~~~')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56400.json', model_path_load=r'./model.pt')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56455.json', model_path_load=r'./model.pt')
    
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56265.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56266.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56267.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56268.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56340.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56341.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56342.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56343.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56344.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56345.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56346.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56347.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56348.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56349.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56350.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56351.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56352.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56353.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56354.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56355.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56356.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56357.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56358.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56370.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56371.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56372.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56373.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56374.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56375.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56376.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56377.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56378.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56379.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56380.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56381.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56383.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56384.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56386.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56387.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56388.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56389.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56390.json', model_path_load=r'./model.pt')
    print('~~~~~~~~~~~~~~~')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56391.json', model_path_load=r'./model.pt')
    print('~~~~~~~~~~~~~~~')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56392.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56393.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56397.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56398.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56399.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56401.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56402.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56403.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56404.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56405.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56406.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56407.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56408.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56409.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56413.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56430.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56431.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56432.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56433.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56434.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56435.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56436.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56437.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56438.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56439.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56440.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56441.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56442.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56443.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56444.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56445.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56446.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56447.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56448.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56449.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56450.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56451.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56452.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56453.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56454.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56458.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56459.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56460.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56461.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56462.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56463.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56464.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56465.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56466.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56467.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56468.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56469.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56470.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56471.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56472.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56473.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56474.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56475.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56476.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56485.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56486.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56487.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56488.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56489.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56490.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56491.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56492.json', model_path_load=r'./model.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56493.json', model_path_load=r'./model.pt')
    print(f'elapsed time sec  : {time.time() - time_start}')

