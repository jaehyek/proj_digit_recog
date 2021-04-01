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
from makeImageFolder import get_Image_Value_List_from_json
import random

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
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    
    # img, label = next(iter(train_data_loader))
    # imshow(img[0])
    
    if phase == 'training'  and model_path_load == None:
        vgg = models.vgg16(pretrained=True)
    else:
        vgg = models.vgg16(pretrained=False)
        
    vgg.classifier[6].out_features = 10
    if is_cuda == True:
        vgg = vgg.cuda()
    optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
    # optimizer = optim.Adam(vgg.classifier.parameters(), lr=1.0)

    if model_path_load != None:
        checkpoint = torch.load(model_path_load)
        vgg.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # check point
    if phase == 'validation' and model_path_load == None:
        print('model_path must not be None.')
        raise Exception("model_path must not be None.")
    
    # if phase == 'training'  and model_path_load == None:
    #     vgg = models.vgg16(pretrained=True)
    #     vgg.classifier[6].out_features = 10
    #     # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
    #     if is_cuda == True:
    #         vgg = vgg.cuda()
    #     optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
    #
    # else:   # phase == 'validation'
    #     if model_path_load == None:
    #         print('model_path must not be None.')
    #         raise Exception("model_path must not be None.")
    #
    #     vgg = models.vgg16(pretrained=False)
    #     vgg.classifier[6].out_features = 10
    #     # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
    #     if is_cuda == True:
    #         vgg = vgg.cuda()
    #     optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
    #
    #     checkpoint = torch.load(model_path_load)
    #     vgg.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
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
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=4, shuffle=True)

    

    train_losses, train_accuracy = [], []
    epoch_loop = 200 if phase=='training' else 1
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,180], gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=20,gamma=0.4)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 1 / (epoch+1))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=3, )
    
    for epoch in range(epoch_loop):
        epoch_loss, epoch_accuracy = fit_numpy(epoch, vgg.classifier, train_feat_loader, phase=phase)
        # if phase=='training':
        #     scheduler.step(random.randint(1, 50))
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
    global is_cuda, optimizer
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    
    train_transform = transforms.Compose([transforms.Resize((224, 224))
                                             , transforms.RandomRotation(0.2)
                                             , transforms.ToTensor()
                                             , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    
    gaugedataset = DigitGaugeDataset(file_json, train_transform)
    len_digit = len(gaugedataset)
    
    gauge_data_loader = torch.utils.data.DataLoader(gaugedataset, batch_size=len_digit, shuffle=False)
    data, target = next(iter(gauge_data_loader))

    vgg = models.vgg16(pretrained=False)
    vgg.classifier[6].out_features = 10
    if is_cuda == True:
        vgg = vgg.cuda()

    checkpoint = torch.load(model_path_load)
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
    print(f'target:{target.view(-1).tolist()}, preds:{list_preds}, accuracy :{count_correct/len_digit}')
    npred = int(''.join([str(aa) for aa in list_preds]))
    
    if gaugedataset.dict_json_info['digitFractNo'] > 0 :
        npred = npred / ( 10 ** gaugedataset.dict_json_info['digitFractNo'])
    return npred
    
if __name__ == '__main__' :
    time_start = time.time()
    # 처음 학습시킬때.
    # loss, acc = DigitRecogModel(r'.\digit_class_train', dropout=0.2, phase='training', model_path_load = None, model_path_save=r'./model.pt')
    # print(f'Model  Traning  loss:{loss}, accuracy:{acc}')
    
    # 학습된 것을 이어 받아서  학습할 때,
    # print('-------------------------------------------------------------')
    # loss, acc = DigitRecogModel(r'.\digit_class_valid', dropout=0.2, phase='training', model_path_load=r'./model.pt', model_path_save=r'./model.pt')
    # print(f'Model  validation  loss:{loss}, accuracy:{acc}')
    
    # 학습을 확인하고자 할 때,
    # print('-------------------------------------------------------------')
    # loss, acc = DigitRecogModel(r'.\digit_class_valid', dropout=0.2, phase='validation', model_path_load=r'./model.pt',model_path_save=None)
    # print(f'Model  validation  loss:{loss}, accuracy:{acc}')
    
    # test할 json을 받아서  예측한 값을 확인.
    # print('-------------------------------------------------------------')
    # npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10050-56359.json', model_path_load=r'./model.pt')
    # npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56385.json', model_path_load=r'./model.pt')
    # npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56394.json', model_path_load=r'./model.pt')
    # npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56400.json', model_path_load=r'./model.pt')
    # npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56455.json', model_path_load=r'./model.pt')

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
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56391.json', model_path_load=r'./model.pt')
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
    print(f'걸련시간 : {time.time() - time_start }')
    
    
'''
epoch:194, training loss is 0.0048 and training accuracy is 5337/5343     99.89
epoch:195, training loss is 0.0044 and training accuracy is 5340/5343     99.94
epoch:196, training loss is 0.004 and training accuracy is 5340/5343     99.94
epoch:197, training loss is 0.0049 and training accuracy is 5340/5343     99.94
epoch:198, training loss is 0.0035 and training accuracy is 5342/5343     99.98
epoch:199, training loss is 0.0041 and training accuracy is 5342/5343     99.98
Model  Traning  loss:0.0041223145090043545, accuracy:99.98128509521484
-------------------------------------------------------------
epoch:195, training loss is 0.0038 and training accuracy is 1400/1401     99.93
epoch:196, training loss is 0.0023 and training accuracy is 1401/1401     100.0
epoch:197, training loss is 0.0029 and training accuracy is 1401/1401     100.0
epoch:198, training loss is 0.0031 and training accuracy is 1401/1401     100.0
epoch:199, training loss is 0.0028 and training accuracy is 1401/1401     100.0
Model  validation  loss:0.0028244229033589363, accuracy:100.0
-------------------------------------------------------------
___ execute the preconvfeat ___
___ finish the preconvfeat ___
epoch:0, validation loss is 0.0013 and validation accuracy is 1401/1401     100.0
Model  validation  loss:0.0013372735120356083, accuracy:100.0
-------------------------------------------------------------
target:[0, 2, 0, 5], preds:[0, 2, 7, 5], accuracy :0.75
target:[0, 4, 0, 9, 6], preds:[0, 4, 0, 3, 6], accuracy :0.800000011920929
target:[9, 5, 4, 6, 2, 1, 1, 5], preds:[9, 6, 4, 6, 2, 1, 1, 0], accuracy :0.75
target:[9, 7, 5, 7, 4, 2, 1, 5], preds:[9, 7, 3, 7, 4, 2, 1, 5], accuracy :0.875
target:[0, 9, 4, 9, 8, 4, 0, 5], preds:[0, 9, 4, 9, 8, 4, 0, 5], accuracy :1.0

target:[0, 2, 1, 1], preds:[0, 2, 1, 1], accuracy :1.0
target:[0, 2, 2, 9], preds:[0, 2, 2, 9], accuracy :1.0
target:[0, 0, 7, 7], preds:[0, 0, 7, 7], accuracy :1.0
target:[0, 0, 1, 4], preds:[0, 0, 1, 4], accuracy :1.0
target:[0, 0, 2, 3], preds:[0, 0, 2, 3], accuracy :1.0
target:[0, 0, 3, 6], preds:[0, 0, 3, 6], accuracy :1.0
target:[0, 0, 4, 3], preds:[0, 0, 4, 3], accuracy :1.0
target:[0, 0, 4, 9], preds:[0, 0, 4, 9], accuracy :1.0
target:[0, 0, 5, 6], preds:[0, 0, 5, 6], accuracy :1.0
target:[0, 0, 6, 5], preds:[0, 0, 6, 5], accuracy :1.0
target:[0, 0, 7, 1], preds:[0, 0, 7, 1], accuracy :1.0
target:[0, 0, 8, 0], preds:[0, 0, 8, 0], accuracy :1.0
target:[0, 0, 8, 9], preds:[0, 0, 8, 9], accuracy :1.0
target:[0, 0, 9, 8], preds:[0, 0, 9, 8], accuracy :1.0
target:[0, 1, 0, 4], preds:[0, 1, 0, 4], accuracy :1.0
target:[0, 1, 1, 5], preds:[0, 1, 1, 5], accuracy :1.0
target:[0, 1, 2, 3], preds:[0, 1, 2, 3], accuracy :1.0
target:[0, 1, 3, 2], preds:[0, 1, 3, 2], accuracy :1.0
target:[0, 1, 4, 9], preds:[0, 1, 4, 9], accuracy :1.0
target:[0, 1, 5, 8], preds:[0, 1, 5, 8], accuracy :1.0
target:[0, 1, 7, 0], preds:[0, 1, 7, 0], accuracy :1.0
target:[0, 1, 8, 2], preds:[0, 1, 8, 2], accuracy :1.0
target:[0, 1, 9, 3], preds:[0, 1, 9, 3], accuracy :1.0
target:[0, 1, 8, 6], preds:[0, 1, 8, 6], accuracy :1.0
target:[0, 1, 7, 6], preds:[0, 1, 7, 6], accuracy :1.0
target:[0, 1, 6, 1], preds:[0, 1, 6, 1], accuracy :1.0
target:[0, 1, 4, 5], preds:[0, 1, 4, 5], accuracy :1.0
target:[0, 1, 2, 5], preds:[0, 1, 2, 5], accuracy :1.0
target:[0, 0, 8, 4], preds:[0, 0, 8, 4], accuracy :1.0
target:[0, 0, 7, 3], preds:[0, 0, 7, 3], accuracy :1.0
target:[0, 0, 6, 1], preds:[0, 0, 6, 1], accuracy :1.0
target:[0, 0, 5, 5], preds:[0, 0, 5, 5], accuracy :1.0
target:[0, 0, 4, 2], preds:[0, 0, 4, 2], accuracy :1.0
target:[0, 0, 3, 2], preds:[0, 0, 3, 2], accuracy :1.0
target:[0, 0, 2, 3], preds:[0, 0, 2, 3], accuracy :1.0
target:[0, 2, 0, 4], preds:[0, 2, 0, 4], accuracy :1.0
target:[0, 2, 0, 4, 8], preds:[0, 2, 0, 4, 8], accuracy :1.0
target:[0, 0, 1, 9, 2], preds:[0, 0, 1, 9, 2], accuracy :1.0
target:[1, 6, 3, 8, 4], preds:[1, 6, 3, 8, 4], accuracy :1.0
target:[3, 2, 7, 6, 8], preds:[3, 2, 7, 6, 8], accuracy :1.0
target:[6, 5, 5, 3, 6], preds:[6, 5, 5, 3, 6], accuracy :1.0
target:[1, 9, 2, 0, 0], preds:[1, 9, 2, 0, 0], accuracy :1.0
target:[3, 8, 4, 0, 0], preds:[3, 8, 4, 0, 0], accuracy :1.0
target:[5, 7, 6, 0, 0], preds:[5, 7, 6, 0, 0], accuracy :1.0
target:[0, 1, 0, 2, 4], preds:[7, 1, 0, 2, 4], accuracy :0.800000011920929
target:[9, 5, 4, 6, 2, 1, 1, 5], preds:[9, 4, 4, 6, 2, 1, 1, 5], accuracy :0.875
target:[9, 6, 4, 6, 2, 1, 1, 5], preds:[9, 4, 4, 6, 2, 1, 1, 5], accuracy :0.875
target:[9, 7, 5, 6, 2, 1, 1, 5], preds:[9, 7, 8, 6, 2, 4, 1, 5], accuracy :0.75
target:[9, 7, 6, 8, 5, 2, 1, 5], preds:[9, 7, 6, 8, 3, 2, 1, 5], accuracy :0.875
target:[0, 7, 6, 9, 5, 2, 1, 5], preds:[0, 7, 6, 9, 3, 2, 1, 5], accuracy :0.875
target:[0, 8, 7, 9, 5, 2, 1, 5], preds:[0, 8, 7, 9, 3, 2, 1, 5], accuracy :0.875
target:[0, 8, 8, 0, 5, 2, 1, 5], preds:[0, 8, 8, 0, 3, 2, 1, 5], accuracy :0.875
target:[0, 8, 9, 1, 5, 2, 1, 5], preds:[0, 8, 9, 1, 3, 2, 1, 5], accuracy :0.875
target:[0, 8, 9, 2, 6, 2, 1, 5], preds:[0, 8, 9, 2, 6, 2, 1, 5], accuracy :1.0
target:[0, 8, 9, 3, 7, 2, 1, 5], preds:[0, 8, 9, 3, 7, 2, 1, 5], accuracy :1.0
target:[0, 8, 9, 4, 8, 2, 1, 5], preds:[0, 8, 9, 4, 8, 2, 1, 5], accuracy :1.0
target:[0, 8, 9, 5, 9, 2, 1, 5], preds:[0, 8, 9, 8, 9, 2, 1, 5], accuracy :0.875
target:[0, 9, 1, 9, 3, 3, 1, 5], preds:[0, 9, 1, 9, 3, 3, 1, 5], accuracy :1.0
target:[0, 9, 4, 7, 3, 3, 0, 5], preds:[0, 9, 4, 7, 3, 3, 0, 5], accuracy :1.0
target:[0, 9, 4, 7, 4, 4, 0, 5], preds:[0, 9, 4, 7, 4, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 7, 5, 4, 0, 5], preds:[0, 9, 4, 7, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 4, 7, 6, 4, 0, 5], preds:[0, 9, 4, 7, 6, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 7, 7, 4, 0, 5], preds:[0, 9, 4, 7, 7, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 7, 8, 4, 0, 5], preds:[0, 9, 4, 7, 8, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 7, 9, 4, 0, 5], preds:[0, 9, 4, 7, 9, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 0, 4, 0, 5], preds:[0, 9, 4, 8, 0, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 1, 4, 0, 5], preds:[0, 9, 4, 8, 1, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 2, 4, 0, 5], preds:[0, 9, 4, 8, 2, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 3, 4, 0, 5], preds:[0, 9, 4, 8, 3, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 4, 4, 0, 5], preds:[0, 9, 4, 8, 4, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 5, 4, 0, 5], preds:[0, 9, 4, 8, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 4, 8, 6, 4, 0, 5], preds:[0, 9, 4, 8, 6, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 7, 4, 0, 5], preds:[0, 9, 4, 8, 7, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 8, 4, 0, 5], preds:[0, 9, 4, 8, 8, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 8, 9, 4, 0, 5], preds:[0, 9, 4, 8, 9, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 0, 4, 0, 5], preds:[0, 9, 4, 9, 0, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 1, 4, 0, 5], preds:[0, 9, 4, 9, 1, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 2, 4, 0, 5], preds:[0, 9, 4, 9, 2, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 3, 4, 0, 5], preds:[0, 9, 4, 9, 3, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 4, 4, 0, 5], preds:[0, 9, 4, 9, 4, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 5, 4, 0, 5], preds:[0, 9, 4, 9, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 4, 9, 6, 4, 0, 5], preds:[0, 9, 4, 9, 6, 4, 0, 5], accuracy :1.0
target:[0, 9, 4, 9, 7, 4, 0, 5], preds:[0, 9, 4, 9, 7, 4, 0, 5], accuracy :1.0
target:[0, 9, 5, 0, 1, 4, 0, 5], preds:[0, 9, 3, 0, 1, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 2, 4, 0, 5], preds:[0, 9, 4, 0, 2, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 3, 4, 0, 5], preds:[0, 9, 8, 0, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 4, 4, 0, 5], preds:[0, 9, 8, 0, 4, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 5, 4, 0, 5], preds:[0, 9, 4, 0, 3, 4, 0, 5], accuracy :0.75
target:[0, 9, 5, 0, 6, 4, 0, 5], preds:[0, 9, 8, 0, 6, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 7, 4, 0, 5], preds:[0, 9, 3, 0, 7, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 8, 4, 0, 5], preds:[0, 9, 8, 0, 8, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 0, 9, 4, 0, 5], preds:[0, 9, 3, 0, 9, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 0, 4, 0, 5], preds:[0, 9, 3, 1, 6, 4, 0, 5], accuracy :0.75
target:[0, 9, 5, 1, 1, 4, 0, 5], preds:[0, 9, 3, 1, 1, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 2, 4, 0, 5], preds:[0, 9, 4, 1, 2, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 3, 4, 0, 5], preds:[0, 9, 8, 1, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 4, 4, 0, 5], preds:[0, 9, 4, 1, 4, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 5, 4, 0, 5], preds:[0, 9, 8, 1, 3, 4, 0, 5], accuracy :0.75
target:[0, 9, 5, 1, 6, 4, 0, 5], preds:[0, 9, 4, 1, 6, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 7, 4, 0, 5], preds:[0, 9, 8, 1, 7, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 8, 4, 0, 5], preds:[0, 9, 4, 1, 8, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 1, 9, 4, 0, 5], preds:[0, 9, 8, 1, 9, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 2, 7, 4, 0, 5], preds:[0, 9, 8, 2, 7, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 2, 8, 4, 0, 5], preds:[0, 9, 8, 2, 8, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 2, 9, 4, 0, 5], preds:[0, 9, 4, 2, 9, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 3, 0, 4, 0, 5], preds:[0, 9, 8, 3, 0, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 4, 0, 4, 0, 5], preds:[0, 9, 4, 4, 6, 4, 0, 5], accuracy :0.75
target:[0, 9, 5, 4, 1, 4, 0, 5], preds:[0, 9, 4, 4, 1, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 4, 2, 4, 0, 5], preds:[0, 9, 4, 4, 2, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 4, 3, 4, 0, 5], preds:[0, 9, 8, 4, 3, 4, 0, 5], accuracy :0.875
target:[0, 9, 5, 4, 4, 4, 0, 5], preds:[0, 9, 4, 4, 4, 4, 0, 5], accuracy :0.875

'''
    