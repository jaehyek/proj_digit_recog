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
from torch.utils.data import Dataset,DataLoader
import time
import sys
from makeImageFolder import get_Image_Value_List_from_json

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



    
train_transform = transforms.Compose([transforms.Resize((64,64))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

# class Net(nn.Module):
#     def __init__(self, dropout):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(56180, 500)
#         self.fc2 = nn.Linear(500,50)
#         self.fc3 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.conv2_drop(x)
#         x = F.max_pool2d(x, 2)
#         x = F.relu(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.dropout)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x,p=self.dropout)
#         x = self.fc3(x)
#         return x


class Net(nn.Module):
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(32768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 10)
        )

        self.f0 = nn.Dropout(p=self.dropout)
        self.f1 = nn.Linear(32768, 128)
        self.f2 = nn.BatchNorm1d(128)
        self.f3 = nn.ReLU(inplace=True)
        self.f4 = nn.Dropout(p=self.dropout)
        self.f5 = nn.Linear(128, 64)
        self.f6 = nn.BatchNorm1d(64)
        self.f7 = nn.ReLU(inplace=True)
        self.f8 = nn.Dropout(p=self.dropout)
        self.f9 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        # x = self.linear_block(x)
        x = self.f0(x)
        x = self.f1(x)
        b, c = x.size()
        if b != 1:
            x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        b, c = x.size()
        if b != 1:
            x = self.f6(x)

        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        return x


def fit_numpy(epoch, model, data_loader, phase='training', volatile=False):
    global is_cuda, optimizer0
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
        # data = data.view(data.size(0), -1)
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
        f'epoch:{epoch}, {phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}   {accuracy:{10}.{4}}')
    return loss, accuracy





def DigitRecogModel(dir_train, phase='training', model_path_load=None, model_path_save=r'./model_simple.pt'):
    global is_cuda, optimizer, train_transform
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True

    # setting parameter
    batch_size = 5
    epoch_loop = 150 if phase == 'training' else 1

    train = ImageFolder(dir_train, train_transform)
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    model = Net(dropout=0.5)
    if is_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)

    if model_path_load != None:
        if is_cuda == True:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

        print(f'load from {model_path_load}')
        checkpoint = torch.load(model_path_load, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # check point
    if phase == 'validation' and model_path_load == None:
        print('model_path must not be None.')
        raise Exception("model_path must not be None.")


    train_losses, train_accuracy = [], []

    for epoch in range(epoch_loop):
        epoch_loss, epoch_accuracy = fit_numpy(epoch, model, train_data_loader, phase='training')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

    if model_path_save != None:
        torch.save({
            'model_state_dict': model.state_dict(),
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

    model = Net(dropout=0.5)

    if is_cuda == True:
        model = model.cuda()

    if is_cuda == True:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    checkpoint = torch.load(model_path_load, map_location=dev)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    # vgg.train()
    volatile = True

    if is_cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile), Variable(target)

    output = model(data)
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
    # loss, acc = DigitRecogModel(r'.\digit_class_aug', phase='training', model_path_load=None,
    #                             model_path_save=r'./model_simple.pt')

    # 기존학습에 추가 학습시킬때.
    loss, acc = DigitRecogModel(r'.\digit_class_aug', phase='training', model_path_load=r'./model_simple.pt',
                                model_path_save=r'./model_simple.pt')
    print(f'Model  Traning  loss:{loss}, accuracy:{acc}')

    print(f'Model  Traning  loss:{loss}, accuracy:{acc}')

    # test할 json을 받아서  예측한 값을 확인.
    print('-------------------------------------------------------------')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10050-56359.json', model_path_load=r'./model_simple.pt')
    print('~~~~~~~~~~~~~~~')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56385.json', model_path_load=r'./model_simple.pt')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10060-56394.json', model_path_load=r'./model_simple.pt')
    print('~~~~~~~~~~~~~~~')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56400.json', model_path_load=r'./model_simple.pt')
    npred = getValueFromJson(r'D:\proj_gauge\민성기\digit_test\10061-56455.json', model_path_load=r'./model_simple.pt')

    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56265.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56266.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56267.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56268.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56340.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56341.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56342.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56343.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56344.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56345.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56346.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56347.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56348.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56349.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56350.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56351.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56352.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56353.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56354.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56355.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56356.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56357.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56358.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56370.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56371.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56372.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56373.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56374.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56375.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56376.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56377.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56378.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56379.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56380.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56381.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56383.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56384.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56386.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56387.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56388.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56389.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56390.json', model_path_load=r'./model_simple.pt')
    print('~~~~~~~~~~~~~~~')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56391.json', model_path_load=r'./model_simple.pt')
    print('~~~~~~~~~~~~~~~')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56392.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10060-56393.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56397.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56398.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56399.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56401.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56402.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56403.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56404.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56405.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56406.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56407.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56408.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56409.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56413.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56430.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56431.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56432.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56433.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56434.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56435.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56436.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56437.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56438.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56439.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56440.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56441.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56442.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56443.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56444.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56445.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56446.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56447.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56448.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56449.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56450.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56451.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56452.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56453.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56454.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56458.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56459.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56460.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56461.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56462.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56463.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56464.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56465.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56466.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56467.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56468.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56469.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56470.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56471.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56472.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56473.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56474.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56475.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56476.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56485.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56486.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56487.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56488.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56489.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56490.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56491.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56492.json', model_path_load=r'./model_simple.pt')
    getValueFromJson(r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56493.json', model_path_load=r'./model_simple.pt')

    print(f'elapsed time sec  : {time.time() - time_start}')