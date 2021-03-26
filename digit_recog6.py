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
    plt.close()
   


train_transform = transforms.Compose([transforms.Resize((224, 224))
                                         , transforms.RandomHorizontalFlip()
                                         , transforms.RandomRotation(0.2)
                                         , transforms.ToTensor()
                                         , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

train = ImageFolder('data/dogsandcats/train/', train_transform)


train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)
img, label = next(iter(train_data_loader))

# imshow(img[5])

img = img[5][None]
vgg = models.vgg16(pretrained=True).cuda()



'''

for i, layer in enumerate( vgg.features.children()):
    print(f'{i}, layer_name:{layer}')

0 , layer_name:Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
1 , layer_name:ReLU(inplace=True)
2 , layer_name:Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
3 , layer_name:ReLU(inplace=True)
4 , layer_name:MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
5 , layer_name:Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
6 , layer_name:ReLU(inplace=True)
7 , layer_name:Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
8 , layer_name:ReLU(inplace=True)
9 , layer_name:MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
10, layer_name:Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
11, layer_name:ReLU(inplace=True)
12, layer_name:Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
13, layer_name:ReLU(inplace=True)
14, layer_name:Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
15, layer_name:ReLU(inplace=True)
16, layer_name:MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
17, layer_name:Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
18, layer_name:ReLU(inplace=True)
19, layer_name:Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
20, layer_name:ReLU(inplace=True)
21, layer_name:Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
22, layer_name:ReLU(inplace=True)
23, layer_name:MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
24, layer_name:Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
25, layer_name:ReLU(inplace=True)
26, layer_name:Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
27, layer_name:ReLU(inplace=True)
28, layer_name:Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
29, layer_name:ReLU(inplace=True)
30, layer_name:MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
'''


    
'''

for i, layer in enumerate( vgg.classifier.children()):
    print(f'{i}, layer_name:{layer}')

0, layer_name:Linear(in_features=25088, out_features=4096, bias=True)
1, layer_name:ReLU(inplace=True)
2, layer_name:Dropout(p=0.5, inplace=False)
3, layer_name:Linear(in_features=4096, out_features=4096, bias=True)
4, layer_name:ReLU(inplace=True)
5, layer_name:Dropout(p=0.5, inplace=False)
6, layer_name:Linear(in_features=4096, out_features=1000, bias=True)
'''


for layer in vgg.classifier.children():
    if (type(layer) == nn.Dropout):
        layer.p = 0.2
        
        
class LayerActivations():
    features = None
    
    def __init__(self, model, layer_num):
        print(model[layer_num])
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        print(f'output shape:{output.shape}')
        self.features = output.cpu().data.numpy()
    
    def remove(self):
        self.hook.remove()

##########################################################
# conv_out = LayerActivations(vgg.features, 0)
#
# o = vgg(Variable(img.cuda()))
#
# conv_out.remove()
#
# act = conv_out.features
#
# fig = plt.figure(figsize=(10, 8))
# fig.subplots_adjust(left=0, right=0.01, bottom=0, top=0.01)
# # fig.subplots_adjust(wspace=0.001, hspace=0.001)
# for i in range(30):
#     ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(act[0][i])
#
# plt.tight_layout()
# plt.savefig('fig1.png', dpi=300)
# plt.show()
#
# exit(0)

##########################################################

conv_out = LayerActivations(vgg.features, 20)
o = vgg(Variable(img.cuda()))
conv_out.remove()

act = conv_out.features

fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0, right=0.01, bottom=0, top=0.01)
for i in range(30):
    ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(act[0][i])
plt.tight_layout()
plt.show()
exit(0)

##########################################################
# conv_out = LayerActivations(vgg.features, 1)
# o = vgg(Variable(img.cuda()))
# conv_out.remove()
#
# act = conv_out.features
#
# fig = plt.figure(figsize=(20, 50))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(act[0][i])

##########################################################

vgg = models.vgg16(pretrained=True).cuda()
vgg.state_dict().keys()
cnn_weights = vgg.state_dict()['features.0.weight'].cpu()

fig = plt.figure(figsize=(30, 30))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 6, i + 1, xticks=[], yticks=[])
    imshow(cnn_weights[i])

fig = plt.figure(figsize=(30, 30))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 6, i + 1, xticks=[], yticks=[])
    imshow(cnn_weights[i])



