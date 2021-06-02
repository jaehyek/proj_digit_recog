import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F
import warnings
import time
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")

class ReduceChannel(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        a = sample[0]
        b = a.reshape((1,) + a.shape)
        return b
    

def to_image(x):
    x = 0.5 * x + 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 64, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 32, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform_image = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        ReduceChannel(),
        transforms.Normalize((0.5), (0.5))
    ])

def make_ref_image(image_ref, index) :
    list_image = [ image_ref[i] for i in index ]
    image_ref_new = torch.stack(list_image, axis = 0)
    return image_ref_new

def conv_autoencoder_model(dir_class_aug, dir_class_ref, model_path_load=None, model_path_save=None):
    
    number_epochs = 1000
    batch_size = 10
    learning_rate = 0.0001

    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')
    
    dataset = ImageFolder(dir_class_aug, transform_image)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_ref = ImageFolder(dir_class_ref, transform_image)
    data_loader_ref = DataLoader(dataset_ref, batch_size=batch_size, shuffle=False)
    
    image_ref, index_ref = next(iter(data_loader_ref))

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')


    if model_path_load == None :
        model = conv_autoencoder()
        if is_cuda == True:
            model.cuda()
            image_ref, index_ref = image_ref.cuda(), index_ref.cuda()
            image_ref, index_ref = Variable(image_ref), Variable(index_ref)
    else:
        model = torch.load(model_path_load, map_location=dev)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    #         # 성능이 향상이 없을 때 learning rate를 감소시킨다.  optimizer에 momentum을 설정해야 사용할 수 있다


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
            # 성능이 향상이 없을 때 learning rate를 감소시킨다.  optimizer에 momentum을 설정해야 사용할 수 있다
    
    criterion = nn.MSELoss()
    total_loss_prev = 10000
    for epoch in range(number_epochs):
        total_loss = 0.0
        for data in data_loader:
            img, index = data
            if is_cuda == True:
                img, index = img.cuda(), index.cuda()
            img, index = Variable(img), Variable(index)
            
            # Forward pass
            output = model(img)
            image_ref_new = make_ref_image(image_ref, index )
            loss = criterion(output, image_ref_new)
            total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(total_loss)
        # Print results
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, number_epochs, total_loss.data))
        if epoch > 20 and ( total_loss < total_loss_prev) and ( model_path_save != None) :

            pic = to_image(output.cpu().data)
            save_image(pic, './dc_img/img_{:04d}_out.png'.format(epoch))

            pic = to_image(img.cpu().data)
            save_image(pic, './dc_img/img_{:04d}_in.png'.format(epoch))

            torch.save(model, model_path_save)
            total_loss_prev = total_loss
            print('model saved')


def get_images_from_model_eval(imgs, model_path_load=r'./conv_ae_7seg.pt'):

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    model = torch.load(model_path_load, map_location=dev)
    model.eval()
    if is_cuda == True:
        imgs = imgs.cuda()
    imgs = Variable(imgs)


    with torch.no_grad():
        output = model(imgs)
    return output


def make_autoencoder_digit_from_class( dir_digit_class, dir_autoencoder, model_path_load=r'./conv_ae_7seg.pt' ):
    try:
        if not os.path.isdir(dir_autoencoder):
            os.mkdir(dir_autoencoder)
    except:
        pass
    
    # create dir for 0, 1, 2, ..., 9
    list_dir_digit = []
    for num in range(10):
        try:
            dir_digit = os.path.join(dir_autoencoder, f'{num}')
            list_dir_digit.append(dir_digit)
            os.mkdir(dir_digit)
        
        except:
            continue


    batch_size = 10

    dataset = ImageFolder(dir_digit_class, transform_image)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    model = torch.load(model_path_load, map_location=dev)

    for loop, data in enumerate(data_loader):
        img, index = data
        if is_cuda == True:
            img, index = img.cuda(), index.cuda()
        img, index = Variable(img), Variable(index)
    
        # Forward pass
        output = model(img)
        save_encoder_image(output, loop, index, list_dir_digit)
        
def save_encoder_image(output, loop, indices, list_dir_digit):
    list_index = indices.tolist()
    pic = to_image(output.cpu().data)
    for i, index in enumerate(list_index) :
        dir_save = list_dir_digit[index]
        save_image(pic[i], os.path.join(dir_save, str(f'{loop}_{i}.jpg')))



if __name__ == '__main__':
    time_start = time.time()
    # 4,5 digit인 7 segment 인 경우.
    # conv_autoencoder_model(r'.\digit_class_7seg_aug', r'.\digit_class_ref', model_path_load=r'./conv_ae7.pt', model_path_save=r'./conv_ae_7seg.pt')
    # make_autoencoder_digit_from_class( r'.\digit_class_7seg_aug', r'.\digit_class_7seg_aug_autoencoder', model_path_load=r'./conv_ae_7seg.pt' )


    # 8 digit인  normal segment 인 경우.
    # conv_autoencoder_model(r'.\digit_class_normal_aug', r'.\digit_class_ref', model_path_load=None, model_path_save=r'./conv_ae_normal.pt')
    make_autoencoder_digit_from_class( r'.\digit_class_normal_aug', r'.\digit_class_normal_aug_autoencoder', model_path_load=r'./conv_ae_normal.pt' )

    print(f'elapsed time sec  : {time.time() - time_start}')