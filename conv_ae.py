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
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
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

    model = conv_autoencoder()
    if is_cuda == True:
        model.cuda()
        image_ref, index_ref = image_ref.cuda(), index_ref.cuda()
        image_ref, index_ref = Variable(image_ref), Variable(index_ref)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    
    if model_path_load != None and os.path.isfile(model_path_load):
        if is_cuda == True:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
    
        print(f'load from {model_path_load}')
        checkpoint = torch.load(model_path_load, map_location=dev)
        if checkpoint.get('model_state_dict', None) != None :
            model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('optimizer_state_dict', None) != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if checkpoint.get('scheduler_state_dict', None) != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    criterion = nn.MSELoss()

    
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
        if epoch % 10 == 0:
            print(index)
            pic = to_image(output.cpu().data)
            save_image(pic, './dc_img/img_{:04d}_in.png'.format(epoch))

            pic = to_image(img.cpu().data)
            save_image(pic, './dc_img/img_{:04d}_out.png'.format(epoch))
            
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_path_save)

def make_autoencoder_digit_from_class( dir_digit_class, dir_autoencoder, model_path_load=r'./conv_ae.pt' ):
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

    model = conv_autoencoder()
    if is_cuda == True:
        model.cuda()

    if model_path_load != None and os.path.isfile(model_path_load):
        if is_cuda == True:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
    
        print(f'load from {model_path_load}')
        checkpoint = torch.load(model_path_load, map_location=dev)
        if checkpoint.get('model_state_dict', None) != None:
            model.load_state_dict(checkpoint['model_state_dict'])

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
    for i, index in enumerate(list_index) :
        dir_save = list_dir_digit[index]
        pic = to_image(output.cpu().data)
        save_image(pic[i], os.path.join(dir_save, str(f'{loop}_{i}.jpg')))



if __name__ == '__main__':
    time_start = time.time()
    # conv_autoencoder_model(r'.\digit_class_aug', r'.\digit_class_ref',model_path_load=r'./conv_ae.pt', model_path_save=r'./conv_ae.pt')
    
    make_autoencoder_digit_from_class( r'.\digit_class_aug', r'.\digit_class_autoencoder', model_path_load=r'./conv_ae.pt' )
    print(f'elapsed time sec  : {time.time() - time_start}')