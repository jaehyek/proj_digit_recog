import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/proj/proj_digit_recog/digit_class_All_aug_train'), help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/proj/proj_digit_recog/digit_class_All_aug_test'), help='path to validation data')
parser.add_argument('--log-dir', default='./logs', help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./model_resnet152_sum.pt', help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False, help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False, help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0, help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32, help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125, help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5, help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, help='random seed')


class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MyImageFolder, self).__init__(root,transform, target_transform )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, (target, path)


def validate():
    model.eval()
    metric_val_loss = Metric('metric_val_loss')
    metric_val_accuracy = Metric('metric_val_accuracy')

    with tqdm(total=len(val_loader), desc='Validate  ') as t:
        with torch.no_grad():
            for data, (target, path) in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                metric_val_loss.update(F.cross_entropy(output, target))
                metric_val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': metric_val_loss.avg.item(), 'accuracy': 100. * metric_val_accuracy.avg.item()})
                t.update(1)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()



# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device("cuda:0")
        torch.cuda.manual_seed(args.seed)

    val_dataset = MyImageFolder(args.val_dir,
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size)


    # Set up standard ResNet-50 model.
    model = models.resnet152()
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    if args.cuda:
        # Move model to GPU.
        model.cuda()

    filepath = args.checkpoint_format
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    validate()
