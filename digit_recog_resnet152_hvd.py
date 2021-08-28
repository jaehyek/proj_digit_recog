import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import gen_digit
from PIL import Image
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_dir', default=os.path.expanduser('./digit_class_aug_train'), help='path to training data')
parser.add_argument('--val_dir', default=os.path.expanduser('./digit_class_aug_val'), help='path to validation data')
parser.add_argument('--log_dir', default='./logs', help='tensorboard log directory')
parser.add_argument('--checkpoint_format', default='./resnet152-digitgen-{epoch}.pt', help='checkpoint file format')
parser.add_argument('--fp16_allreduce', action='store_true', default=False, help='use fp16 compression during allreduce')
parser.add_argument('--batches_per_allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use_adasum', action='store_true', default=False, help='use adasum algorithm to do reduction')
parser.add_argument('--gradient_predivide_factor', type=float, default=1.0, help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32, help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--base_lr', type=float, default=0.0125, help='learning rate for a single GPU')
parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')
parser.add_argument('--disable_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, help='random seed')


def train(epoch, verbose,log_writer):
    model.train()
    train_sampler.set_epoch(epoch)
    metric_train_loss = Metric('metric_train_loss')
    metric_train_accuracy = Metric('metric_train_accuracy')


    with tqdm(total=len(train_loader), desc='Train Epoch     #{}'.format(epoch + 1), disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                metric_train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                metric_train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': metric_train_loss.avg.item(), 'accuracy': 100. * metric_train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', metric_train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', metric_train_accuracy.avg, epoch)


def validate(epoch, verbose,log_writer):
    model.eval()
    metric_val_loss = Metric('metric_val_loss')
    metric_val_accuracy = Metric('metric_val_accuracy')

    with tqdm(total=len(val_loader), desc='Validate Epoch  #{}'.format(epoch + 1), disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                metric_val_loss.update(F.cross_entropy(output, target))
                metric_val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': metric_val_loss.avg.item(), 'accuracy': 100. * metric_val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', metric_val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', metric_val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

class Dataset_GenDigit(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.args = gen_digit.gen_digit_ready()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.args.len_total

    def __getitem__(self, idx):
        image, label, np_keypoint = gen_digit.generate_digits_by_index(self.args, idx)
        image_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv2)

        if self.transform is not None:
            image_pil = self.transform(image_pil)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image_pil, label

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0, name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
        mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # train_dataset = datasets.ImageFolder(args.train_dir,transform=transform)
    train_dataset = Dataset_GenDigit(transform=transform)

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs)

    val_dataset = train_dataset
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs)

    # Set up standard ResNet-50 model.
    model = models.resnet152()
    model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=(args.base_lr * lr_scaler), momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch, verbose,log_writer)
        save_checkpoint(epoch)
        # validate(epoch, verbose,log_writer)
