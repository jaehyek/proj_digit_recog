import argparse
import os
import time


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import svhn_dataset as dataset
import svhn_model as model
from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--channel', type=int, default=32, help='first conv channel (default: 32)')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--data_root', default=r'D:\proj_gauge\proj_digit_recog\data\svhn', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)

print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)
model = model.svhn(n_channel=args.channel)
model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    model.cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
try:
    for epoch in range(args.epochs):
        model.train()
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = correct * 1.0 / len(data)
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data[0], acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda().long().squeeze()
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                test_loss += F.cross_entropy(output, target).data[0]
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader) # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset), acc))
            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))

                best_acc = acc
                old_file = new_file
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))