import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import shutil

import sys
sys.path.append("..")
import model.dit_resnet_cifar10 as ResNet
import argparse
import os
import shutil
import torch.nn.functional as F
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    return

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '{}_'.format(args.model)+filename)
    if is_best:
        shutil.copyfile('{}_'.format(args.model)+filename, '{}_best.pth'.format(args.model))

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--model', metavar='MODEL', default='dit_resnet20', type=str, choices=['dit_resnet20', 'dit_resnet32', 'dit_resnet56', 'dit_resnet110'], help='model to train')
parser.add_argument('--gpu-id', default=[0], nargs='+', type=int, help='available GPU IDs')
parser.add_argument('--random-seed', default=[0], nargs='+', type=int, help='Random seed')
parser.add_argument('--epochs', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('-w', '--workers', default=12, type=int, metavar='N', help='num_workers, at most 16, must be 0 on windows')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-t', '--train', dest='train', action='store_true', help='test model on test set')

args = parser.parse_known_args()[0]

setup_seed(args.random_seed[0])

# Data
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)
valid_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=10000)

# Model
if args.model == 'dit_resnet20':
    model = ResNet.dit_resnet20()
elif args.model == 'dit_resnet32':
    model = ResNet.dit_resnet32()
elif args.model == 'dit_resnet56':
    model = ResNet.dit_resnet56()
elif args.model == 'dit_resnet110':
    model = ResNet.dit_resnet110()

device = torch.device("cuda:{}".format(args.gpu_id[0]))
model = nn.DataParallel(model, device_ids=args.gpu_id)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD([
    {'params': [param for name, param in model.named_parameters() if "quadratic" in name or "ln_in" in name], 'weight_decay': args.weight_decay, 'lr': 10*args.lr},
    {'params': [param for name, param in model.named_parameters() if "quadratic" not in name and "ln_in" not in name], 'weight_decay': args.weight_decay, 'lr': args.lr},
], momentum=args.momentum)
train_accuracy = []
test_accuracy = []
best_prec = 0

if args.resume:
    if os.path.isfile(args.resume):
        print('=> loading checkpoint "{}"'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {} best_acc {})".format(args.resume, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)

save_checkpoint({
    'epoch': 0,
    'state_dict': model.state_dict(),
        'best_prec': 0.0,
        'optimizer': optimizer.state_dict()
    }, True)

for epoch in range(args.start_epoch, args.epochs):
    ## Use the following code to adjust learning rate when training dit-resnet-56 and dit-resnet-110
    # if epoch == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    # if epoch == 1:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 10
    if epoch == 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    elif epoch == 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    model.eval()
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.long().to(device)
            output = model(input)

            _, predicted = torch.max(output.data, 1)
            valid_total += output.shape[0]
            valid_correct += (predicted == target).sum().item()

        prec = valid_correct / valid_total
        test_accuracy.append(prec)
        print('Accuary on test images:{:.2f}%'.format(prec*100))
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        if is_best:

            print('Best')
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
                'best_prec': prec,
                'optimizer': optimizer.state_dict()
            }, is_best)

    # train for one epoch
    train_total = 0
    train_correct = 0
    train_loss = 0
    for i, (input, target) in enumerate(train_loader):
        model.train()

        input, target = input.to(device), target.long().to(device)
        output = model(input)
        loss = criterion(output, target)
        train_loss += loss.item()*input.size(0)

        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        prec = train_correct / train_total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prec = train_correct / train_total
    print('Epoch [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, args.epochs, train_loss/len(train_loader.dataset), prec*100))

print('Best accuracy: {:.2f}%'.format(best_prec*100))