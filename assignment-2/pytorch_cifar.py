"""
Assignment 2, questions 11 and 12: the CIFAR-10 classifier from the PyTorch 60-minute blitz,
with options for the hyperparameters and for the changes in question 12.

Examples:
    python pytorch_cifar.py --lr 0.001 --batch-size 4 --epochs 2 --name blitz_default
    python pytorch_cifar.py --optimizer adam --lr 0.001 --batch-size 64 --epochs 10 --name adam

Training uses 45,000 of the 50,000 training images, the other 5,000 are a validation set.
With --final, the model is trained on all 50,000 and evaluated on the test set.
Each run saves its results to results/<name>.json (same format as experiments.py, so plot.py works).
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    """
    The network from the blitz, with optional batch normalization and a width multiplier.
    """
    def __init__(self, batchnorm=False, width=1):
        super().__init__()
        c1, c2 = 6 * width, 16 * width
        self.conv1 = nn.Conv2d(3, c1, 5)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.bn1 = nn.BatchNorm2d(c1) if batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(c2) if batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = net(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    net.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    p.add_argument('--momentum', type=float, default=0.9, help='only used by sgd')
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--batchnorm', action='store_true')
    p.add_argument('--width', type=int, default=1, help='multiplier for the number of conv channels')
    p.add_argument('--augment', action='store_true', help='random crops and horizontal flips')
    p.add_argument('--final', action='store_true', help='train on all training data, evaluate on the test set')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    train_transform = plain
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize])

    train_full = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    if args.final:
        trainset = train_full
        evalset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=plain)
    else:
        eval_full = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=plain)
        perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42))
        trainset = torch.utils.data.Subset(train_full, perm[5000:].tolist())
        evalset = torch.utils.data.Subset(eval_full, perm[:5000].tolist())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1000, shuffle=False)

    net = Net(batchnorm=args.batchnorm, width=args.width)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    history = {'epoch': [], 'val_acc': [], 'train_loss': []}
    start = time.time()

    for epoch in range(args.epochs):
        running_loss, seen = 0.0, 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        acc = evaluate(net, evalloader)
        history['epoch'].append(epoch + 1)
        history['val_acc'].append(acc)
        history['train_loss'].append(running_loss / seen)
        print(f'epoch {epoch + 1:3d}   train loss {running_loss / seen:.4f}   '
              f'{"test" if args.final else "val"} acc {acc:.4f}   ({time.time() - start:.0f} s)')

    seconds = time.time() - start
    print(f'final {"test" if args.final else "validation"} accuracy: {history["val_acc"][-1]:.4f}   ({seconds:.0f} s)')

    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', f'{args.name}.json'), 'w') as f:
        json.dump({'args': vars(args), 'seconds': seconds, **history}, f, indent=1)


if __name__ == '__main__':
    main()
