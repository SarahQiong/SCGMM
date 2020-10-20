import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--ckpf',
                    default='./nist_convnet_model_epoch_10.pth',
                    help="path to model checkpoint file (to continue training)")
parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=1024,
                    metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval',
                    type=int,
                    default=10,
                    metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train',
                    action='store_true',
                    help='training a ConvNet model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true', help='evaluate a [pre]trained model')

args = parser.parse_args()
# use CUDA?
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# From MNIST to Tensor
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Load MNIST only if training
if args.train:
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='by_class/NIST_train',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(64),
                transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.8693, ), (0.3081, ))
                # transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='by_class/NIST_test',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(64),
                transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.8693, ), (0.3081, ))
                # transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)


class backbone(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(backbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1)

model = backbone().to(device)
classifier = Classifier().to(device)



# Load checkpoint
if args.ckpf != '':
    if use_cuda:
        model.load_state_dict(torch.load(args.ckpf))
    else:
        # Load GPU model on CPU
        model.load_state_dict(torch.load(args.ckpf, map_location=lambda storage, loc: storage))

optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, momentum=args.momentum)


def train(args, model, device, train_loader, optimizer, epoch):
    """Training"""
    model.train()
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature = model(data)
        output = classifier(feature)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('{{"metric": "Train - CE Loss", "value": {}}}'.format(loss.item()))


def test(args, model, device, test_loader, epoch):
    """Testing"""
    model.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            output = classifier(feature)
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    print('{{"metric": "Eval - CE Loss", "value": {}, "epoch": {}}}'.format(test_loss, epoch))
    print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
        100. * correct / len(test_loader.dataset), epoch))


def test_image(train):
    if train:
        data_set = datasets.ImageFolder(
            root='by_class/NIST_train',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(64),
                transforms.Resize(28),
                transforms.ToTensor()
            ]))
    else:
        data_set = datasets.ImageFolder(
            root='by_class/NIST_test',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(64),
                transforms.Resize(28),
                transforms.ToTensor()
            ]))

    eval_loader = torch.utils.data.DataLoader(data_set, batch_size=1024)

    # Name generator
    accuracy = 0
    total = 0
    model.eval()
    classifier.eval()
    feature = []
    true_label = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            feature.append(output.cpu().numpy())
            output = classifier(output) 
            label = output.argmax(dim=1, keepdim=True)
            true_label.append(target.cpu().numpy())
            print(data.shape, target.shape[0], torch.sum(torch.squeeze(label) == target))
            accuracy += torch.sum(torch.squeeze(label) == target)
            total += target.shape[0]
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(accuracy, total, accuracy.div(total)))
    feature = np.concatenate(feature)
    true_label = np.concatenate(true_label)

    if train:
        np.save('NIST_feature_train_50d.npy', feature)
        np.save('NIST_label_train.npy', true_label)
    else:
        np.save('NIST_feature_test_50d.npy', feature)
        np.save('NIST_label_test.npy', true_label)




# Train?
if args.train:
    # Train + Test per epoch
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    # Do checkpointing 
    torch.save(model.state_dict(), 'nist_convnet_model_epoch_%d.pth' % (args.epochs))

# Evaluate?
if args.evaluate:
    test_image(True)
    test_image(False)
    

