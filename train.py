import torch
import torch.nn as nn
from torch import optim
from model import LeNet
from data import data_train_loader
from matplotlib import pyplot as plt
import torchvision.transforms as transforms


def train():
    model = LeNet()
    model.train()
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_acc = []
    loss_data = []

    train_loss = 0
    correct = 0
    total = 0
    epoch = 16
    sample = []
    for time in range(epoch):
        for batch_idx, (inputs, targets) in enumerate(data_train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(data_train_loader),
                  'Loss: %3.f | Acc: %.3f%%(%d/%d)' % (
                      train_loss, 100. * correct / total, correct, total))
            train_acc.append(correct / total)
            loss_data.append(train_loss)
    for i in range(len(train_acc)):
        sample.append(i)

    plt.figure()
    plt.plot(sample, train_acc, 'blue', label='Training accuracy')
    plt.xlabel('Epoch')
    plt.title('Training Process')
    plt.show()

    plt.figure()
    plt.plot(sample, loss_data, 'red', label='Training Loss')
    plt.xlabel('Epoch')
    plt.title('Training Process')
    plt.show()

    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()
