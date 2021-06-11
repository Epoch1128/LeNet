import torch
import torch.nn as nn
from data import data_test_loader
from matplotlib import pyplot as plt
from model import LeNet
import torchvision.transforms as transforms


def test():
    model = LeNet()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    toPIL = transforms.ToPILImage()
    idx = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(data_test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            predict = predicted.eq(targets).sum().item()
            correct += predict
            if predict == 0 and idx < 10:
                pic = toPIL(inputs)
                pic.save('{}.jpg'.format(predicted))
                idx += 1

            print(batch_idx, len(data_test_loader),
                  'Loss: %3.f | Acc: %.3f%%(%d/%d)' % (
                      test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == '__main__':
    test()
