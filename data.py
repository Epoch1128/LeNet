from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST('./data',
                   download=True,
                   transform=transforms.Compose(
                       [
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()
                       ]
                   )
)
data_test = MNIST('./data',
                  train=False,
                  download=True,
                  transform=transforms.Compose(
                      [
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()
                      ]
                  )
)
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

"""
if __name__ == '__main__':
    toPIL = transforms.ToPILImage()
    for i in range(10):
        pic = toPIL(data_test[i][0])
        pic.save('unknown{}.jpg'.format(i))
"""

