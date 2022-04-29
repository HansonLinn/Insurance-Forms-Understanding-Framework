from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils import data
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

class ImageFolder(data.Dataset):
    def __init__(self, root):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        print("image count in {}".format(len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        image_path = self.image_paths[index]
        cls = image_path.split('/')[-1].split('_')[0]
        image = Image.open(image_path)
        image = image.convert('1')

        Transform = []
        # Transform.append(T.Resize((32, 32), interpolation=InterpolationMode.NEAREST))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)

        image = image.to(torch.float32)
        return image, int(cls)

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

#将下载的MNIST数据导入到dataloader中
def get_loader(root,batch_size):
    dataset = ImageFolder(root=root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader


def train(epoch):

    train_loader = get_loader('./data/train_img/',batch_size=512)
    valid_loader = get_loader('./data/valid_img/',batch_size=512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet5()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    net.train()
    loss_list, batch_list = [], []
    flag = 0
    for i in range(epoch):
        total_correct = 0
        length = 0
        for j, (images, labels) in enumerate(train_loader):
            # print(torch.max(images))
            images = images.to(device)
            labels = labels.to(device)
            length += images.size(0)
            output = net(images)
            # output = F.softmax(output)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            loss = criterion(output, labels)
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(j+1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train - Epoch %d, Loss: %f,Accuracy: %f' % (i, sum(loss_list)/len(loss_list),float(total_correct) / length))

        # net = torch.load('lenet5.pkl')
        net.eval()
        total_correct = 0

        length = 0
        for k, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            # output = F.softmax(output)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            length += images.size(0)

        print('Test Avg. Accuracy: %f' % (float(total_correct) / length))

        if float((total_correct) / length) >= flag:
            torch.save(net,'./models/lenet5.pkl')
            flag = float((total_correct) / length)


if __name__ == '__main__':
    train(1000)
