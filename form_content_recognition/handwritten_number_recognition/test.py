import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils import data
import os
from PIL import Image
import numpy as np
import torchvision
from scipy.stats import mode
import torch.nn.functional as F
from lenet import LeNet5

#定义batch size

#数据集
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
                                  shuffle=True)
    return data_loader



if __name__ == '__main__':

    test_loader = get_loader(root='./data/test_img/',batch_size = 128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net1 = LeNet5()
    net1 = torch.load('./models/lenet5.pkl')
    net1.to(device)
    net1.eval()
    d = 0
    total_correct = 0
    length = 0
    for i,(images,labels) in enumerate(test_loader):
        images = images.to(device)
        output1 = net1(images)
        pred1 = output1.cpu().detach().max(1)[1]
        total_correct += pred1.eq(labels.view_as(pred1)).sum()
        length += images.size(0)
        for j in range(np.shape(images)[0]):
            if not os.path.exists('./save_num/' + str(pred[j])):
                os.makedirs('./save_num/' + str(pred[j]))
            torchvision.utils.save_image(torch.from_numpy(np.array(images[j,:,:,:])),
                                         os.path.join('./save_num/', str(pred[j]),'%d_%s_%s.png' % (d,np.array(labels[j]), str(pred1[j]))))
            d += 1
    print(np.array(total_correct/length))



