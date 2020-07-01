import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from pylab import *


class DogCat(data.Dataset):
    def __init__(self, root, transform=None):
        self.count = 0
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # self.imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
