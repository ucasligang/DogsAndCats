import os
import torch
from torch.utils import data
import numpy as np
import torchvision
from torch.nn import functional
from torch.autograd import Variable
from data.dataset import DogCat
from models.ResNet34 import ResNet34
from matplotlib import pyplot as plt
import time
import shutil

n_epochs = 100
path_train = "data/DogsAndCats/train"
path_test = "data/DogsAndCats/test"
path_left = "data/DogsAndCats/left"

def split_dataset():
    source = path_train
    dest = path_test
    left = path_left
    if os.path.exists(dest) == False:
        os.mkdir(dest)
    filenames = os.listdir(source)
    for filename in filenames:
        num_data = filename.split(".")[1]  # 取出数据编号 将dog和cat的后2500个数据放入测试数据集中
        if(int(num_data) >= 1500):
            shutil.move(source+"/"+filename, left+"/"+filename)
        elif(int(num_data) >= 1000):
            shutil.move(source+"/"+filename, dest+"/"+filename)
    print("test数据集已经成功划分！")

def train():
    # step1:加载数据
    train_data_root = path_train
    test_data_root = path_test
    train_data = DogCat(train_data_root)
    test_data = DogCat(test_data_root)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=True,
    )

    """
    # 显示一个batch的图片
    images, labels = next(iter(train_data_loader))
    # print(images)
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = (img*std+mean)*255
    img = img.astype(int)
    print([labels[i] for i in range(64)])
    plt.imshow(img)
    plt.show()
    """

    #step2:加载模型
    model = ResNet34()

    # step3:目标函数和优化器
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.005,
                                 weight_decay=1e-4)

    # step4:训练
    start = time.time()
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*20)
        for data in train_data_loader:
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            outputs = model(x_train)
            _, y_pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(y_pred == y_train.data)
        testing_correct = 0
        for data in test_data_loader:
            x_test, y_test = data
            x_test, y_test = Variable(x_test), Variable(y_test)
            outputs = model(x_test)
            _, y_pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(y_pred == y_test.data)
        print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%,Time:{}"
              .format(running_loss*1.0/len(train_data), 100*running_correct*1.0/len(train_data),
                      100*testing_correct*1.0/len(test_data), time.time()-start))
    name = 'models/resnet34.pth'
    torch.save(model, name)


if __name__ == '__main__':
    #split_dataset()
    train()