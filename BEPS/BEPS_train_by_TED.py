import argparse
import os
import cv2

import torch
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import numpy as np
import gc
from dataset import Dataset
import time
import torch.nn as nn


def get_today_date():
    return time.strftime("%Y%m%d")


Train = False
Test = True
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=r'../dataset/minitrain/eti/',  # 加入自己需要的训练路径
                    help=' path of dataset')
parser.add_argument('--train_url', type=str, default=r'../models',
                    help=' path of model')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()

# 超参数
BATCH_SIZE = 10
EPOCH = 100
best_score = 0


## 定义网络
class BEPS(nn.Module):
    def __init__(self):
        super(BEPS, self).__init__()
        self.block = nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=3, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(x + self.block(x))


train_transform = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.545, 0.506, 0.472], std=[0.169, 0.170, 0.172])
])

test_img = Dataset(root=r'../dataset/test', test=True)  # 加入自己需要的测试路径
test_loader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False)

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

all_step = 0
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

model = BEPS().to(device)
model = model.to(device)

start_lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# loss function
criterion = torch.nn.BCELoss()
# if test in cpu
model.load_state_dict(torch.load(r"model45_2.pth", map_location=torch.device('cpu')))
# else
# model.load_state_dict(torch.load(r"model45_2.pth"))

if Train:
    train_img = Dataset(root=args.data_url, mean=True)
    train_loader = torch.utils.data.DataLoader(train_img, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    length = len(train_loader)
    for epoch in range(EPOCH):
        scheduler.step()
        lr = scheduler.get_lr()

        print("Start train_bin the %d epoch!" % (epoch + 1))
        for i, data in enumerate(train_loader):
            x_train, z_train, edge_gt = data
            x_train = Variable(x_train.float())
            z_train = Variable(z_train.float())
            edge_gt = Variable(edge_gt)
            x_train = x_train.to(device)
            z_train = z_train.to(device)
            edge_gt = edge_gt.to(device)
            optimizer.zero_grad()  # 将梯度初始化为零  每个batch的梯度并不需要被累加
            output = model(x_train)
            l1_loss = torch.nn.L1Loss()(output, z_train)
            loss = l1_loss
            loss.backward()
            optimizer.step()
            all_step += 1
            print(str(i + 1) + "/" + str(length) + " Loss:" + str(loss.item()))
        if epoch % 15 == 0:
            torch.save(model.state_dict(), "model%d.pth" % (epoch))
if Test:
    model.eval()
    for i, data in enumerate(test_loader):
        gc.collect()
        with torch.no_grad():
            x_train, x_path = data
            x_train = x_train[0]
            x_path = x_path[0]
            x_train = x_train.unsqueeze(0)
            x_train = Variable(x_train.float())
            x_train = x_train.to(device)
            # test if the dir is existed
            dir = "BEPS__train_byTED"
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(os.path.join(dir, "filter")):
                os.makedirs(os.path.join(dir, "filter"))
            if not os.path.exists(os.path.join(dir, "origin")):
                os.makedirs(os.path.join(dir, "origin"))
            cv2.imwrite(
                os.path.join(os.path.join(dir, "origin"), os.path.basename(x_path).split('.')[0] + ".jpg"),
                cv2.imread(x_path))
            res = model(x_train)
            res = res.cpu().data.numpy()
            res = res[0].transpose(1, 2, 0) * 255
            res = res.astype(np.uint8)
            cv2.imwrite(os.path.join(os.path.join(dir, "filter"),
                                     os.path.basename(x_path).split('.')[0] + ".jpg"), res)
