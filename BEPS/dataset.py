# -*- coding:utf-8 -*-
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import cv2
import math
import torch.nn.functional as F
import random
from torchvision import transforms
from PIL import ImageFile
import torch




infinite = 1e-10
INF = 1e-3
IMG_H = 224





def make_dataset(root):
    imgs = []
    count = 0
    for i in os.listdir(root):
        img = os.path.join(root, i)
        (filename, extension) = os.path.splitext(i)
        if os.path.exists(root.replace("eti", "emi")):
            mask = os.path.join(root.replace("eti", "emi"), filename + '.png')
        if os.path.exists(root.replace("eti", "em")):
            edge = os.path.join(root.replace("eti", "em"), filename + '.png')
        imgs.append((img, mask, edge))
    return imgs

class Dataset(data.Dataset):
    def __init__(self, root,  test=False, mean=False, stage=1):
        imgs = make_dataset(root)
        imgs_num = len(imgs)
        imgs = np.random.permutation(imgs)
        self.stage = stage
        if not test:
            self.imgs = imgs[int(0.2 * imgs_num):]
        elif test:
            self.imgs = imgs
        self.test = test
        self.mean = mean
    def __getitem__(self, index):  # x y m 分别为原图，尺度，pure
        if len(self.imgs[0]) == 3:
            x_path,  m_path, edge = self.imgs[index]

        img_x = cv2.imread(x_path)
        img_x = img_x.transpose(2, 0, 1)

        if not self.test:
            img_z = cv2.imread(m_path)
            edge = cv2.imread(edge)
            img_z = img_z.transpose(2, 0, 1)
        else:
            img_z = None
        if self.mean:
            img_z = cv2.imread(m_path)
            img_z = img_z.transpose(2, 0, 1)
        if not self.test:

            size = 224
            _, x, y = img_x.shape
            x_rand = random.randint(0, x - size)
            y_rand = random.randint(0, y - size)
            img_x = img_x[:, x_rand:x_rand + size, y_rand:y_rand + size]
            edge = edge[x_rand:x_rand + size, y_rand:y_rand + size]
            if self.mean:
                img_z = img_z[:, x_rand:x_rand + size, y_rand:y_rand + size]

                return img_x / 255., img_z / 255,edge/255
                # return img_x / 255., img_y, img_z, band, structure / 255
            # img_y = self.train_transform(img_y)


        else:
            _, height, width = img_x.shape
            print(img_x.shape)
            # height -= height % 16
            # width -= width % 16
            # img_x = img_x[:, :height, :width]
            # img_x = get_bigger(img_x)
        return img_x / 255.,x_path

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dist_path = r"../datasets/unsmooth_fill_8_24/result8-16/ILSVRC2012_test_00000649_1598258577.npy"
    img_path = r"../datasets/unsmooth_fill_8_24/pure/ILSVRC2012_test_00000649_1598258577.png"
    img = cv2.imread(img_path)
    dist = np.load(dist_path)
    dist = check_edge(img, dist)
    for i in range(6):
        cv2.imshow("%d" % i, dist[:, :, i] * 255 / 21)
    cv2.waitKey()
