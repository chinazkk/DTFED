import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import os
import sys
import cv2
import bdcn
from datasets.dataset import Data
import argparse
import cfg
from matplotlib import pyplot as plt

params_dir = {
    'DTFED': 'params/bdcn_120000.pth',
    'NYUD': 'params/bdcn_pretrained_on_nyudv2_rgb.pth',
    'BSDS500': 'params/bdcn_pretrained_on_bsds500.pth'
}

IS_CUDA = torch.cuda.is_available()


def sigmoid(x):
    return 1. / (1 + np.exp(np.array(-1. * x)))


def test(model, args):
    root = '../dataset/test'
    test_img = Data(root)
    test_loader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False)
    save_dir = args.save_dir
    train_dataset = args.train_dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda and IS_CUDA:
        model.cuda()
    model.eval()
    start_time = time.time()
    all_t = 0

    for i, data in enumerate(test_loader):
        if args.cuda and IS_CUDA:
            data = data.cuda()
        x_path, x = data
        tm = time.time()
        out = model(x.float())
        fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        if not os.path.exists(os.path.join(save_dir, 'result_%s' % train_dataset)):
            os.mkdir(os.path.join(save_dir, 'result_%s' % train_dataset))
        cv2.imwrite(os.path.join(save_dir, 'result_%s' % train_dataset, '%s' % x_path), fuse * 255)
        print(x_path[0])
        all_t += time.time() - tm
    # print(all_t)
    print('Overall Time use: ', time.time() - start_time)


def main():
    import time
    print(time.localtime())
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('%s' % (params_dir[args.train_dataset]), map_location=torch.device('cpu')))
    test(model, args)


def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str,
                        default='...', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('-t', '--train_dataset', type=str, default='DTFED',
                        help='the model train on this dataset')
    parser.add_argument('-m', '--model', type=str, default='params/bdcn_pretrained_on_bsds500.pth',
                        help='the model to test')
    parser.add_argument('-s', '--save_dir', type=str, default='result',
                        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
                        help='the k-th split set of multicue')
    return parser.parse_args()


if __name__ == '__main__':
    main()
