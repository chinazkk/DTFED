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

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def test(model, args):
    root = '../dataset/test'


    test_img = Data(root)
    testloader = torch.utils.data.DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=8)

    save_res = True
    save_dir = args.res_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.cuda:
        model.cuda()
    model.eval()
    data_iter = iter(testloader)
    iter_per_epoch = len(testloader)
    start_time = time.time()
    all_t = 0
    for i, data in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
        x_path,x= data

        tm = time.time()
        out = model(x.double())
        fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
        if not os.path.exists(os.path.join(save_dir, 'result_TED')):
            os.mkdir(os.path.join(save_dir, 'result_TED'))
        cv2.imwrite(os.path.join(save_dir, 'result_TED', '%s'%x_path), fuse*255)
        all_t += time.time() - tm
        print(x_path)
    print(all_t)
    print('Overall Time use: ', time.time() - start_time)

def main():
    import time
    print(time.localtime())
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = bdcn.BDCN()
    model.load_state_dict(torch.load('%s' % (args.model)))
    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str,
                        default='...', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default='params/bdcn_120000.pth',
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='result',
        help='the dir to store result')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    return parser.parse_args()

if __name__ == '__main__':
    main()
