from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
# from skimage import  color
# import matplotlib.pyplot  as plt
# matplotlib inline
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.misc
import scipy.io as sio

from module import model_vdsr

tf.reset_default_graph()
test_input = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
# test_input_scale=test_input/127.5-1 ########[-1,1]
test_input_scale = test_input  ######[0,255]
test_output, temp_weights = model_vdsr(test_input_scale, reuse=False)
saver = tf.train.Saver()
sess = tf.Session()
# saver.restore(sess, 'checkpoint_l1Loss/model-52002')

# saver.restore(sess, 'checkpoint_l1Loss/model-52002') #l1 loss
saver.restore(sess, 'checkpoint_ReguTerm/model-140002')  # mae:
# saver.restore(sess, 'checkpoint_ReguTerm/model-286002')
# saver.restore(sess, 'checkpoint/model-104002')  #l2 loss

# MAE



test_total_ae = 0
test_total_se = 0
test_total_count = 0
start = time.time()
image_dir = r'../../dataset/test/'
l = os.listdir(image_dir)
import random

random.shuffle(l)
for i in l:
    im = scipy.misc.imread(os.path.join(image_dir, i))
    im = np.float32(im)
    batch_images = [im]
    test_output_eval = sess.run(test_output, feed_dict={test_input: batch_images})
    im_out = test_output_eval[0]
    h, w, channel = im_out.shape
    # im_out=(np.float64(im_out)+1)/2*255
    im_out[im_out > 255] = 255
    im_out[im_out < 0] = 0
    scipy.misc.imsave('result/%s' % i, im_out / 255)

