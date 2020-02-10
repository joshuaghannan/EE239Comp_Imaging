import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
import PIL
from PIL import Image

from scipy.stats import norm
from scipy.signal import convolve2d
# import scipy
from scipy.ndimage.filters import gaussian_filter

IMAGES_DIR = './BSR/BSDS500/data/images/'
TEST_DIR = IMAGES_DIR + 'test/'
TRAIN_DIR = IMAGES_DIR + 'train/'
VAL_DIR = IMAGES_DIR + 'val/'


def create_gaussian_kernel(sigma, truncate=4):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    sigma2 = sigma * sigma
    kern1d = np.exp(-0.5 / sigma2 * x ** 2)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


test_img = np.array(Image.open(TEST_DIR + '2018.jpg'))
blurry_img = gaussian_filter(test_img, (1.6, 1.6, 0))
blurry_img = np.array(blurry_img, dtype=np.uint8)

kernel = np.array(create_gaussian_kernel(1.6))
# temp0 = convolve2d(kernel, test_img[:,:,0])
# temp1 = convolve2d(kernel, test_img[:,:,1])
# temp2 = convolve2d(kernel, test_img[:,:,2])
# test_blurry_img = np.stack((temp0, temp1, temp2), axis=-1)
# test_blurry_img = np.array(test_blurry_img, dtype=np.uint8)


### Assume the image is n x n, and the kernel is d x d.
n = 256
d = kernel.shape[0]
new_size = n + d - 1
extra = int((d - 1) / 2)

# Initialize the kernel matrix
kernel_mat = np.zeros((n ** 2, new_size ** 2))

# First row of first block
row1 = np.hstack((kernel[0], np.zeros(n - 1)))
# Build first row, without extra 0s
for i in range(1, d):
    temp = np.hstack((kernel[i], np.zeros(n - 1)))
    row1 = np.hstack((row1, temp))
block1 = row1.copy()
# Build first block row, without extra 0s
for i in range(1, n):
    temp = np.roll(row1, i)
    block1 = np.vstack((block1, temp))
# Attach extra 0s
z = np.zeros((block1.shape[0], new_size ** 2 - block1.shape[1]))
block1 = np.hstack((block1, z))
# Build the remaining blocks
for i in range(n):
    temp = np.roll(block1, i * new_size)
    kernel_mat[i * n:(i + 1) * n] = temp

test_img_n = test_img[:n, :n]
img_flat = np.reshape(test_img_n, (-1, 3))
blurry_vec = np.matmul(kernel_mat.T, img_flat)
blurry_mat = np.reshape(blurry_vec, (new_size, new_size, 3))
blurry_mat = np.array(blurry_mat, dtype=np.uint8)
blurry_mat = blurry_mat[extra:-extra, extra:-extra, :]

fig = plt.figure()
fig.add_subplot(121)
plt.imshow(test_img_n)
fig.add_subplot(122)
plt.imshow(blurry_mat)
plt.show()
