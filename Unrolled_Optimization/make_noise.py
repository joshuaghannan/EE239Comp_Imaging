""" This script generates noisy images from the BSDS500 data set. """

import os

import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from tqdm import tqdm
import scipy.misc

IMAGES_DIR = './BSR/BSDS500/data/images/'
TEST_DIR = IMAGES_DIR + 'test/'
TRAIN_DIR = IMAGES_DIR + 'train/'
VAL_DIR = IMAGES_DIR + 'val/'

NOISY_IMAGES_DIR = './images_noisy/'
if not os.path.exists(NOISY_IMAGES_DIR):
    os.mkdir(NOISY_IMAGES_DIR)
NOISY_TEST_DIR = NOISY_IMAGES_DIR + 'test/'
if not os.path.exists(NOISY_TEST_DIR):
    os.mkdir(NOISY_TEST_DIR)
NOISY_TRAIN_DIR = NOISY_IMAGES_DIR + 'train/'
if not os.path.exists(NOISY_TRAIN_DIR):
    os.mkdir(NOISY_TRAIN_DIR)
NOISY_VAL_DIR = NOISY_IMAGES_DIR + 'val/'
if not os.path.exists(NOISY_VAL_DIR):
    os.mkdir(NOISY_VAL_DIR)

NOISY_ARRAYS_DIR = './arrays_noisy/'
if not os.path.exists(NOISY_ARRAYS_DIR):
    os.mkdir(NOISY_ARRAYS_DIR)
NOISY_ARR_TEST_DIR = NOISY_ARRAYS_DIR + 'test/'
if not os.path.exists(NOISY_ARR_TEST_DIR):
    os.mkdir(NOISY_ARR_TEST_DIR)
NOISY_ARR_TRAIN_DIR = NOISY_ARRAYS_DIR + 'train/'
if not os.path.exists(NOISY_ARR_TRAIN_DIR):
    os.mkdir(NOISY_ARR_TRAIN_DIR)
NOISY_ARR_VAL_DIR = NOISY_ARRAYS_DIR + 'val/'
if not os.path.exists(NOISY_ARR_VAL_DIR):
    os.mkdir(NOISY_ARR_VAL_DIR)


def make_noisy_img(image):
    noise = np.random.normal(0, 25, image.shape)
    return np.array(np.minimum(np.maximum(image + noise, 0), 255), dtype=np.uint8)


for fn in tqdm(os.listdir(TEST_DIR)):
    if fn[-3:] != 'jpg':
        continue
    curr_img = np.array(Image.open(TEST_DIR + fn))
    image_noisy = make_noisy_img(curr_img)

    Image.fromarray(image_noisy).save(NOISY_TEST_DIR + fn)
    np.save(NOISY_ARR_TEST_DIR + fn[:-4], image_noisy)

for fn in tqdm(os.listdir(TRAIN_DIR)):
    if fn[-3:] != 'jpg':
        continue
    curr_img = np.array(Image.open(TRAIN_DIR + fn))
    image_noisy = make_noisy_img(curr_img)

    Image.fromarray(image_noisy).save(NOISY_TRAIN_DIR + fn)
    np.save(NOISY_ARR_TRAIN_DIR + fn[:-4], image_noisy)

for fn in tqdm(os.listdir(VAL_DIR)):
    if fn[-3:] != 'jpg':
        continue
    curr_img = np.array(Image.open(VAL_DIR + fn))
    image_noisy = make_noisy_img(curr_img)

    Image.fromarray(image_noisy).save(NOISY_VAL_DIR + fn)
    np.save(NOISY_ARR_VAL_DIR + fn[:-4], image_noisy)
