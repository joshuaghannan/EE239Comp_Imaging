import os
import cv2 as cv
import numpy as np
import random
from tqdm import tqdm

if __name__ == '__main__':

    IMAGES_DIR = './images/'
    TEST_DIR = IMAGES_DIR + 'test/'
    TRAIN_DIR = IMAGES_DIR + 'train/'
    VAL_DIR = IMAGES_DIR + 'val/'

    BLURRY_IMAGES_DIR = './images_motblur/'
    if not os.path.exists(BLURRY_IMAGES_DIR):
        os.mkdir(BLURRY_IMAGES_DIR)
    BLURRY_TEST_DIR = BLURRY_IMAGES_DIR + 'test/'
    if not os.path.exists(BLURRY_TEST_DIR):
        os.mkdir(BLURRY_TEST_DIR)
    BLURRY_TRAIN_DIR = BLURRY_IMAGES_DIR + 'train/'
    if not os.path.exists(BLURRY_TRAIN_DIR):
        os.mkdir(BLURRY_TRAIN_DIR)
    BLURRY_VAL_DIR = BLURRY_IMAGES_DIR + 'val/'
    if not os.path.exists(BLURRY_VAL_DIR):
        os.mkdir(BLURRY_VAL_DIR)

    BLURRY_ARRAYS_DIR = './arrays_motblur/'
    if not os.path.exists(BLURRY_ARRAYS_DIR):
        os.mkdir(BLURRY_ARRAYS_DIR)
    BLURRY_ARR_TEST_DIR = BLURRY_ARRAYS_DIR + 'test/'
    if not os.path.exists(BLURRY_ARR_TEST_DIR):
        os.mkdir(BLURRY_ARR_TEST_DIR)
    BLURRY_ARR_TRAIN_DIR = BLURRY_ARRAYS_DIR + 'train/'
    if not os.path.exists(BLURRY_ARR_TRAIN_DIR):
        os.mkdir(BLURRY_ARR_TRAIN_DIR)
    BLURRY_ARR_VAL_DIR = BLURRY_ARRAYS_DIR + 'val/'
    if not os.path.exists(BLURRY_ARR_VAL_DIR):
        os.mkdir(BLURRY_ARR_VAL_DIR)

    # define blur kernel size (choose an odd size)
    k = 11

    for fn in tqdm(os.listdir(TEST_DIR)):
        if fn[-3:] != 'jpg':
            continue
        # calculate a random blur kernel for horizontal and vertical motion blur
        vert_blur = random.randint(1, k)
        hor_blur = random.randint(1, k)
        bk = np.zeros((k, k))
        i_vert = (k - vert_blur) // 2
        i_hor = (k - hor_blur) // 2
        tot = -1
        for i in range(k):
            if i >= i_vert and i <= (i_vert + vert_blur):
                bk[i, int((k - 1) / 2)] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1
            if i >= i_hor and i <= (i_hor + hor_blur):
                bk[int((k - 1) / 2), i] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1

        # get next image, blur it and save as image and array
        inImg = np.array(cv.imread(TEST_DIR + fn))
        outImg = cv.filter2D(inImg, -1, bk)
        outAr = np.asarray(outImg)
        cv.imwrite(BLURRY_TEST_DIR + fn, outImg)
        np.save(BLURRY_ARR_TEST_DIR + fn[:-4], outAr)
        #cv.imshow('image', outImg)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

    for fn in tqdm(os.listdir(TRAIN_DIR)):
        if fn[-3:] != 'jpg':
            continue
        vert_blur = random.randint(1, k)
        hor_blur = random.randint(1, k)
        bk = np.zeros((k, k))
        i_vert = (k - vert_blur) // 2
        i_hor = (k - hor_blur) // 2
        tot = -1
        for i in range(k):
            if i >= i_vert and i <= (i_vert + vert_blur):
                bk[i, int((k - 1) / 2)] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1
            if i >= i_hor and i <= (i_hor + hor_blur):
                bk[int((k - 1) / 2), i] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1
        inImg = np.array(cv.imread(TRAIN_DIR + fn))
        outImg = cv.filter2D(inImg, -1, bk)
        outAr = np.asarray(outImg)
        cv.imwrite(BLURRY_TRAIN_DIR + fn, outImg)
        np.save(BLURRY_ARR_TRAIN_DIR + fn[:-4], outAr)


    for fn in tqdm(os.listdir(VAL_DIR)):
        if fn[-3:] != 'jpg':
            continue
        # calculate a random blur kernel for horizontal and vertical motion blur
        vert_blur = random.randint(1, k)
        hor_blur = random.randint(1, k)
        bk = np.zeros((k, k))
        i_vert = (k - vert_blur) // 2
        i_hor = (k - hor_blur) // 2
        tot = -1
        for i in range(k):
            if i >= i_vert and i <= (i_vert + vert_blur):
                bk[i, int((k - 1) / 2)] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1
            if i >= i_hor and i <= (i_hor + hor_blur):
                bk[int((k - 1) / 2), i] = float(1 / (vert_blur + hor_blur - 1))
                tot += 1

        # get next image, blur it and save as image and array
        inImg = np.array(cv.imread(VAL_DIR + fn))
        outImg = cv.filter2D(inImg, -1, bk)
        outAr = np.asarray(outImg)
        cv.imwrite(BLURRY_VAL_DIR + fn, outImg)
        np.save(BLURRY_ARR_VAL_DIR + fn[:-4], outAr)

# Some cv commands for reference
'''
    output = cv.filter2D(orig, -1, vertBk)
    cv.imshow('image', finalout)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''









