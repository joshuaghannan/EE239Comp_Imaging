# Adding python version of ImageGenerator, so we can serialize the function for multiprocess to access it in a
# Jupyter Notebook

import numpy as np
import keras
import cv2
import os
import imgaug as ia
import imgaug.augmenters as iaa
import scipy


########################################################

# Custom Keras Data Generators:

class ImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, x_path, y_path, batch_size, sigma_=0, blur_kernel_=None, crop_size=180, augment=True,
                 shuffle=True, seed=42, normalize=False):
        'Initialization'
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        np.random.seed(seed)
        self.seed = seed
        self.sigma = sigma_
        self.blur_kernel = blur_kernel_
        self.normalize = normalize

        # generate the list of images available from x_path
        self.x_image_names = self.get_image_names(self.x_path)
        self.y_image_names = self.get_image_names(self.y_path)

        # for this case, the file names in each path should equal one another:
        assert [os.path.basename(path) for path in self.x_image_names] == \
               [os.path.basename(path) for path in self.y_image_names]

        # convert x_image_names and y_image_names to np arrays for easy indexing
        self.x_image_names = np.array(self.x_image_names)
        self.y_image_names = np.array(self.y_image_names)

        # determine total images in our generator
        self.total_images = len(self.x_image_names)

        self.indexes = []
        self.on_epoch_end()

        # Augmentation pipeline:
        # Crop to width/height
        self.crop = iaa.Sequential([
            iaa.CropToFixedSize(width=crop_size, height=crop_size, position="uniform"),
        ])
        # Flip images sometimes
        self.flip = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        # Rotate the image in a 90 degree interval 75% of the time
        self.rotate_sometimes = iaa.OneOf([
            iaa.Rot90(0),
            iaa.Rot90(1),
            iaa.Rot90(2),
            iaa.Rot90(3)
        ])
        # Add White Gaussian noise
        self.add_noise = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=self.sigma)
        ])

    @staticmethod
    def get_image_names(src_path):
        raw = sorted(os.listdir(src_path))
        return [os.path.join(src_path, name) for name in raw if '.jpg' in name]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.total_images, dtype=np.int32)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        assert int(np.floor(self.total_images / self.batch_size)) > 0
        return int(np.floor(self.total_images / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x_aug, y_aug = self.__data_generation(indexes)

        return x_aug, y_aug

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'

        # pull in ONLY the images we need
        x_ = [cv2.imread(im_path) for im_path in self.x_image_names[indexes]]
        y_ = [cv2.imread(im_path) for im_path in self.y_image_names[indexes]]

        if self.augment:
            # perform augmentations on x:
            ia.seed(self.seed)
            # (1) rotate
            x_ = self.rotate_sometimes(images=x_)
            # (2) flip
            x_ = self.flip(images=x_)
            # (2.5) blur (prior to cropping)
            if self.blur_kernel is not None:
                x_ = [self.apply_blur(x.astype('float32')) for x in x_]
            # (3) crop
            # ia.seed(self.seed)
            x_ = self.crop(images=x_)
            # (3.5) add noise
            if self.sigma > 0:
                x_ = self.add_noise(images=x_)

            # perform augmentations on y:
            ia.seed(self.seed)
            # (1) rotate
            y_ = self.rotate_sometimes(images=y_)
            # (2) flip
            y_ = self.flip(images=y_)
            # (3) crop
            # ia.seed(self.seed)
            y_ = self.crop(images=y_)

        x_ = np.array(x_)
        y_ = np.array(y_)

        # normalize inputs from 0-255 to 0.0-1.0
        if self.normalize:
            x_ = x_.astype('float32')
            x_ = x_ / 255.0

            y_ = y_.astype('float32')
            y_ = y_ / 255.0

        # update to use new seed next time
        self.seed = np.random.randint(1, 100000)

        return x_, y_

    def apply_blur(self, x):
        x_blurred = [scipy.ndimage.convolve(x[..., c], self.blur_kernel, mode='constant') for c in range(x.shape[2])]
        x_blurred = np.stack(x_blurred)
        x_blurred = np.moveaxis(x_blurred, 0, -1)
        return x_blurred