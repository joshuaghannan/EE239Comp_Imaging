%%% Make blurry images using disk blur kernel with width 7 and linear
%%% motion blur with length 21.

clear;
clf;

%% Get files
IMAGES_DIR = './BSR/BSDS500/data/images/';
TEST_DIR = strcat(IMAGES_DIR, 'test/');
TRAIN_DIR = strcat(IMAGES_DIR, 'train/');
VAL_DIR = strcat(IMAGES_DIR, 'val/');

% Make folders to save images to
IMAGES_BLURDISK_DIR = './images_blurry_disk/';
IMAGES_BLURMOT_DIR = './images_blurry_motion/';
TEST_BLURDISK_DIR = strcat(IMAGES_BLURDISK_DIR, 'test/');
TEST_BLURMOT_DIR = strcat(IMAGES_BLURMOT_DIR, 'test/');
TRAIN_BLURDISK_DIR = strcat(IMAGES_BLURDISK_DIR, 'train/');
TRAIN_BLURMOT_DIR = strcat(IMAGES_BLURMOT_DIR, 'train/');
VAL_BLURDISK_DIR = strcat(IMAGES_BLURDISK_DIR, 'val/');
VAL_BLURMOT_DIR = strcat(IMAGES_BLURMOT_DIR, 'val/');

% Only need to run the following lines once.
s1 = mkdir(IMAGES_BLURDISK_DIR);
s2 = mkdir(IMAGES_BLURMOT_DIR);
s11 = mkdir(TEST_BLURDISK_DIR);
s21 = mkdir(TEST_BLURMOT_DIR);
s12 = mkdir(TRAIN_BLURDISK_DIR);
s22 = mkdir(TRAIN_BLURMOT_DIR);
s13 = mkdir(VAL_BLURDISK_DIR);
s23 = mkdir(VAL_BLURMOT_DIR);

%% Get images and make them blurry
% Make blur kernel
h_disk = fspecial('disk', 7);
h_motion = fspecial('motion', 21, 30); % At an angle of 30 degrees

save ./disk_blur_kernel.mat h_disk;
save ./motion_blur_kernel.mat h_motion;

% Noise characteristics
mean = 0;
sigma = 5.702;

% Iterate through train directory. No noise on training images.
counter = 1;
files = dir(strcat(TRAIN_DIR, '*.jpg'));
for file = files'
    img = imread(strcat(TRAIN_DIR, file.name));
    img_disk = imfilter(img, h_disk);
    img_motion = imfilter(img, h_motion);
    if counter == 1
%         images = {img};
%         images_disk = {img_disk};
%         images_motion = {img_motion};
        imwrite(img_disk, strcat(TRAIN_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(TRAIN_BLURMOT_DIR, file.name));
    else
%         images{counter} = img;
%         images_disk{counter} = img_disk;
%         images_motion{counter} = img_motion;
        imwrite(img_disk, strcat(TRAIN_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(TRAIN_BLURMOT_DIR, file.name));
    end
    counter = counter + 1;
end

% Iterate through test directory
counter = 1;
files = dir(strcat(TEST_DIR, '*.jpg'));
for file = files'
    img = imread(strcat(TEST_DIR, file.name));
    img_disk = imfilter(img, h_disk);
    img_motion = imfilter(img, h_motion);
    noise1 = uint8(sigma * randn(size(img)) + mean);
    noise2 = uint8(sigma * randn(size(img)) + mean);
    img_disk = img_disk + noise1;
    img_motion = img_motion + noise2;
    if counter == 1
%         images = {img};
%         images_disk = {img_disk};
%         images_motion = {img_motion};
        imwrite(img_disk, strcat(TEST_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(TEST_BLURMOT_DIR, file.name));
    else
%         images{counter} = img;
%         images_disk{counter} = img_disk;
%         images_motion{counter} = img_motion;
        imwrite(img_disk, strcat(TEST_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(TEST_BLURMOT_DIR, file.name));
    end
    counter = counter + 1;
end

% Iterate through val directory
counter = 1;
files = dir(strcat(VAL_DIR, '*.jpg'));
for file = files'
    img = imread(strcat(VAL_DIR, file.name));
    img_disk = imfilter(img, h_disk);
    img_motion = imfilter(img, h_motion);
    noise1 = uint8(sigma * randn(size(img)) + mean);
    noise2 = uint8(sigma * randn(size(img)) + mean);
    img_disk = img_disk + noise1;
    img_motion = img_motion + noise2;
    if counter == 1
%         images = {img};
%         images_disk = {img_disk};
%         images_motion = {img_motion};
        imwrite(img_disk, strcat(VAL_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(VAL_BLURMOT_DIR, file.name));
    else
%         images{counter} = img;
%         images_disk{counter} = img_disk;
%         images_motion{counter} = img_motion;
        imwrite(img_disk, strcat(VAL_BLURDISK_DIR, file.name));
        imwrite(img_motion, strcat(VAL_BLURMOT_DIR, file.name));
    end
    counter = counter + 1;
end

%% Look at images

% clf;
% i = 1;
% subplot(1,3,1);
% imshow(images{i});
% subplot(1,3,2);
% imshow(images_disk{i});
% subplot(1,3,3);
% imshow(images_motion{i});







