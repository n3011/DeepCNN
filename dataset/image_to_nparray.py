#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#
import cv3
import numpy as np

from PIL import Image
import os

import glob


def read_images_from(data_dir):
    images = []
    jpeg_files_path = glob.glob(os.path.join(data_dir, '*.[jJ][pP][eE][gG]'))
    for filename in jpeg_files_path[:10]:
        print filename
        im = Image.open(filename)
        im = im.resize((512, 512), Image.ANTIALIAS)
        im = np.asarray(im, np.uint8)
        images.append(im)
        print im[0][1][1]

    images_only = [np.asarray(image, np.uint8) for image in images]  # Use unint8 or you will be !!!
    images_only = np.array(images_only)
    print(images_only.shape)
    return images_only

if __name__ == '__main__':
    images = read_images_from('/home/haloi/Data/train/')
    np.save("train.npy", images)
