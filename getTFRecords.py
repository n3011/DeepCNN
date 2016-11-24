# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from dataset import image_to_tfrecords as im2tf


if __name__ == '__main__':
    im2r = im2tf.TFRecords()
    im2r.process_dataset('tf_records_dataset', 'validation_images', 16, 'validation_labels.csv')
