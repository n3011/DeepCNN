#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from models import inception_resnet_v2
from dataset import dataset

from config.config import cfg


def main(_):
    dataset = dataset.Dataset(subset='train')
    assert dataset.data_files()
    ir_train = inception_resnet_v2.InceptionResnet('InceptionResnet')
    ir_train.train(dataset, 'train.ckpt', 0.001, num_classes=2, batch_size=32, max_steps=1000000000000, train_dir=cfg.TRAIN.train_dir, tower_name=cfg.TRAIN.tower_name, optname='adam', decay=0.9, momentum=0.9, epsilon=0.0000008, beta1=0.9, beta2=0.999, num_epoch_per_decay=30, lr_decay_factor=0.16)


if __name__ == '__main__':
    tf.app.run()
