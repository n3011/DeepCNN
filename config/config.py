#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAIN = edict()
__C.VAL = edict()

__C.TRAIN.data_dir = '/media/Data/eyepacs/train/'
__C.TRAIN.train_dir = '/media/Data/gb/train_tfrecords/'
__C.TRAIN.test_dir = '/media/Data/gb/test_tfrecords/'
__C.TRAIN.pretrained_model_checkpoint_path = None
__C.VAL.run_once = False
__C.TRAIN.eval_interval_secs = 60 * 60
__C.TRAIN.log_device_placement = False
__C.num_gpus = 1
__C.memory_allocation = 1
__C.TRAIN.batch_size = 16
__C.TRAIN.im_height = 299
__C.TRAIN.im_width = 299
__C.TRAIN.crop_height = 299
__C.TRAIN.crop_width = 299# 256 for inception #128 for vgg
__C.TRAIN.validation_size = 10000
__C.TRAIN.validation_split = 10
__C.TRAIN.learning_rate = 0.001
__C.TRAIN.num_epochs = 200
__C.TRAIN.training_size = 40000
__C.TRAIN.num_class = 5
__C.TRAIN.label = ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate')
__C.TRAIN.model_dir = '/media/Data/eyepacs/output/model'
__C.TRAIN.checkpoint_dir = '/media/Data/eyepacs/output/ckpt'
__C.TRAIN.im_depth = 3
__C.num_dataprocess_threads = 8
__C.num_readers = 4
__C.queue_memory_factor = 8
__C.input_queue_memory_factor = 8
__C.TRAIN.max_angle = 11
__C.TRAIN.sigma_max = 0.01
__C.num_threads = 8
__C.num_shards = 8
__C.output_dir = '/media/Data/eyepacs/output'
__C.TRAIN.fine_tune = False

__C.TRAIN.tower_name = 'tower'
__C.TRAIN.batchnorm_moving_average_decay = 0.9997
__C.TRAIN.batchnorm_epsilon = 0.001
__C.TRAIN.initial_learning_rate = 0.001
__C.TRAIN.moving_average_decay = 0.9999
__C.TRAIN.rmsprop_decay = 0.9
__C.TRAIN.rmsprop_momentum = 0.9
__C.TRAIN.rmsprop_epsilon = 1.0
__C.TRAIN.sgd_momentum = 0.9
__C.TRAIN.adam_beta1 = 0.9
__C.TRAIN.adam_beat2 = 0.999
__C.TRAIN.adam_epsilon = 1e-08
__C.TRAIN.num_epochs_per_decay = 30
__C.TRAIN.learning_rate_decay_factor = 0.16
__C.TRAIN.max_steps = 9100000
__C.VAL.batch_size = 64
__C.VAL.subset = 'val'
