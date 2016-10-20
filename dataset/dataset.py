#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import re
import tensorflow as tf
from config.config import cfg


class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name, subset):
        """Initialize dataset using a subset and the path to the data."""
        # assert subset in self.available_subsets(), self.available_subsets()
        self.name = name
        self.subset = subset

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train']

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.
        Returns:
            python list of all (sharded) data set files.
        Raises:
            ValueError: if there are not data_files matching the subset.
        """
        data_files = [f for f in os.listdir(cfg.TRAIN.train_dir) if re.match(self.subset, f)]
        data_files = [os.path.join(cfg.TRAIN.train_dir, f) for f in data_files]
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name, self.subset, cfg.TRAIN.train_dir))
            exit(-1)
        return data_files

    def reader(self):
        """Return a reader for a single entry from the data set.
        See io_ops.py for details of Reader class.
        Returns:
            Reader object that reads the data set.
        """
        return tf.TFRecordReader()
