"""Contains the definition of the Inception Resnet V2 architecture.
#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
import copy

from tensorflow.python.framework import ops

from config.config import cfg
from models.basecnn import CNNModel
from utils import data_augmentation
slim = tf.contrib.slim


class InceptionResnet(CNNModel):
    """A class for Inception v3 model definition"""

    def __init__(self, name):
        super(InceptionResnet, self).__init__(name)

    def get_num_conv_layers(self):
        """Calculate number of convolutional layers in the network and return"""
        return 120

    def get_num_fc_layers(self):
        """Calculate number of fully connected layers and return"""
        return 1

    def with_batchnorm(self):
        """Test whether it has batch norm"""
        return True

    def with_dropout(self):
        """"Check whether network has dropout"""
        return True

    def get_net_params(self):
        """Get the parametrs used to train the cnn network"""
        params = {
                'lr': self._lr,
                'weight_decay': self._weight_Decay,
                'batch_norm_decay': self._batch_norm_decay,
                'batch_norm_epsilon': self._batch_norm_epsilon,
                'dropout_keep_prob': self._dropout_keep_prob,
                }
        return params

    def set_net_params(self, params):
        """Set networks parameters for training"""
        self._lr = params['lr']
        self._weight_Decay = params['weight_decay']
        self._batch_norm_decay = params['batch_norm_decay']
        self._batch_norm_epsilon = params['batch_norm_epsilon']

    def inference_graph(self, images, num_classes, for_training=False, restore_logits=True, scope=None, use_auxiliary=False):
        """Build Inception v3 model architecture.
        Args:
            images: Images returned from inputs() or distorted_inputs().
            num_classes: number of classes
            for_training: If set to `True`, build the inference model for training.
            Kernels that operate differently for inference during training
            e.g. dropout, are appropriately configured.
            restore_logits: whether or not the logits layers should be restored.
            Useful for fine-tuning a model with different num_classes.
            scope: optional prefix string identifying the ImageNet tower.
        Returns:
            Logits. 2-D float Tensor.
            Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
        """
        net_params = self.get_net_params()
        # Parameters for BatchNorm.
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': cfg.TRAIN.batchnorm_moving_average_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': cfg.TRAIN.batchnorm_epsilon,
        }
        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004), biases_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                logits, endpoints = self.model_def(images, dropout_keep_prob=0.8, num_classes=num_classes, is_training=for_training,
                    restore_logits=restore_logits, scope=scope)

        # Add summaries for viewing model statistics on TensorBoard.
        # self._activation_summaries(endpoints)
        # Grab the logits associated with the side head. Employed during training.
        if use_auxiliary:
            auxiliary_logits = endpoints['aux_logits']
            return logits, auxiliary_logits
        else:
            return logits

    def add_loss(self, logits, labels, batch_size=None, use_auxiliary=True):
        """Adds all losses for the model.
        Note the final loss is not returned. Instead, the list of losses are collected
        by slim.losses. The losses are accumulated in tower_loss() and summed to
        calculate the total loss.

        Args:
            logits: List of logits from inference(). Each entry is a 2-D float Tensor.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
            batch_size: integer
        """
        if not batch_size:
            batch_size = cfg.TRAIN.batch_size

        # Reshape the labels into a dense Tensor of shape cfg.TRAIN.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = logits.get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
        # Cross entropy loss for the main softmax prediction.
        slim.losses.softmax_cross_entropy(logits, dense_labels, label_smoothing=0.1, weight=1.0)

    def block35(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Builds the 35x35 resnet block."""
        with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
            mixed = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net

    def block17(iself, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Builds the 17x17 resnet block."""
        with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7], scope='Conv2d_0b_1x7')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            mixed = tf.concat(3, [tower_conv, tower_conv1_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net

    def block8(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Builds the 8x8 resnet block."""
        with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3], scope='Conv2d_0b_1x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1], scope='Conv2d_0c_3x1')
            mixed = tf.concat(3, [tower_conv, tower_conv1_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net

    def model_def(self, inputs, dropout_keep_prob=0.8, num_classes=5, is_training=True, restore_logits=True, reuse=None, scope='InceptionResnetv2'):
        """Creates the Inception Resnet V2 model.

        Args:
            inputs: a 4-D tensor of size [batch_size, height, width, 3].
            num_classes: number of predicted classes.
            is_training: whether is training or not.
            dropout_keep_prob: float, the fraction to keep before final layer.
            reuse: whether or not the network and its variables should be reused. To be
                able to reuse 'scope' must be given.
            scope: Optional variable_scope.

        Returns:
            logits: the logits outputs of the model.
            end_points: the set of end_points from the inception model.
        """
        end_points = {}

        with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

                    # 149 x 149 x 32
                    net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    end_points['Conv2d_1a_3x3'] = net
                    # 147 x 147 x 32
                    net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                    end_points['Conv2d_2a_3x3'] = net
                    # 147 x 147 x 64
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                    end_points['Conv2d_2b_3x3'] = net
                    # 73 x 73 x 64
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                    end_points['MaxPool_3a_3x3'] = net
                    # 73 x 73 x 80
                    net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
                    end_points['Conv2d_3b_1x1'] = net
                    # 71 x 71 x 192
                    net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
                    end_points['Conv2d_4a_3x3'] = net
                    # 35 x 35 x 192
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_5a_3x3')
                    end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    with tf.variable_scope('Mixed_5b'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
                        net = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1])

                    end_points['Mixed_5b'] = net
                    net = slim.repeat(net, 10, self.block35, scale=0.17)

                    # 17 x 17 x 1024
                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, scope='Conv2d_0b_3x3')
                            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                        net = tf.concat(3, [tower_conv, tower_conv1_2, tower_pool])

                    end_points['Mixed_6a'] = net
                    net = slim.repeat(net, 20, self.block17, scale=0.10)

                    # Auxillary tower
                    with tf.variable_scope('AuxLogits'):
                        aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID', scope='Conv2d_1a_3x3')
                        aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
                        aux = slim.conv2d(aux, 768, aux.get_shape()[1:3], padding='VALID', scope='Conv2d_2a_5x5')
                        aux = slim.flatten(aux)
                        aux = slim.fully_connected(aux, num_classes, activation_fn=None, scope='Logits')
                        end_points['AuxLogits'] = aux

                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                        net = tf.concat(3, [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool])

                    end_points['Mixed_7a'] = net

                    net = slim.repeat(net, 9, self.block8, scale=0.20)
                    net = self.block8(net, activation_fn=None)

                    net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                    end_points['Conv2d_7b_1x1'] = net

                    with tf.variable_scope('Logits'):
                        end_points['PrePool'] = net
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8')
                        net = slim.flatten(net)

                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')

                        end_points['PreLogitsFlatten'] = net
                        logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
                        end_points['Logits'] = logits
                        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

            return logits, end_points

    # inception_resnet_v2.default_image_size = 299

    def inception_resnet_v2_arg_scope(self, weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        """Yields the scope with the default parameters for inception_resnet_v2.

        Args:
            weight_decay: the weight decay for weights variables.
            batch_norm_decay: decay for the moving average of batch_norm momentums.
            batch_norm_epsilon: small float added to variance to avoid dividing by zero.

        Returns:
            a arg_scope with the parameters needed for inception_resnet_v2.
        """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay), biases_regularizer=slim.l2_regularizer(weight_decay)):

            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params) as scope:
                return scope

    def lr_policy(self, initial_lr, global_step, num_batches_per_epoch, num_epochs_per_decay, lr_decay_factor=0.16, staircase=True):
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
        lr = tf.train.exponential_decay(initial_lr, global_step, decay_steps, lr_decay_factor, staircase=staircase)
        return lr

    def optimizer(self, lr, optname='adam', decay=0.9, momentum=0.9, epsilon=0.0000008, beta1=0.9, beta2=0.999):
        if optname == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr, decay, momentum=momentum, epsilon=epsilon)
        if optname == 'moemntum':
            opt = tf.train.MomentumOptimizer(lr, momentum, use_locking=False, name='Momentum', use_nesterov=True)
        if optname == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=False, name='Adam')
        return opt

    def train(self, dataset, output_model_name, initial_lr, num_classes=5, batch_size=32, max_steps=1000000000000, train_dir=None, tower_name='mvision', optname='rmsprop', decay=0.9, momentum=0.9, epsilon=0.0000008, beta1=0.9, beta2=0.999, num_epoch_per_decay=30, lr_decay_factor=0.16):
        """Train on dataset for a number of steps."""
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * cfg.num_gpus.
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            # Calculate the learning rate schedule.
            num_batches_per_epoch = (dataset.num_examples_per_epoch() / batch_size)

            # Decay the learning rate exponentially based on the number of steps.
            lr = self.lr_policy(initial_lr, global_step, num_batches_per_epoch, num_epoch_per_decay, lr_decay_factor=lr_decay_factor, staircase=True)

            # Create an optimizer that performs gradient descent.
            # Different optimizer may give different results
            opt = self.optimizer(lr, optname=optname, decay=decay, momentum=momentum, epsilon=epsilon, beta1=beta1, beta2=beta2)
            # Get images and labels for ImageNet and split the batch across GPUs.
            assert batch_size % cfg.num_gpus == 0, ('Batch size must be divisible by number of GPUs')

            # Override the number of preprocessing threads to account for the increased
            # number of GPU towers.
            num_preprocess_threads = cfg.num_dataprocess_threads * cfg.num_gpus
            images, labels = data_augmentation.distorted_inputs(dataset, num_preprocess_threads=num_preprocess_threads)

            input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

            # Number of classes in the Dataset label set plus 1.
            # Label 0 is reserved for an (unused) background class.
            # Split the batch of images and labels for towers.
            images_splits = tf.split(0, cfg.num_gpus, images)
            labels_splits = tf.split(0, cfg.num_gpus, labels)

            # Calculate the gradients for each model tower.
            tower_grads = []
            for i in xrange(cfg.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (tower_name, i)) as scope:
                        # Force all Variables to reside on the CPU.
                        with slim.arg_scope([slim.variable], device='/cpu:0'):
                            loss = self._tower_loss(images_splits[i], labels_splits[i], num_classes, scope, use_auxiliary=False)
                        tf.get_variable_scope().reuse_variables()
                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        batchnorm_updates = tf.get_collection(ops.GraphKeys.UPDATE_OPS, scope)
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            # grads = self._average_gradients(tower_grads)
            grads = self._sum_clones_gradients(tower_grads)
            # grads = tower_grads[0]
            # Add a summaries for the input processing and global_step.
            summaries.extend(input_summaries)

            # Add a summary to track the learning rate.
            summaries.append(tf.scalar_summary('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.histogram_summary(var.op.name, var))

            # Track the moving averages of all trainable variables.
            # Note that we maintain a "double-average" of the BatchNormalization
            # global statistics. This is more complicated then need be but we employ
            # this for backward-compatibility with our previous models.
            variable_averages = tf.train.ExponentialMovingAverage(cfg.TRAIN.moving_average_decay, global_step)

            # Another possiblility is to use tf.slim.get_variables().
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

            # Group all updates to into a single train op.
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.merge_summary(summaries)

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.memory_allocation)
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=cfg.TRAIN.log_device_placement))
            sess.run(init)

            if cfg.TRAIN.pretrained_model_checkpoint_path:
                assert tf.gfile.Exists(cfg.TRAIN.pretrained_model_checkpoint_path)
                saver = tf.train.import_meta_graph(cfg.TRAIN.pretrained_model_checkpoint_path + '.meta')
                # variables_to_restore = tf.get_collection(slim.get_variables_to_restore())
                # restorer = tf.train.Saver(variables_to_restore)
                saver.restore(sess, cfg.TRAIN.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %(datetime.now(), cfg.TRAIN.pretrained_model_checkpoint_path))

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            summary_writer = tf.train.SummaryWriter(train_dir, graph=sess.graph)

            for step in xrange(max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    examples_per_sec = batch_size / float(duration)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 5000 == 0 or (step + 1) == max_steps:
                    checkpoint_path = os.path.join(train_dir, output_model_name)
                    saver.save(sess, checkpoint_path, global_step=step)
