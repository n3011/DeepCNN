# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os
import re
import math
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from config.config import cfg
from utils import data_augmentation

slim = tf.contrib.slim


class CNNModel(object):
    """A high level class for Convolutional Neural Network model definition"""
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self._name = name
        self._lr = 0.001
        self._weight_Decay = 0.000004
        self._batch_norm_decay = 0.9997
        self._batch_norm_epsilon = 0.001
        self._solver_type = 'SGD'
        self._momentum = 0.9
        self._dropout_keep_prob = 0.8

    @abstractmethod
    def get_num_conv_layers(self):
        """Returns the number of convolutional layers in the network"""
        pass

    @abstractmethod
    def with_batchnorm(self):
        """Returns True if bathnorm is applied to reduce internal covariate shift"""
        pass

    @abstractmethod
    def with_dropout(self):
        """Returns true if dropout is applied to reduce overfitting"""
        pass

    @abstractmethod
    def get_num_fc_layers(self):
        """Returns total number of fully connected layers in the network"""
        pass

    @abstractmethod
    def set_net_params(self, params):
        """Set the parameters for training the network"""
        pass

    @abstractmethod
    def get_net_params(self):
        """Set the parameters for training the network"""
        pass

    def inference_graph(self, images, num_classes, for_training=False, restore_logits=True, scope=None, use_auxiliary=False):
        """Build the inference graph
        See here for reference: http://arxiv.org/abs/1512.00567

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
        # net_params = self.get_net_params()
        # Parameters for BatchNorm.
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': cfg.TRAIN.batchnorm_moving_average_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
        }
        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fc], weight_decay=0.00004):
            with slim.arg_scope([slim.conv2d], stddev=0.1, activation=tf.nn.relu, batch_norm_params=batch_norm_params):
                logits, endpoints = self.model_def(images, dropout_keep_prob=0.8, num_classes=num_classes, is_training=for_training, restore_logits=restore_logits, scope=scope)

        # Add summaries for viewing model statistics on TensorBoard.
        self._activation_summaries(endpoints)
        # Grab the logits associated with the side head. Employed during training.
        if use_auxiliary:
            auxiliary_logits = endpoints['aux_logits']
            return logits, auxiliary_logits
        else:
            return logits

    def add_loss(self, logits, labels, batch_size=None, use_auxiliary=False):
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

        # Reshape the labels into a dense Tensor of shape [cfg.TRAIN.batch_size, num_classes].
        sparse_labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        if use_auxiliary:
            num_classes = logits[0].get_shape()[-1].value
            dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
            slim.losses.softmax_cross_entropy(logits[0], dense_labels, label_smoothing=0.1, weight=1.0)
            slim.losses.softmax_cross_entropy(logits[1], dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')
        else:
            num_classes = logits.get_shape()[-1].value
            dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
            slim.losses.softmax_cross_entropy(logits, dense_labels, label_smoothing=0.1, weight=1.0)

        # dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)
        # Cross entropy loss for the main softmax prediction.
        # slim.losses.cross_entropy_loss(logits[0], dense_labels, label_smoothing=0.1, weight=1.0)
        # Cross entropy loss for the auxiliary softmax head.
        # if use_auxiliary:
        # slim.losses.cross_entropy_loss(logits[1], dense_labels, label_smoothing=0.1, weight=0.4, scope='aux_loss')

    @abstractmethod
    def model_def(self, inputs, dropout_keep_prob=0.8, num_classes=5, is_training=True, restore_logits=True, scope=''):
        """Define CNN model architecture"""
        pass

    def _activation_summary(self, x):
        """create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
            x: Tensor
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % cfg.TRAIN.tower_name, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _activation_summaries(self, endpoints):
        with tf.name_scope('summaries'):
            for act in endpoints.values():
                self._activation_summary(act)

    def _tower_loss(self, images, labels, num_classes, scope, use_auxiliary=False):
        """Calculate the total loss on a single tower running the ImageNet model.

        We perform 'batch splitting'. This means that we cut up a batch across
        multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
        then each tower will operate on an batch of 16 images.

        Args:
            images: Images. 4D tensor of size [batch_size, cfg.TRAIN.im_height,
                                           cfg.TRAIN.im_width, 3].
            labels: 1-D integer Tensor of [batch_size].
            num_classes: number of classes
            scope: unique prefix string identifying the ImageNet tower, e.g.
            'tower_0'.

        Returns:
            Tensor of shape [] containing the total loss for a batch of data
        """
        # When fine-tuning a model, we do not restore the logits but instead we
        # randomly initialize the logits. The number of classes in the output of the
        # logit is the number of classes in specified Dataset.
        restore_logits = not cfg.TRAIN.fine_tune

        # Build inference Graph.
        logits = self.inference_graph(images, num_classes, for_training=True, restore_logits=restore_logits, scope=scope, use_auxiliary=use_auxiliary)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        split_batch_size = images.get_shape().as_list()[0]
        self.add_loss(logits, labels, batch_size=split_batch_size, use_auxiliary=use_auxiliary)

        # Assemble all of the losses for the current tower only.
        # losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

        # Calculate the total loss for the current tower.
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
        losses = slim.losses.get_losses()
        total_loss = slim.losses.get_total_loss(add_regularization_losses=True, name='total_loss')
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summmary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on TensorBoard.
            loss_name = re.sub('%s_[0-9]*/' % cfg.TRAIN.tower_name, '', l.op.name)
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(loss_name +' (raw)', l)
            tf.scalar_summary(loss_name, loss_averages.average(l))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _sum_clones_gradients(self, clone_grads):
        """Calculate the sum gradient for each shared variable across all gpus.

        This function assumes that the clone_grads has been scaled appropriately by
        1 / num_towers/gpu.

        Args:
            tower_grads: A List of List of tuples (gradient, variable), one list per
            tower.

        Returns:
            List of tuples of (gradient, variable) where the gradient has been summed
            across all gpus.
        """
        sum_grads = []
        for grad_and_vars in zip(*clone_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
            grads = []
            var = grad_and_vars[0][1]
            for g, v in grad_and_vars:
                assert v == var
                if g is not None:
                    grads.append(g)
            if grads:
                if len(grads) > 1:
                    sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
                else:
                    sum_grad = grads[0]
                sum_grads.append((sum_grad, var))
        return sum_grads

    def _eval_once(self, saver, summary_writer, top_1_op, top_2_op, summary_op):
        """Runs Validation on the validation or testing set.

        Args:
            saver: Saver.
            summary_writer: Summary writer.
            top_1_op: Top 1 op.
            top_2_op: Top 2 op.
            summary_op: Summary op.
        """
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cfg.TRAIN.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(cfg.TRAIN.checkpoint_dir, ckpt.model_checkpoint_path))

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %(ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(cfg.TRAIN.num_examples / cfg.VAL.batch_size))
                # Counts the number of correct predictions.
                count_top_1 = 0.0
                count_top_2 = 0.0
                total_sample_count = num_iter * cfg.VAL.batch_size
                step = 0

                print('%s: starting evaluation on (%s).' % (datetime.now(), cfg.VAL.subset))
                start_time = time.time()
                while step < num_iter and not coord.should_stop():
                    top_1, top_2 = sess.run([top_1_op, top_2_op])
                    count_top_1 += np.sum(top_1)
                    count_top_2 += np.sum(top_2)
                    step += 1
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = cfg.VAL.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f''sec/batch)' % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                        start_time = time.time()

                # Compute precision @ 1.
                precision_at_1 = count_top_1 / total_sample_count
                recall_at_2 = count_top_2 / total_sample_count
                print('%s: precision @ 1 = %.4f recall @ 2 = %.4f [%d examples]' %(datetime.now(), precision_at_1, recall_at_2, total_sample_count))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
                summary.value.add(tag='Recall @ 2', simple_value=recall_at_2)
                summary_writer.add_summary(summary, global_step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    def test(self, dataset, moving_average_decay, test_dir, test_interval, run_once=False):
        """Test model on Dataset for a number of steps."""
        with tf.Graph().as_default():
            # Get images and labels from the dataset.
            images, labels = data_augmentation.inputs(dataset)

            # Number of classes in the Dataset label set plus 1.
            # Label 0 is reserved for an (unused) background class.
            num_classes = dataset.num_classes()

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = self.inference_graph(images, num_classes)

            # Calculate predictions.
            top_1_op = tf.nn.in_top_k(logits, labels, 1)
            top_2_op = tf.nn.in_top_k(logits, labels, 2)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            graph_def = tf.get_default_graph().as_graph_def()
            summary_writer = tf.train.SummaryWriter(test_dir, graph_def=graph_def)

            while True:
                self._eval_once(saver, summary_writer, top_1_op, top_2_op, summary_op)
                if run_once:
                    break
                time.sleep(test_interval)
