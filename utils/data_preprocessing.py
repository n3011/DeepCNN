#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#
from __future__ import division, print_function, absolute_import

import numpy as np
import pickle
import tensorflow as tf

_EPSILON = 1e-8


class DataPreprocessing(object):
    """ Data Preprocessing.
    Base class for applying common real-time data preprocessing.
    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined pre-processing methods will be applied at both
    training and testing time.
    Arguments:
        None.
    Parameters:
        methods: `list of function`. Augmentation methods to apply.
        args: A `list` of arguments to use for these methods.
    """

    def __init__(self, name="DataPreprocessing"):
        self.session = None
        # Data Persistence
        with tf.name_scope(name) as scope:
            self.scope = scope
        self.global_mean = self.PersistentParameter(scope, name="mean")
        self.global_std = self.PersistentParameter(scope, name="std")
        self.global_pc = self.PersistentParameter(scope, name="pc")


    def restore_params(self, session):
        self.global_mean.is_restored(session)
        self.global_std.is_restored(session)
        self.global_pc.is_restored(session)

    def initialize(self, dataset, session, limit=None):
        """ Initialize preprocessing methods that pre-requires
        calculation over entire dataset. """
        # If a value is already provided, it has priority
        if self.global_mean.value is not None:
            self.global_mean.assign(self.global_mean.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_mean.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... Calculating mean over all dataset "
                  "(this may take long)...")
            self.compute_global_mean(dataset, session, limit)
            print("Mean: " + str(self.global_mean.value) + " (To avoid "
                  "repetitive computation, add it to argument 'mean' of "
                  "`add_featurewise_zero_center`)")
        # If a value is already provided, it has priority
        if self.global_std.value is not None:
            self.global_std.assign(self.global_std.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_std.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... Calculating std over all dataset "
                  "(this may take long)...")
            self.compute_global_std(dataset, session, limit)
            print("STD: " + str(self.global_std.value) + " (To avoid "
                  "repetitive computation, add it to argument 'std' of "
                  "`add_featurewise_stdnorm`)")
        # If a value is already provided, it has priority
        if self.global_pc.value is not None:
            self.global_pc.assign(self.global_pc.value, session)
        # Otherwise, if it has not been restored, compute it
        if not self.global_pc.is_restored(session):
            print("---------------------------------")
            print("Preprocessing... PCA over all dataset "
                  "(this may take long)...")
            self.compute_global_pc(dataset, session, limit)
            with open('PC.pkl', 'wb') as f:
                pickle.dump(self.global_pc.value, f)
            print("PC saved to 'PC.pkl' (To avoid repetitive computation, "
                  "load this pickle file and assign its value to 'pc' "
                  "argument of `add_zca_whitening`)")


    def samplewise_zero_center(self, image):
        image -= np.mean(image, axis=0)
        return image

    def samplewise_stdnorm(self, image):
        image /= (np.std(image, axis=0) + _EPSILON)
        return image

    def featurewise_zero_center(self, image):
        image -= self.global_mean.value
        return image

    def featurewise_stdnorm(self, image):
        image /= (self.global_std.value + _EPSILON)
        return image

    def zca_whitening(self, image):
        flat = np.reshape(image, image.size)
        white = np.dot(flat, self.global_pc.value)
        s1, s2, s3 = image.shape[0], image.shape[1], image.shape[2]
        image = np.reshape(white, (s1, s2, s3))
        return image

    # ---------------------------------------
    #  Calulation with Persistent Parameters
    # ---------------------------------------

    def compute_global_mean(self, dataset, session, limit=None):
        """ Compute mean of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        mean = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray):
            mean = np.mean(_dataset)
        else:
            # Iterate in case of non numpy data
            for i in range(len(dataset)):
                mean += np.mean(dataset[i]) / len(dataset)
        self.global_mean.assign(mean, session)
        return mean

    def compute_global_std(self, dataset, session, limit=None):
        """ Compute std of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        std = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray):
            std = np.std(_dataset)
        else:
            for i in range(len(dataset)):
                std += np.std(dataset[i]) / len(dataset)
        self.global_std.assign(std, session)
        return std

    def compute_global_pc(self, dataset, session, limit=None):
        """ Compute the Principal Component. """
        _dataset = dataset
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        d = _dataset
        s0, s1, s2, s3 = d.shape[0], d.shape[1], d.shape[2], d.shape[3]
        flat = np.reshape(d, (s0, s1 * s2 * s3))
        sigma = np.dot(flat.T, flat) / flat.shape[1]
        U, S, V = np.linalg.svd(sigma)
        pc = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + _EPSILON))), U.T)
        self.global_pc.assign(pc, session)
        return pc

    # -----------------------
    #  Persistent Parameters
    # -----------------------

    class PersistentParameter:
        """ Create a persistent variable that will be stored into the Graph.
        """
        def __init__(self, scope, name):
            self.is_required = False
            with tf.name_scope(scope):
                with tf.device('/cpu:0'):
                    # One variable contains the value
                    self.var = tf.Variable(0., trainable=False, name=name,
                                           validate_shape=False)
                    # Another one check if it has been restored or not
                    self.var_r = tf.Variable(False, trainable=False,
                                             name=name+"_r")
            # RAM saved vars for faster access
            self.restored = False
            self.value = None

        def is_restored(self, session):
            if self.var_r.eval(session=session):
                self.value = self.var.eval(session=session)
                return True
            else:
                return False

        def assign(self, value, session):
            session.run(tf.assign(self.var, value, validate_shape=False))
            self.value = value
            session.run(self.var_r.assign(True))
            self.restored = True


class ImagePreprocessing(DataPreprocessing):
    """ Image Preprocessing.
    Base class for applying real-time image related pre-processing.
    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined pre-processing methods will be applied at both
    training and testing time. Note that ImageAugmentation is similar to
    ImagePreprocessing, but only applies at training time.
    """

    def __init__(self):
        super(ImagePreprocessing, self).__init__()
        self.global_mean_pc = False
        self.global_std_pc = False


    def normalize_image(self, batch):
        return np.array(batch) / 255.

    def crop_center(self, batch, shape):
        oshape = np.shape(batch[0])
        nh = int((oshape[0] - shape[0]) * 0.5)
        nw = int((oshape[1] - shape[1]) * 0.5)
        new_batch = []
        for i in range(len(batch)):
            new_batch.append(batch[i][nh: nh + shape[0], nw: nw + shape[1]])
        return new_batch


    # --------------------------------------------------
    #  Preprocessing Calculation (Overwrited from Base)
    # --------------------------------------------------

    def samplewise_zero_center(self, image, per_channel=False):
        if not per_channel:
            im_zero_mean = image - np.mean(image)
        else:
            im_zero_mean = image - np.mean(image, axis=(0, 1, 2), keepdims=True)
        return im_zero_mean

    def samplewise_stdnorm(self, image, per_channel=False):
        if not per_channel:
            im_std = np.std(image)
            im_zero_std = image / (im_std + _EPSILON)
        else:
            im_std = np.std(image, axis=(0, 1, 2), keepdims=True)
            im_zero_std = image / (im_std + _EPSILON)
        return im_zero_std

    # --------------------------------------------------------------
    #  Calulation with Persistent Parameters (Overwrited from Base)
    # --------------------------------------------------------------

    def compute_global_mean(self, dataset, session, limit=None):
        """ Compute mean of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        mean = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_mean_pc:
            mean = np.mean(_dataset)
        else:
            # Iterate in case of non numpy data
            for i in range(len(dataset)):
                if not self.global_mean_pc:
                    mean += np.mean(dataset[i]) / len(dataset)
                else:
                    mean += (np.mean(dataset[i], axis=(0, 1),
                             keepdims=True) / len(dataset))[0][0]
        self.global_mean.assign(mean, session)
        return mean

    def compute_global_std(self, dataset, session, limit=None):
        """ Compute std of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        std = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_std_pc:
            std = np.std(_dataset)
        else:
            for i in range(len(dataset)):
                if not self.global_std_pc:
                    std += np.std(dataset[i]) / len(dataset)
                else:
                    std += (np.std(dataset[i], axis=(0, 1),
                             keepdims=True) / len(dataset))[0][0]
        self.global_std.assign(std, session)
        return std

    def parse_example_proto(self, example_serialized):
        """Parses an Example proto containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the following fields:
            image/height: 462
            image/width: 581
            image/colorspace: 'RGB'
            image/channels: 3
            image/class/label: 615
            image/class/synset: 'n03623198'
            image/class/text: 'knee pad'
            image/object/bbox/xmin: 0.1
            image/object/bbox/xmax: 0.9
            image/object/bbox/ymin: 0.2
            image/object/bbox/ymax: 0.6
            image/object/bbox/label: 615
            image/format: 'JPEG'
            image/filename: 'ILSVRC2012_val_00041207.JPEG'
            image/encoded: <JPEG encoded string>
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.
        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
            text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value=''),
        }
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                       'image/object/bbox/ymin',
                                       'image/object/bbox/xmax',
                                       'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat(0, [ymin, xmin, ymax, xmax])

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        return features['image/encoded'], label, bbox, features['image/class/text']

    def batch_inputs(self, dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
        """Contruct batches of training or evaluation examples from the image dataset.
        Args:
            dataset: instance of Dataset class specifying the dataset.
            See dataset.py for details.
            batch_size: integer
            train: boolean
            num_preprocess_threads: integer, total number of preprocessing threads
            num_readers: integer, number of parallel readers
        Returns:
            images: 4-D float Tensor of a batch of images
            labels: 1-D integer Tensor of [batch_size].
        Raises:
            ValueError: if data is not found
        """
        with tf.name_scope('batch_processing'):
            data_files = dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            # Create filename_queue
            if train:
                filename_queue = tf.train.string_input_producer(data_files,
                                                          shuffle=True,
                                                          capacity=16)
            else:
                filename_queue = tf.train.string_input_producer(data_files,
                                                          shuffle=False,
                                                          capacity=1)
            if num_preprocess_threads is None:
                num_preprocess_threads = cfg.num_dataprocess_threads

            if num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                           'of 4 (%d % 4 != 0).', num_preprocess_threads)

            if num_readers is None:
                num_readers = cfg.num_readers

            if num_readers < 1:
                raise ValueError('Please make num_readers at least 1')

            examples_per_shard = 1024
                min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
            if train:
                examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                    min_after_dequeue=min_queue_examples, dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image_buffer, label_index, bbox, _ = self.parse_example_proto(example_serialized)
                image = image_preprocessing(image_buffer, bbox, train, thread_id)
                images_and_labels.append([image, label_index])
            images, label_index_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size,
                 capacity=2 * num_preprocess_threads * batch_size)

            # Reshape images into these desired dimensions.
            height = cfg.TRAIN.im_height
            width = cfg.TRAIN.im_width
            depth = 3

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[batch_size, height, width, depth])

            # Display the training images in the visualizer.
            tf.image_summary('images', images)

            return images, tf.reshape(label_index_batch, [batch_size])

