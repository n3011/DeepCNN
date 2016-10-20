#-------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com 
# Copyright 2016, Mrinal Haloi
#-------------------------------------------------------------------#
"""Read and preprocess image data.
 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.
 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.
 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.
 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.
 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import scipy.ndimage
from config.config import cfg


def inputs(dataset, batch_size=None, num_preprocess_threads=None):
    """Generate batches of ImageNet images for evaluation.
    Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.
    Returns:
        images: Images. 4D tensor of size [batch_size, cfg.TRAIN.image_size,cfg.TRAIN.image_size, 3].
        labels: 1-D integer Tensor of [cfg.TRAIN.batch_size].
    """
    if not batch_size:
        batch_size = cfg.TRAIN.batch_size

    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset, batch_size, train=False, num_preprocess_threads=num_preprocess_threads, num_readers=1)

    return images, labels


def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
    """Generate batches of distorted versions of Training images.
    Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
            None defaults to cfg.num_preprocess_threads.
    Returns:
        images: Images. 4D tensor of size [batch_size, cfg.TRAIN.crop_image_size, cfg.TRAIN.crop_image_size, 3].
        labels: 1-D integer Tensor of [cfg.TRAIN.batch_size].
    """
    if not batch_size:
        batch_size = cfg.TRAIN.batch_size

    with tf.device('/cpu:0'):
        images, labels = batch_inputs(dataset, batch_size, train=True, num_preprocess_threads=num_preprocess_threads, num_readers=cfg.num_readers)
    return images, labels

def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for name_scope.
    Returns:
        color-distorted image
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def random_crop(image, crop_width, crop_height, padding=None):
    """Randmly crop a image.
    Args:
        image: 3-D float Tensor of image
        crop_width:int, output image width
        crop_height: int, output image height
        padding: int, padding use to restore original image size, padded with 0's
    Returns:
        3-D float Tensor of randomly flipped updown image used for training.
    """
    if not crop_width:
        crop_width = cfg.TRAIN.crop_width
    if not crop_height:
        crop_height = cfg.TRAIN.crop_height
    oshape = np.shape(image)
    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    npad = ((padding, padding), (padding, padding), (0, 0))
    modified_image = image
    if padding:
        modified_image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_width)
    nw = random.randint(0, oshape[1] - crop_height)
    modified_image = modified_image[nh:nh + crop_width, nw:nw + crop_height]
    return modified_image

def random_flip_leftright(image):
    """Randmly flip one image updown.
    Args:
        image: 3-D float Tensor of image
    Returns:
        3-D float Tensor of randomly flipped left right image used for training.
    """
    if bool(random.getrandbits(1)):
        image = np.fliplr(image)
    return image

def random_flip_updown(image):
    """Randmly flip one image updown.
    Args:
        image: 3-D float Tensor of image
    Returns:
        3-D float Tensor of randomly flipped updown image used for training.
    """
    if bool(random.getrandbits(1)):
        image = np.flipud(image)
    return image

def random_rotation(image, max_angle):
    """Randmly rotate one image. Random rotation introduces rotation invarant in the image.
    Args:
        image: 3-D float Tensor of image
        max_angle: float, max value of rotation
    Returns:
        3-D float Tensor of randomly rotated image used for training.
    """
    if not max_angle:
        max_angle = cfg.TRAIN.max_angle
    if bool(random.getrandbits(1)):
        angle = random.uniform(-max_angle, max_angle)
        image = scipy.ndimage.interpolation.rotate(image, angle, reshape=False)
    return image

def random_blur(image, sigma_max):
    """Randmly blur one image with gaussian blur. Bluring reduces noise present in the image.
    Args:
        image: 3-D float Tensor of image
        sigma_max: maximum value of standard deviation to use
    Returns:
        3-D float Tensor of randomly blurred image used for training.
    """
    if not sigma_max:
        sigma_max = cfg.TRAIN.sigma_max
    if bool(random.getrandbits(1)):
        sigma = random.uniform(0., sigma_max)
        image = scipy.ndimage.filters.gaussian_filter(image, sigma)
    return image




def distort_image(image, height, width, thread_id=0, scope=None):
    """Distort one image for training a network.
    Args:
        image: 3-D float Tensor of image
        height: integer
        width: integer
        thread_id: integer indicating the preprocessing thread.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(scope, 'distort_image', [image, height, width]):
        # Crop the image to the specified bounding box.
        # Resize image as per memroy constarints
        image = tf.image.resize_images(image, cfg.TRAIN.im_height, cfg.TRAIN.im_width, 3)

        distorted_image = tf.random_crop(image, [cfg.TRAIN.crop_height, cfg.TRAIN.crop_width, 3], 12345)
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, cfg.TRAIN.crop_height, cfg.TRAIN.crop_width, resize_method)
        if not thread_id:
            tf.image_summary('cropped_resized_image', tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Randomly flip the image up and down.
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)
        if not thread_id:
            tf.image_summary('final_distorted_image', tf.expand_dims(distorted_image, 0))
        return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.
    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
        return image

def image_preprocessing(image_buffer, train, thread_id=0):
    """Decode and preprocess one image for evaluation or training.
    Args:
        image_buffer: JPEG encoded string Tensor
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        train: boolean
        thread_id: integer indicating preprocessing thread
    Returns:
        3-D float Tensor containing an appropriately scaled image
    """

    image = decode_jpeg(image_buffer)
    height = cfg.TRAIN.im_height
    width = cfg.TRAIN.im_width

    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label, features['image/class/text']


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
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
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)
        if num_preprocess_threads is None:
            num_preprocess_threads = cfg.num_dataprocess_threads

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple ' 'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = cfg.num_readers

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        examples_per_shard = 1024
        min_queue_examples = examples_per_shard * cfg.input_queue_memory_factor
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
            image_buffer, label_index, _ = parse_example_proto(example_serialized)
            image = image_preprocessing(image_buffer, train, thread_id)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(images_and_labels,
            batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = cfg.TRAIN.im_height
        width = cfg.TRAIN.im_width
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])
