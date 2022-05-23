"""
Code for loading CIFAR10 and MNIST for training.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def get_image_input(name='MNIST', batch_size=100, width=32, height=None, pad=True, pad_value=0):
    if height is None:
        height = width

    if name == 'MNIST':
        image_input = _MNIST_image_batch(batch_size)

    if name == 'CIFAR10':
        image_input = _CIFAAR10_image_batch(batch_size)

    if pad == False:
        image_input = tf.clip_by_value(tf.image.resize(image_input, [width, height], method=tf.image.ResizeMethod.BICUBIC, align_corners=True), clip_value_min=0, clip_value_max=1)
        image_input = tf.transpose(image_input, perm=[0, 3, 1, 2])
    else:
        image_input = tf.transpose(image_input, perm=[0, 3, 1, 2])
        image_input = _pad_image_batch_to_size(image_input, width, height, pad_value)

    return image_input

def _pad_image_batch_to_size(image_input, width, height, pad_value):
    s = tf.shape(image_input).eval()
    w_0 = s[2]
    h_0 = s[3]

    # If no padding is needed, stop
    if w_0 == width and h_0 == height:
        return image_input

    pad_size_x = (width - w_0) // 2
    pad_size_y = (height - h_0) // 2

    if width > 2 * pad_size_x + w_0:
        delta_x = 1
    else:
        delta_x = 0

    if height > 2 * pad_size_y + h_0:
        delta_y = 1
    else:
        delta_y = 0

    paddings = tf.constant([[0, 0], [0, 0], [pad_size_x, pad_size_x + delta_x], [pad_size_y, pad_size_y + delta_y]])
    image_input = tf.pad(image_input, paddings=paddings, mode='CONSTANT', constant_values=pad_value)
    return image_input

def _MNIST_image_batch(batch_size):
    with tf.device("/cpu:0"):
        train, test = tf.keras.datasets.mnist.load_data()
        im_train, _ = train
        im_test, _ = test
        images_nparray = np.array(im_train, dtype=np.float) / 255.0
        number_of_images_in_dataset = images_nparray.shape[0]
        images_tensor = tf.constant(1 - images_nparray, dtype=tf.float32)
        random_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=number_of_images_in_dataset, dtype=tf.int32)
        image_input = tf.gather(images_tensor, random_indices, axis=0)
        image_input = tf.reshape(image_input, shape=[batch_size, 28, 28, 1])
        #image_input = tf.transpose(image_input, perm=[0, 3, 1, 2])
        return image_input

def get_MNIST_np_array():
    train, test = tf.keras.datasets.mnist.load_data()
    im_train, _ = train
    im_test, _ = test
    images_nparray = np.array(np.concatenate([im_train, im_test], axis=0), dtype=np.float) / 255.0
    images_nparray = images_nparray[:1, :, :]
    return 1 - images_nparray


def _CIFAAR10_image_batch(batch_size):
    with tf.device("/cpu:0"):
        train, test = tf.keras.datasets.cifar10.load_data()
        im_train, _ = train
        im_test, _ = test
        images_nparray = np.array(im_train, dtype=np.float) / 255.0
        number_of_images_in_dataset = images_nparray.shape[0]
        images_tensor = tf.constant(images_nparray, dtype=tf.float32)
        random_indices = tf.random_uniform(shape=[batch_size], minval=0, maxval=number_of_images_in_dataset, dtype=tf.int32)
        image_input = tf.gather(images_tensor, random_indices, axis=0)
        image_input = tf.reshape(image_input, shape=[batch_size, 32, 32, 3])
        #image_input = tf.transpose(image_input, perm=[0, 3, 1, 2])
        return image_input
