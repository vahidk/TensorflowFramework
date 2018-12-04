"""Image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from common.ops import shape_ops


def random_crop_image(image, landmark, margin, random):
  """Crop labeled image."""
  shape = shape_ops.get_shape(image)

  max_offset = np.array(margin) * 2
  crop_width, crop_height = np.array(shape[1::-1]) - max_offset

  if random:
    offset_x = tf.random_uniform([], maxval=max_offset[0], dtype=tf.int32)
    offset_y = tf.random_uniform([], maxval=max_offset[1], dtype=tf.int32)
  else:
    offset_x, offset_y = margin

  image = tf.image.crop_to_bounding_box(
    image, offset_y, offset_x, crop_height, crop_width)

  landmark = landmark - tf.to_float(tf.expand_dims([offset_x, offset_y], 0))

  return image, landmark

def compute_compact_crop(landmarks, input_size):
  int_labels = tf.to_int32(landmarks)
  minimum = tf.maximum(tf.reduce_min(int_labels, axis=1), 0)
  maximum = tf.minimum(tf.reduce_max(int_labels, axis=1), input_size - 1)
  centers = (minimum + maximum) // 2
  half_sizes = tf.reduce_max(maximum - minimum, axis=1, keep_dims=True) // 2
  low = tf.maximum(half_sizes - centers, 0)
  high = tf.maximum(half_sizes + centers - input_size, 0)
  shifts = tf.maximum(low, high)
  half_sizes -= tf.reduce_max(shifts, axis=1, keep_dims=True)
  offsets = centers - half_sizes
  offsets = tf.to_float(offsets)
  sizes = half_sizes * 2
  return offsets, sizes


def crop_images(images, landmarks, offsets, sizes, scales, target_size=None):
  if target_size is None:
    target_size = image_size = np.array(images.shape.as_list()[2:0:-1])

  def _crop_and_scale_image(input_):
    im, off, sz = input_
    im = tf.image.crop_to_bounding_box(im, off[1], off[0], sz[0], sz[0])
    im = tf.image.resize_images(im, target_size)
    return im

  int_offsets = tf.to_int32(offsets)
  images = tf.map_fn(
    _crop_and_scale_image, (images, int_offsets, sizes), dtype=tf.float32)

  scales = tf.to_float(np.array([target_size])) / tf.to_float(sizes)

  landmarks -= tf.expand_dims(offsets, 1) 
  landmarks *= tf.expand_dims(scales, 1)
  return images, landmarks


def augment_image(image):
  """Augment the input with its mirrored image."""
  flipped = tf.reverse(image, axis=[-2])
  image = tf.concat([image, flipped], axis=-2)
  return image

