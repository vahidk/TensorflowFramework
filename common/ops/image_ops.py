"""Image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from common.ops import shape_ops


def center_crop(image, landmark, size):
  """Crop labeled image."""
  image_size = image.shape[-2].value
  if image_size == size:
    return image, landmark

  offset = (image_size - size) // 2
  image = tf.image.crop_to_bounding_box(
    image, offset, offset, size, size)
  if landmark is not None:
    landmark -= offset
  return image, landmark


def scale_image(image, landmark, size):
  image_size = image.shape[-2].value
  if image_size == size:
    return image, landmark
    
  image = tf.image.resize_images(image, [size, size])
  if landmark is not None:
    scale = tf.cast(size, tf.float32) / image_size
    landmark *= scale
  return image, landmark


def random_crop_image(image, landmark, margin):
  """Crop labeled image."""
  shape = shape_ops.get_shape(image)

  max_offset = np.array(margin) * 2
  crop_width, crop_height = np.array(shape[1::-1]) - max_offset

  offset = tf.random_uniform([2], maxval=max_offset, dtype=tf.int32)

  image = tf.image.crop_to_bounding_box(
    image, offset[1], offset[0], crop_height, crop_width)

  landmark = landmark - tf.cast(tf.expand_dims(offset, 0), tf.float32)

  return image, landmark


def transform(image, landmark, translation=[0, 0], rotation=0, scale=1):
  """Apply an affine transformation to the image."""
  image = tf.convert_to_tensor(image)
  landmark = tf.convert_to_tensor(landmark, dtype=tf.float32)
  translation = tf.convert_to_tensor(translation, dtype=tf.float32)
  rotation = tf.convert_to_tensor(rotation, dtype=tf.float32)
  scale = tf.convert_to_tensor(scale, dtype=tf.float32)
  # Generate a transformation matrix
  h, w = image.shape.as_list()[-3:-1]
  tx, ty = tf.unstack(translation, axis=-1)
  sc = tf.cos(rotation) / scale
  ss = tf.sin(rotation) / scale
  cx = (sc - 1) * w * 0.5 + ss * h * 0.5
  cy = -ss * w * 0.5 + (sc - 1) * h * 0.5
  ze = tf.zeros_like(scale)
  # Apply transformation to image
  p = tf.transpose([sc, ss, -cx - tx, -ss, sc, -cy - ty, ze, ze])
  image_shape = image.shape
  image = tf.contrib.image.transform(image, p, interpolation="BILINEAR")
  image.set_shape(image_shape)
  # Apply transformation to landmarks
  a_r = tf.linalg.inv(tf.transpose([[sc, -ss], [ss, sc]]))
  a_t = tf.expand_dims(tf.transpose([cx + tx, cy + ty]), -2)
  landmark = tf.matmul(landmark + a_t, a_r, transpose_b=True)
  return image, landmark


def random_transform(image, landmark, translation=[0, 0], rotation=0, scale=0):
  """Randomly apply an affine transformation to the image."""
  shape = shape_ops.get_shape(image)[:len(image.shape)-3]
  t = translation * tf.random_uniform(shape + [2], -1., 1., dtype=tf.float32)
  r = rotation * tf.random_uniform(shape, -1., 1., dtype=tf.float32)
  s = scale * tf.random_uniform(shape, -1., 1., dtype=tf.float32) + 1.
  return transform(image, landmark, t, r, s)


def flip_image(image, landmark, reorder, random):
  """Flip images and landmarks."""
  assert(landmark.shape.ndims == 2)
  w = image.shape[-2].value
  image_t = tf.image.flip_left_right(image)
  landmark_r = tf.gather(landmark, reorder)
  landmark_t = tf.stack([w - 1 - landmark_r[:, 0], landmark_r[:, 1]], -1)
  if random:
    flip = tf.random_uniform([]) > 0.5
    image_t = tf.cond(flip, lambda: image_t, lambda: image)
    landmark_t = tf.cond(flip, lambda: landmark_t, lambda: landmark)
  return image_t, landmark_t


def compute_compact_crop(landmarks, input_size):
  int_labels = tf.cast(landmarks, tf.int32)
  minimum = tf.reduce_min(int_labels, axis=1)
  maximum = tf.reduce_max(int_labels, axis=1)
  centers = (minimum + maximum) // 2
  half_sizes = tf.reduce_max(maximum - minimum, axis=1, keepdims=True) // 2
  low = tf.maximum(half_sizes - centers, 0)
  high = tf.maximum(half_sizes + centers - input_size, 0)
  shifts = tf.maximum(low, high)
  half_sizes -= tf.reduce_max(shifts, axis=1, keepdims=True)
  offsets = centers - half_sizes
  offsets = tf.cast(offsets, tf.float32)
  sizes = half_sizes * 2
  return offsets, sizes


def crop_images(images, landmarks, offsets, sizes, scales, target_size=None):
  if target_size is None:
    target_size = np.array(images.shape.as_list()[2:0:-1])

  def _crop_and_scale_image(input_):
    im, off, sz = input_
    im = tf.image.crop_to_bounding_box(im, off[1], off[0], sz[0], sz[0])
    im = tf.image.resize_images(im, target_size)
    return im

  int_offsets = tf.cast(offsets, tf.int32)
  images = tf.map_fn(
    _crop_and_scale_image, (images, int_offsets, sizes), dtype=tf.float32)

  scales = tf.cast(np.array([target_size]), tf.float32) / sizes

  landmarks -= tf.expand_dims(offsets, 1) 
  landmarks *= tf.expand_dims(scales, 1)
  return images, landmarks


def augment_image(image):
  """Augment the input with its mirrored image."""
  flipped = tf.reverse(image, axis=[-2])
  image = tf.concat([image, flipped], axis=-2)
  return image


def cutout_image(image, min_size, max_size):
  """Cutout part of the image."""
  assert(image.shape.ndims == 3)
  h, w = image.shape.as_list()[:2]
  s = tf.random_uniform([2], min_size, max_size, tf.int32)
  y = tf.random_uniform([], 0, h - s[0], tf.int32)
  x = tf.random_uniform([], 0, w - s[1], tf.int32)
  mask = tf.pad(tf.zeros([s[0], s[1], 3]), 
                [[y, h - s[0] - y], [x, w - s[1] - x], [0, 0]], 
                constant_values=1.0)
  masked_image = image * mask
  return masked_image


def image_noise(image, params):
  """Add image noise.
  
  args:
      image: Input 3D image tensor.
      params: list of triplets (scale, gaussian_std, salt_and_pepper)
  returns:
      Noisy image.
  """
  image = tf.convert_to_tensor(image)
  h, w, d = image.shape.as_list()
  for scale, std, snp in params:
    sh, sw = int(h * scale), int(w * scale)
    if std > 0:
      noise = tf.random_normal([sh, sw, d], stddev=std)
    else:
      noise = tf.zeros([sh, sw, d])
    if snp > 0:
      noise += tf.cast(
        tf.random_uniform([sh, sw, 1]) < snp * 0.5, tf.float32) * 2.0
      noise -= tf.cast(
        tf.random_uniform([sh, sw, 1]) < snp * 0.5, tf.float32) * 2.0
    noise = tf.image.resize_images(noise, [h, w])
    image += noise
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image
