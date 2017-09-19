"""Resnet ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
import tensorflow as tf
from common import ops


def resnet_block(tensor, filters, strides, training, weight_decay=0.0002,
                 kernel_size=3, activation=tf.nn.relu, drop_rate=0.0,
                 se_ratio=None):
  """Residual block."""
  original = tensor

  with tf.variable_scope("input"):
    tensor = ops.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, strides, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  with tf.variable_scope("output"):
    tensor = tf.layers.dropout(tensor, drop_rate)
    tensor = ops.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    if se_ratio is not None:
      tensor = ops.squeeze_and_excite(tensor, se_ratio)

    in_dims = original.shape[-1].value
    if in_dims != filters or strides > 1:
      diff = filters - in_dims
      original = tf.layers.average_pooling2d(original, strides, strides)
      original = tf.pad(original, [[0, 0], [0, 0], [0, 0], [0, diff]])

    tensor += original

  return tensor


def resnet_blocks(tensor, filters, strides, sub_layers, training, drop_rates,
                  **kwargs):
  if drop_rates is None:
    drop_rates = [0.] * len(filters)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(filters)

  for i, (filter, stride, drp) in enumerate(zip(filters, strides, drop_rates)):
    with tf.variable_scope("group_%d" % i):
      for j in range(sub_layers):
        with tf.variable_scope("block_%d" % j):
          stride = stride if j == 0 else 1
          tensor = resnet_block(
            tensor, filter, stride, training, drop_rate=drp, **kwargs)
  return tensor

