"""Layers ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numbers
import numpy as np
import tensorflow as tf
from common.ops import merge_ops
from common.ops import norm_ops


def dense_layers(tensor,
                 units,
                 activation=tf.nn.relu,
                 use_bias=True,
                 linear_top_layer=False,
                 drop_rates=None,
                 batch_norm=False,
                 training=False,
                 weight_decay=0.0002,
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  for i, (size, drp) in enumerate(zip(units, drop_rates)):
    if i == len(units) - 1 and linear_top_layer:
      activation = None
    with tf.variable_scope("dense_block_%d" % i):
      tensor = tf.layers.dropout(tensor, drp)
      tensor = tf.layers.dense(
        tensor, size, use_bias=use_bias,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        **kwargs)
      if activation:
        if batch_norm:
          tensor = norm_ops.batch_normalization(tensor, training=training)
        tensor = activation(tensor)
  return tensor


def conv_layers(tensor,
                filters,
                kernels,
                strides=None,
                pool_sizes=None,
                pool_strides=None,
                padding="same",
                activation=tf.nn.relu,
                use_bias=False,
                linear_top_layer=False,
                drop_rates=None,
                conv_method="conv",
                pool_method="conv",
                pool_activation=None,
                batch_norm=False,
                training=False,
                weight_decay=0.0002,
                **kwargs):
  """Builds a stack of convolutional layers with dropout and max pooling."""
  if pool_sizes is None:
    pool_sizes = [1] * len(filters)
  if pool_strides is None:
    pool_strides = pool_sizes
  if strides is None:
    strides = [1] * len(filters)
  if drop_rates is None:
    drop_rates = [0.] * len(filters)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(filters)

  if conv_method == "conv":
    conv = functools.partial(
      tf.layers.conv2d,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "transposed":
    conv = functools.partial(
      tf.layers.conv2d_transpose,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "separable":
    conv = functools.partial(
      tf.layers.separable_conv2d,
      depthwise_initializer=tf.glorot_uniform_initializer(),
      pointwise_initializer=tf.glorot_uniform_initializer(),
      depthwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      pointwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  for i, (fs, ks, ss, pz, pr, drp) in enumerate(
    zip(filters, kernels, strides, pool_sizes, pool_strides, drop_rates)):
    with tf.variable_scope("conv_block_%d" % i):
      if i == len(filters) - 1 and linear_top_layer:
        activation = None
        pool_activation = None
      tensor = tf.layers.dropout(tensor, drp)
      tensor = conv(
        tensor, fs, ks, ss, padding, use_bias=use_bias, name="conv2d",
        **kwargs)
      if activation:
        if batch_norm:
          tensor = norm_ops.batch_normalization(tensor, training=training)
        tensor = activation(tensor)
      if pz > 1:
        if pool_method == "max":
          tensor = tf.layers.max_pooling2d(
            tensor, pz, pr, padding, name="max_pool")
        elif pool_method == "std":
          tensor = tf.space_to_depth(tensor, pz, name="space_to_depth")
        elif pool_method == "dts":
          tensor = tf.depth_to_space(tensor, pz, name="depth_to_space")
        else:
          tensor = conv(
            tensor, fs, pz, pr, padding, use_bias=use_bias,
            name="strided_conv2d", **kwargs)
          if pool_activation:
            if batch_norm:
              tensor = norm_ops.batch_normalization(tensor, training=training)
            tensor = pool_activation(tensor)
  return tensor


def merge_layers(tensors, units, activation=tf.nn.relu,
                 linear_top_layer=False, drop_rates=None,
                 name=None, **kwargs):
  """Merge followed by a stack of dense layers."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  with tf.variable_scope(name, default_name="merge_layers"):
    result = tf.layers.dropout(result, drop_rates[0])
    result = merge_ops.merge(tensors, units[0], activation, **kwargs)
    result = dense_layers(result, units[1:],
                          activation=activation,
                          drop_rates=drop_rates[1:],
                          linear_top_layer=linear_top_layer,
                          **kwargs)
  return result


def squeeze_and_excite(tensor, ratio, name=None):
  """Apply squeeze/excite on given 4-D tensor.

  Based on: https://arxiv.org/abs/1709.01507
  """
  with tf.variable_scope(name, default_name="squeeze_and_excite"):
    original = tensor
    units = tensor.shape.as_list()[-1]
    tensor = tf.reduce_mean(tensor, [1, 2], keepdims=True)
    tensor = dense_layers(
      tensor, [units / ratio, units], use_bias=False, linear_top_layer=True)
    tensor = tf.nn.sigmoid(tensor)
    tensor = original * tensor
  return tensor


def resnet_block(tensor, filters, strides, training, weight_decay=0.0002,
                 kernel_size=3, activation=tf.nn.relu, drop_rate=0.0,
                 se_ratio=None):
  """Residual block."""
  original = tensor

  with tf.variable_scope("input"):
    tensor = norm_ops.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, strides, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  with tf.variable_scope("output"):
    tensor = tf.layers.dropout(tensor, drop_rate)
    tensor = norm_ops.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.layers.conv2d(
      tensor, filters, kernel_size, padding="same", use_bias=False,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    if se_ratio is not None:
      tensor = squeeze_and_excite(tensor, se_ratio)

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
