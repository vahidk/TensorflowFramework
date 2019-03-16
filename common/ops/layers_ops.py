"""Layers ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numbers
import numpy as np
import tensorflow as tf
from common.ops import merge_ops
from common.ops import noise_ops
from common.ops import regularizer_ops


def _to_array(value, size, default_val=None):
  if value is None:
    value = [default_val] * size
  elif not isinstance(value, list):
    value = [value] * size
  return value


def _merge_dicts(dict, *others):
  dict = dict.copy()
  for other in others:
    dict.update(other)
  return dict


def dense_layers(tensor,
                 units,
                 activation=tf.nn.relu,
                 linear_top_layer=False,
                 drop_rates=None,
                 drop_type="regular",
                 batch_norm=False,
                 training=False,
                 weight_decay=0.0,
                 weight_regularizer="l2",
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  drop_rates = _to_array(drop_rates, len(units), 0.)
  kernel_initializer = tf.glorot_uniform_initializer()
  kernel_regularizer = regularizer_ops.weight_regularizer(
    weight_decay, weight_regularizer)
  for i, (size, drp) in enumerate(zip(units, drop_rates)):
    if i == len(units) - 1 and linear_top_layer:
      activation = None
    with tf.variable_scope("dense_block_%d" % i):
      tensor = noise_ops.dropout(tensor, drp, training=training, type=drop_type)
      tensor = tf.keras.layers.Dense(
        size, 
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        **kwargs).apply(tensor)
      if activation:
        if batch_norm:
          tensor = tf.layers.batch_normalization(tensor, training=training)
        tensor = activation(tensor)
  return tensor


def merge_layers(tensors, 
                 units, 
                 activation=tf.nn.relu,
                 linear_top_layer=False, 
                 drop_rates=None,
                 training=False, 
                 name=None, 
                 **kwargs):
  """Merge followed by a stack of dense layers."""
  drop_rates = _to_array(drop_rates, len(units), 0.)
  with tf.variable_scope(name, default_name="merge_layers"):
    tensors = noise_ops.dropout(tensors, drop_rates[0], training=training)
    tensors = merge_ops.merge(tensors, units[0], 
                              activation=activation,
                              **kwargs)
    tensors = dense_layers(tensors, units[1:],
                          activation=activation,
                          drop_rates=drop_rates[1:],
                          linear_top_layer=linear_top_layer,
                          training=training,
                          **kwargs)
  return tensors


def squeeze_and_excite(tensor, ratio, name=None):
  """Apply squeeze/excite on given 4-D tensor.

  Based on: https://arxiv.org/abs/1709.01507
  """
  with tf.variable_scope(name, default_name="squeeze_and_excite"):
    original = tensor
    units = tensor.shape.as_list()[-1]
    tensor = tf.reduce_mean(tensor, [1, 2], keepdims=True)
    tensor = dense_layers(
      tensor, [units // ratio, units], linear_top_layer=True)
    tensor = tf.nn.sigmoid(tensor)
    tensor = original * tensor
  return tensor


def AsymmetricConv2D(filters, kernel_size, **kwargs):
  name = kwargs.pop("name", "AsymmetricConv2D")
  activation = kwargs.pop("activation", None)
  layer = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters, [kernel_size, 1], 
      name="horizontal", **kwargs),
    tf.keras.layers.Conv2D(filters, [1, kernel_size], 
      name="vertical", activation=activation, **kwargs)
  ], name=name)
  return layer


def conv_layers(tensor,
                filters,
                kernels,
                strides=None,
                pool_sizes=None,
                pool_strides=None,
                padding="same",
                activation=tf.nn.relu,
                linear_top_layer=False,
                drop_rates=None,
                drop_type="regular",
                conv_method="conv",
                pool_method="conv",
                pool_activation=None,
                dilations=None,
                batch_norm=False,
                training=False,
                weight_decay=0.0,
                weight_regularizer="l2",
                **kwargs):
  """Builds a stack of convolutional layers with dropout and max pooling."""
  if not filters:
    return tensor

  kernels = _to_array(kernels, len(filters), 1)
  pool_sizes = _to_array(pool_sizes, len(filters), 1)  
  pool_strides = _to_array(pool_strides, len(filters), 1)  
  strides = _to_array(strides, len(filters), 1)
  drop_rates = _to_array(drop_rates, len(filters), 0.)
  dilations = _to_array(dilations, len(filters), 1)
  conv_method = _to_array(conv_method, len(filters), "conv")
  pool_method = _to_array(pool_method, len(filters), "conv")

  kernel_initializer = tf.glorot_uniform_initializer()
  kernel_regularizer = regularizer_ops.weight_regularizer(
    weight_decay, weight_regularizer)

  conv = {
    "conv": functools.partial(
      tf.keras.layers.Conv2D,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer),
    "transposed": functools.partial(
      tf.keras.layers.Conv2DTranspose,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer),
    "separable": functools.partial(
      tf.keras.layers.SeparableConv2D,
      depthwise_initializer=kernel_initializer,
      pointwise_initializer=kernel_initializer,
      depthwise_regularizer=kernel_regularizer,
      pointwise_regularizer=kernel_regularizer),
    "asymmetric": functools.partial(
      AsymmetricConv2D,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer),
  }

  for i, (fs, ks, ss, pz, pr, drp, dl, cm, pm) in enumerate(
    zip(filters, kernels, strides, pool_sizes, pool_strides, 
    drop_rates, dilations, conv_method, pool_method)):

    with tf.variable_scope("conv_block_%d" % i):
      if i == len(filters) - 1 and linear_top_layer:
        activation = None
        pool_activation = None
      tensor = noise_ops.dropout(tensor, drp, training=training, type=drop_type)
      if dl > 1:
        conv_kwargs = _merge_dicts(kwargs, {"dilation_rate": dl})
      else:
        conv_kwargs = kwargs
      tensor = conv[cm](
        filters=fs, 
        kernel_size=ks, 
        strides=ss, 
        padding=padding, 
        use_bias=False, 
        name="conv2d",
        **conv_kwargs).apply(tensor)
      if activation:
        if batch_norm:
          tensor = tf.layers.batch_normalization(tensor, training=training)
        tensor = activation(tensor)
      if pz > 1:
        if pm == "max":
          tensor = tf.keras.layers.MaxPool2D(
            pz, pr, padding, name="max_pool").apply(tensor)
        elif pm == "std":
          tensor = tf.space_to_depth(tensor, pz, name="space_to_depth")
        elif pm == "dts":
          tensor = tf.depth_to_space(tensor, pz, name="depth_to_space")
        else:
          tensor = conv["conv"](
            fs, pz, pr, padding, use_bias=False,
            name="strided_conv2d", **kwargs).apply(tensor)
          if pool_activation:
            if batch_norm:
              tensor = tf.layers.batch_normalization(tensor, training=training)
            tensor = pool_activation(tensor)
  return tensor


def resnet_block(tensor, 
                 filters, 
                 strides, 
                 training, 
                 weight_decay=0.0,
                 weight_regularizer="l2",
                 kernel_size=3, 
                 activation=tf.nn.relu, 
                 drop_rate=0.0, 
                 drop_type="regular",
                 se_ratio=None):
  """Residual block."""
  original = tensor

  kernel_initializer = tf.glorot_uniform_initializer()
  kernel_regularizer = regularizer_ops.weight_regularizer(
    weight_decay, weight_regularizer)

  with tf.variable_scope("input"):
    tensor = tf.layers.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.keras.layers.Conv2D(
      filters, kernel_size, strides, padding="same", use_bias=False,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer).apply(tensor)

  with tf.variable_scope("output"):
    tensor = noise_ops.dropout(
      tensor, drop_rate, training=training, type=drop_type)
    tensor = tf.layers.batch_normalization(tensor, training=training)
    tensor = activation(tensor)
    tensor = tf.keras.layers.Conv2D(
      filters, kernel_size, padding="same", use_bias=False,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer).apply(tensor)

    if se_ratio is not None:
      tensor = squeeze_and_excite(tensor, se_ratio)

    in_dims = original.shape[-1].value
    if in_dims != filters or strides > 1:
      diff = filters - in_dims
      original = tf.keras.layers.AvgPool2D(strides, strides).apply(original)
      original = tf.pad(original, [[0, 0], [0, 0], [0, 0], [0, diff]])

    tensor += original

  return tensor


def resnet_blocks(tensor, filters, strides, sub_layers, training, drop_rates,
                  **kwargs):
  strides = _to_array(strides, len(filters), 1)
  drop_rates = _to_array(drop_rates, len(filters), 0.)

  for i, (filter, stride, drp) in enumerate(zip(filters, strides, drop_rates)):
    with tf.variable_scope("group_%d" % i):
      for j in range(sub_layers):
        with tf.variable_scope("block_%d" % j):
          stride = stride if j == 0 else 1
          tensor = resnet_block(
            tensor, filter, stride, training, drop_rate=drp, **kwargs)
  return tensor
