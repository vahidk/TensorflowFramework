"""Normalization ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batch_normalization(tensor, training=False, epsilon=0.001, momentum=0.9,
                        fused_batch_norm=False, name=None):
  """Performs batch normalization on given 4-D tensor.

  The features are assumed to be in NHWC format. Noe that you need to
  run UPDATE_OPS in order for this function to perform correctly, e.g.:

  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(loss)

  Based on: https://arxiv.org/abs/1502.03167
  """
  with tf.variable_scope(name, default_name="batch_normalization"):
    channels = tensor.shape.as_list()[-1]
    axes = list(range(tensor.shape.ndims - 1))

    beta = tf.get_variable(
      "beta", channels, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
      "gamma", channels, initializer=tf.ones_initializer())

    avg_mean = tf.get_variable(
      "avg_mean", channels, initializer=tf.zeros_initializer(),
      trainable=False)
    avg_variance = tf.get_variable(
      "avg_variance", channels, initializer=tf.ones_initializer(),
      trainable=False)

    if training:
      if fused_batch_norm:
        mean, variance = None, None
      else:
        mean, variance = tf.nn.moments(tensor, axes=axes)
    else:
      mean, variance = avg_mean, avg_variance

    if fused_batch_norm:
      tensor, mean, variance = tf.nn.fused_batch_norm(
        tensor, scale=gamma, offset=beta, mean=mean, variance=variance,
        epsilon=epsilon, is_training=training)
    else:
      tensor = tf.nn.batch_normalization(
        tensor, mean, variance, beta, gamma, epsilon)

    if training:
      update_mean = tf.assign(
        avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
      update_variance = tf.assign(
        avg_variance, avg_variance * momentum + variance * (1.0 - momentum))

      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

  return tensor


def squeeze_and_excite(tensor, ratio, name=None):
  """Apply squeeze/excite on given 4-D tensor.

  Based on: https://arxiv.org/abs/1709.01507
  """
  with tf.variable_scope(name, default_name="squeeze_and_excite"):
    original = tensor
    units = tensor.shape.as_list()[-1]
    tensor = tf.reduce_mean(tensor, [1, 2], keep_dims=True)
    tensor = dense_layers(
      tensor, [units / ratio, units], use_bias=False, linear_top_layer=True)
    tensor = tf.nn.sigmoid(tensor)
    tensor = original * tensor
  return tensor
