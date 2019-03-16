"""Activation ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common.ops import shape_ops


def leaky_relu(tensor, alpha=0.1):
  """Computes the leaky rectified linear activation."""
  return tf.maximum(tensor, alpha * tensor)


def activation_fn(activation):
  """Apply activation function."""
  return {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "crelu": tf.nn.crelu,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "leaky_relu": leaky_relu,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "sigmoid": tf.sigmoid,
    "tanh": tf.tanh
  }[activation]


def softmax(tensor, dims):
  """Computes softmax on arbitrary dimensions."""
  numerator = tf.exp(tensor - tf.reduce_max(tensor, dims, keepdims=True))
  denominator = tf.reduce_sum(numerator, dims, keepdims=True)
  return numerator / denominator


def log_softmax(tensor, dims):
  """Computes softmax on arbitrary dimensions."""
  denominator = tf.reduce_logsumexp(tensor, dims, keepdims=True)
  return tensor - denominator
