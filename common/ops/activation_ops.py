"""Activation ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def leaky_relu(tensor, alpha=0.1):
  """Computes the leaky rectified linear activation."""
  return tf.maximum(tensor, alpha * tensor)
