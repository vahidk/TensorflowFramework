"""Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def top_k_error(labels, predictions, k, name=None):
  with tf.name_scope(name, default_name="top_k_error"):
    labels = tf.expand_dims(tf.to_int32(labels), axis=-1)
    _, top_k = tf.nn.top_k(predictions, k=k)
    in_top_k = tf.reduce_mean(tf.to_float(tf.equal(top_k, labels)), -1)
    return 1 - in_top_k
