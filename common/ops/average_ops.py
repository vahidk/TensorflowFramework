"""Average ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def moving_average(tensor, training=False, name=None):
  """Computes moving average of the given tensor."""
  with tf.variable_scope(name, default_name="moving_average"):
    sum_var = tf.get_variable(
      "value", shape=tensor.shape[1:], dtype=tf.float32,
      initializer=tf.zeros_initializer(),
      trainable=False)
    count_var = tf.get_variable("count", shape=[], dtype=tf.float32,
      initializer=tf.zeros_initializer(),
      trainable=False)

    sum = sum_var + tf.reduce_sum(tensor, axis=0)
    count = count_var + tf.to_float(tf.shape(tensor)[0])
    mean = sum / count

    if training:
      update_sum = tf.assign(sum_var, sum)
      update_count = tf.assign(count_var, count)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sum)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_count)

    return mean

