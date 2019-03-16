"""Average ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MovingAverage(tf.keras.layers.Layer):
  """Computes moving average of the given tensor."""
  def __init__(self, **kwargs):
    super(MovingAverage, self).__init__(**kwargs)
   
  def build(self, input_shape):
    self.sum_var = self.add_variable(
      name="value", shape=input_shape[1:], dtype=tf.float32,
      initializer=tf.zeros_initializer(),
      trainable=False)
    self.count_var = self.add_variable(
      name="count", shape=input_shape[1:], dtype=tf.float32,
      initializer=tf.zeros_initializer(),
      trainable=False)
    super(MovingAverage, self).build(input_shape)

  def call(self, tensor, weights=None, training=False):
    if weights is None:
      sum = self.sum_var + tf.reduce_sum(tensor, axis=0)
      count = self.count_var + tf.cast(tf.shape(tensor)[0], tf.float32)
    else:
      sum = self.sum_var + tf.reduce_sum(tensor * weights, axis=0)
      count = self.count_var + tf.reduce_sum(weights, axis=0)      

    if training:
      self.add_update(tf.assign(self.sum_var, sum))
      self.add_update(tf.assign(self.count_var, count))
      with tf.control_dependencies(self.updates):
        mean = self.sum_var / self.count_var
    else:
      mean = self.sum_var / self.count_var

    return mean
