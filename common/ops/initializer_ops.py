"""Image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DiskInitializer(object):
  """Initializer that generates points within a circle."""

  def __init__(self, mean=0.0, radius=1.0, axis=-1, dtype=tf.float32):
    self.mean = mean
    self.radius = radius
    self.dtype = dtype
    self.axis = axis

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    h_shape = shape[:]
    del h_shape[self.axis]
    angle = tf.random_uniform(h_shape, 0, 2 * np.pi, dtype)
    squared_radius = tf.random_uniform(
      h_shape, 0, tf.square(self.radius), dtype)
    radius = tf.sqrt(squared_radius)

    x = tf.sin(angle) * radius
    y = tf.cos(angle) * radius
    output = tf.stack([x, y], axis=self.axis)
    if self.mean:
      output += mean
    return output

  def get_config(self):
    return {"mean": self.mean,
            "radius": self.radius,
            "dtype": self.dtype.name,
            "axis": self.axis}


disk_initializer = DiskInitializer
