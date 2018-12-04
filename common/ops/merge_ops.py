"""Merge ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def merge(tensors, units, activation=tf.nn.relu, name=None, **kwargs):
  """Merge tensors with broadcasting support."""
  with tf.variable_scope(name, default_name="merge"):
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.layers.dense(
          tensor, units, name="proj_%d" % i, **kwargs)
      projs.append(proj)

    result = projs.pop()
    for proj in projs:
      result = result + proj

    if activation:
      result = activation(result)
  return result
