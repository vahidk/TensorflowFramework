"""Merge ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.ops import regularizer_ops


def merge(tensors, units, 
          activation=tf.nn.relu, 
          name=None, 
          weight_decay=0.0,
          weight_regularizer="l2",
          **kwargs):
  """Merge tensors with broadcasting support."""
  with tf.variable_scope(name, default_name="merge"):
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.keras.layers.Dense(
          units,
          use_bias=False,
          kernel_initializer=tf.glorot_uniform_initializer(),
          kernel_regularizer=regularizer_ops.weight_regularizer(
            weight_decay, weight_regularizer),
          name="proj_%d" % i,
          **kwargs).apply(tensor)
      projs.append(proj)

    result = projs.pop()
    for proj in projs:
      result = result + proj

    if activation:
      result = activation(result)
  return result
