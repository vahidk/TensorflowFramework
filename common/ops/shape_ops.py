"""Shape ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
import tensorflow as tf


def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def reshape(tensor, dims_list):
  """Reshape the given tensor by collapsing dimensions."""
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, numbers.Number):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.reduce_prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor


def expand_dims(tensor, dims):
  """Expand the rank of a tensor by inserting singular dimensions."""
  if isinstance(dims, numbers.Number):
    dims = [dims]
  for dim in dims:
    tensor = tf.expand_dims(tensor, dim)
  return tensor


def tile_like(tensor, like):
  """Tile a tensor to match another."""
  tensor = tf.tile(tensor, tf.shape(like) // tf.shape(tensor))
  return tensor
