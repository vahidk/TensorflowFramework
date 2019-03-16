"""Image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.ops import shape_ops


def dropout(tensor, drop_rate, training, type="regular"):
  if type == "regular":
    noise_shape = None
  elif type == "spatial":
    shape = shape_ops.get_shape(tensor)
    assert len(shape) == 4
    noise_shape = [shape[0], 1, 1, shape[3]]
  else:
    raise ValueError("Invalid dropout type {}".format(type))

  dropped = tf.nn.dropout(
    tensor, 
    rate=drop_rate, 
    noise_shape=noise_shape)
  
  output = dropped if training else tensor
  return output
