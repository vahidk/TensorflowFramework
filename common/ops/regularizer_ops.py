"""Beam-search ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_regularizer(scale=0.0, type="l2"):
  if scale < 0.0:
    raise ValueError("Invalid regularizer scale {}".format(scale))

  if scale == 0.0:
    return lambda _: None
  
  if type == "l1":
    return tf.keras.regularizers.l1(scale)
  elif type == "l2":
    return tf.keras.regularizers.l2(scale)
  elif type == "l1_l2":
    return tf.keras.regularizers.l1_l2(scale, scale)
  else:
    raise ValueError("Invalid regularier type {}".format(type))
