"""Math ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lerp(first, second, ratio):
  """Interpolate between two tensors."""
  return first * (1 - ratio) + second * ratio
