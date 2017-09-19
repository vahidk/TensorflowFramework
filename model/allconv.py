"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import metrics
from common import ops

FLAGS = tf.flags.FLAGS

def get_params():
  return {
    "weight_decay": 0.0002,
    "input_drop_rate": 0.2,
    "drop_rate": 0.5
  }


def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN
  drop_rate = params.drop_rate if training else 0.0

  images = tf.layers.dropout(images, params.input_drop_rate)

  features = ops.conv_layers(
    images,
    filters=[96, 96, 192, 192, 192, 192, params.num_classes],
    kernels=[3, 3, 3, 3, 3, 1, 1],
    pool_sizes=[1, 3, 1, 3, 1, 1, 1],
    pool_strides=[1, 2, 1, 2, 1, 1, 1],
    drop_rates=[0, 0, drop_rate, 0, drop_rate, 0, 0],
    batch_norm=True,
    training=training,
    pool_activation=tf.nn.relu,
    linear_top_layer=True,
    weight_decay=params.weight_decay)

  logits = tf.reduce_mean(features, [1, 2])

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  tf.summary.image("images", images)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics
