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
    "drop_rate": 0.5
  }


def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN
  drop_rate = params.drop_rate if training else 0.0

  features = ops.conv_layers(
    images,
    filters=[64, 128, 256],
    kernels=[3, 3, 3],
    pool_sizes=[2, 2, 2])

  features = tf.contrib.layers.flatten(features)

  logits = ops.dense_layers(
    features, [512, params.num_classes],
    drop_rates=drop_rate,
    linear_top_layer=True)

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  tf.summary.image("images", images)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  tf.add_to_collection(
    "batch_logging", tf.identity(labels, name="labels"))
  tf.add_to_collection(
    "batch_logging", tf.identity(predictions, name="predictions"))

  return {"predictions": predictions}, loss, eval_metrics
