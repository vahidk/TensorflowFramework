"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import metrics
from common import ops


def get_params():
  return {}


def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  training = mode == tf.estimator.ModeKeys.TRAIN

  features = ops.conv_layers(
    images, [16], [3], linear_top_layer=True)

  features = ops.resnet_blocks(
    features, [16, 32, 64], [1, 2, 2], 5, training)

  features = ops.batch_normalization(features, training=training)
  features = tf.nn.relu(features)

  features = tf.reduce_mean(features, axis=[1, 2])
  logits = ops.dense_layers(
    features, [params.num_classes], linear_top_layer=False)

  predictions = tf.argmax(logits, axis=-1)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  tf.summary.image("images", images)

  eval_metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions),
    "top_1_error": tf.metrics.mean(metrics.top_k_error(labels, logits, 1)),
  }

  return {"predictions": predictions}, loss, eval_metrics
