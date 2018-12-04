"""Metric ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def interocular_ratio(labels, predictions, left_eye, right_eye, name=None):
  with tf.name_scope(name, default_name="interocular_ratio"):
    labels_t = tf.transpose(labels, [1, 0, 2])
    left_eye_pos = tf.reduce_mean(tf.gather(labels_t, left_eye), axis=0)
    right_eye_pos = tf.reduce_mean(tf.gather(labels_t, right_eye), axis=0)
    landmarks_l2 = tf.norm(predictions - labels, axis=2)
    eyes_l2 = tf.norm(left_eye_pos - right_eye_pos, axis=1)
    ratio = landmarks_l2 / tf.expand_dims(eyes_l2, axis=1)
    return ratio


def top_k_error(labels, predictions, k, name=None):
  with tf.name_scope(name, default_name="top_k_error"):
    labels = tf.expand_dims(tf.to_int32(labels), axis=-1)
    _, top_k = tf.nn.top_k(predictions, k=k)
    in_top_k = tf.reduce_mean(tf.to_float(tf.equal(top_k, labels)), -1)
    return 1 - in_top_k
