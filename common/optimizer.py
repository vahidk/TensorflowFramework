"""Optimization ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def exponential_decay(learning_rate, 
                      warmup_steps=100, 
                      constant_steps=20000,
                      decay_steps=20000,
                      decay_rate=0.5):
  """Exponential decay."""
  step = tf.to_float(tf.train.get_or_create_global_step())

  if warmup_steps:
    learning_rate *= tf.minimum(1., (step + 1.0) / warmup_steps)
    step = tf.maximum(0., step - warmup_steps)

  if constant_steps:
    step = tf.maximum(0., step - constant_steps)

  if decay_steps:
    learning_rate *= decay_rate ** (step // decay_steps)

  return learning_rate


def cyclic_decay(learning_rate, 
                 min_learning_rate=1e-4,
                 cycle_length=1000,
                 decay_steps=20000,
                 decay_rate=0.5):
  """Cyclic learning rate."""
  step = tf.to_float(tf.train.get_or_create_global_step())

  decay = decay_rate ** (step // decay_steps)
  min_learning_rate = min_learning_rate * decay
  max_learning_rate = learning_rate * decay

  cycle = tf.sin(step * 2 * 3.141592 / cycle_length)
  learning_rate = ((max_learning_rate - min_learning_rate) * (cycle + 1) * 0.5 + 
                    min_learning_rate)

  return learning_rate


def average_gradients(tower_grads):
  """Compute average gradients."""
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = [g for g, _ in grad_and_vars]
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)
    v = grad_and_vars[0][1]
    average_grads.append((grad, v))
  return average_grads


def make_model_fn(model_fn, num_gpus=None):
  def _model_fn(features, labels, mode, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = exponential_decay(
      params.learning_rate, params.warmup_steps, params.constant_steps,
      params.decay_steps, params.decay_rate)
    tf.summary.scalar("learning_rate", learning_rate)

    opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[params.optimizer](
      learning_rate, params.momentum)

    if num_gpus:
      split_features = {k: tf.split(v, num_gpus)
                        for k, v in features.iteritems()}
      split_labels = {k: tf.split(v, num_gpus)
                      for k, v in labels.iteritems()}
      grads = []
      predictions = collections.defaultdict(list)
      losses = []

      for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
          with tf.name_scope("tower_%d" % i) as name_scope:
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
              device_features = {k: v[i] for k, v in split_features.iteritems()}
              device_labels = {k:v[i] for k, v in split_labels.iteritems()}

              device_predictions, device_loss, device_metrics = model_fn(
                device_features, device_labels, mode, params)
              tf.summary.scalar("loss/main", device_loss)

              if i == 0:
                eval_metrics = device_metrics
                update_ops = tf.get_collection(
                  tf.GraphKeys.UPDATE_OPS, name_scope)

                reg_losses = tf.get_collection(
                  tf.GraphKeys.REGULARIZATION_LOSSES, name_scope)
                tf.summary.scalar("loss/regularization", tf.add_n(reg_losses))

                device_loss = tf.add_n([device_loss] + reg_losses)

              for k, v in device_predictions.iteritems():
                predictions[k].append(v)

              if device_loss is not None:
                losses.append(device_loss)

              if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.control_dependencies(update_ops):
                  device_grads = opt.compute_gradients(device_loss)
                grads.append(device_grads)

      with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        if mode == tf.estimator.ModeKeys.TRAIN:
          grads = average_gradients(grads)
          train_op = opt.apply_gradients(grads, global_step=global_step)
        else:
          train_op = None

        for k, v in predictions.iteritems():
          predictions[k] = tf.concat(v, axis=0)

        loss = tf.add_n(losses) if losses else None
    else:
      predictions, loss, eval_metrics = model_fn(features, labels, mode, params)
      tf.summary.scalar("loss/main", loss)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      tf.summary.scalar("loss/regularization", tf.add_n(reg_losses))

      loss = tf.add_n([loss] + reg_losses)

      if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.control_dependencies(update_ops):
          train_op = opt.minimize(loss, global_step=global_step)
      else:
        train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      opts = tf.profiler.ProfileOptionBuilder().trainable_variables_parameter()
      stats = tf.profiler.profile(tf.get_default_graph(), options=opts)
      print("Total parameters:", stats.total_parameters)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)

  return _model_fn
