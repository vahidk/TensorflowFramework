"""Optimization ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf


def create_optimizer(params):
  return {
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "Ftrl": tf.train.FtrlOptimizer,
    "Momentum": lambda lr: tf.train.MomentumOptimizer(
      lr, params.momentum),
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
  }[params.optimizer](params.learning_rate)


def exponential_decay(learning_rate,
                      step,
                      decay_steps=20000,
                      decay_rate=0.5):
  """Exponential decay."""
  learning_rate *= decay_rate ** (step // decay_steps)
  return learning_rate


def cyclic_decay(learning_rate, 
                 step,
                 decay_steps=1000,
                 decay_rate=0.1):
  """Cyclic decay."""
  min_learning_rate = learning_rate * decay_rate
  cycle = tf.cos(tf.mod(step * np.pi / decay_steps, np.pi)) * 0.5 + 0.5
  learning_rate = ((learning_rate - min_learning_rate) * cycle + 
                    min_learning_rate)
  return learning_rate


def get_learning_rate(params):
  step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

  learning_rate = params.learning_rate

  if params.warmup_steps:
    learning_rate *= tf.minimum(1., (step + 1.0) / params.warmup_steps)
    step = tf.maximum(0., step - params.warmup_steps)

  if params.constant_steps:
    step = tf.maximum(0., step - params.constant_steps)

  if params.exponential_decay_rate < 1:
    learning_rate = exponential_decay(
      learning_rate=learning_rate, 
      step=step, 
      decay_steps=params.exponential_decay_steps, 
      decay_rate=params.exponential_decay_rate)
  
  if params.cycle_decay_rate < 1:
    learning_rate = cyclic_decay(
      learning_rate=learning_rate, 
      step=step,
      decay_steps=params.cycle_decay_steps,
      decay_rate=params.cycle_decay_rate)
  
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



def make_parallel(model_fn, features, labels, mode, params, num_gpus):
  with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
    split_features = {k: tf.split(v, num_gpus)
                      for k, v in features.items()}
    split_labels = {k: tf.split(v, num_gpus)
                    for k, v in labels.items()}

  predictions = collections.defaultdict(list)
  losses = []

  for i in range(num_gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
      with tf.name_scope("tower_%d" % i) as name_scope:
        with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
          device_features = {k: v[i] for k, v in split_features.items()}
          device_labels = {k:v[i] for k, v in split_labels.items()}

          device_predictions, device_loss, device_metrics = model_fn(
            device_features, device_labels, mode, params)

          if i == 0:
            eval_metrics = device_metrics
            update_ops = tf.get_collection(
              tf.GraphKeys.UPDATE_OPS, name_scope)

            reg_losses = tf.get_collection(
              tf.GraphKeys.REGULARIZATION_LOSSES, name_scope)

          for k, v in device_predictions.items():
            predictions[k].append(v)

          if device_loss is not None:
            losses.append(device_loss)

  for k, v in predictions.items():
    predictions[k] = tf.concat(v, axis=0)

  return predictions, losses, reg_losses, update_ops, eval_metrics


def make_model_fn(model_fn, process_fn, num_gpus=None, gpu_id=None,
                  weight_averaging_decay=None):
  """Build the model function."""
  def _model_fn_wpp(features, labels, mode, params):
    features, labels = process_fn(mode, params, features, labels)
    return model_fn(features, labels, mode, params)

  def _model_fn(features, labels, mode, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = get_learning_rate(params)
    tf.summary.scalar("learning_rate", learning_rate)

    opt = create_optimizer(params)

    if num_gpus:
      predictions, losses, reg_losses, update_ops, eval_metrics = make_parallel(
        _model_fn_wpp, features, labels, mode, params, num_gpus)
    else:
      if gpu_id is not None:
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
          predictions, loss, eval_metrics = _model_fn_wpp(features, labels, mode, params)
      else:
        predictions, loss, eval_metrics = _model_fn_wpp(features, labels, mode, params)
      losses = [loss]
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    loss = None

    if weight_averaging_decay is not None:
      ema = tf.train.ExponentialMovingAverage(
        decay=weight_averaging_decay)

    if losses:
      loss = tf.add_n(losses) / len(losses)
      tf.summary.scalar("loss/main", tf.add_n(losses))
    
      if reg_losses:
        loss += tf.add_n(reg_losses)
        tf.summary.scalar("loss/regularization", tf.add_n(reg_losses))

    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.control_dependencies(update_ops):
        train_op = opt.minimize(
          loss, 
          global_step=global_step,
          colocate_gradients_with_ops=True)

      if weight_averaging_decay is not None:
        with tf.control_dependencies([train_op]):
          train_op = ema.apply(tf.trainable_variables())

      opts = tf.profiler.ProfileOptionBuilder().trainable_variables_parameter()
      stats = tf.profiler.profile(tf.get_default_graph(), options=opts)
      print("Total parameters:", stats.total_parameters)
    else:
      train_op = None

      if weight_averaging_decay is not None:
        saver = tf.train.Saver(ema.variables_to_restore())
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics)

  return _model_fn
