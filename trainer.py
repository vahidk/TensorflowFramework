"""Trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

from common import hooks
from common import ops
from common import metrics

import dataset.cifar10
import dataset.cifar100
import dataset.mnist

import model.alexnet
import model.allconv
import model.resnet

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string("model", "alexnet", "Model name.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset name.")
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training epochs.")
tf.flags.DEFINE_integer("shuffle_batches", 500, "Shuffle batches.")
tf.flags.DEFINE_integer("save_summary_steps", 400, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 400, "Checkpoint steps.")
tf.flags.DEFINE_integer("eval_frequency", 1, "Eval frequency.")
tf.flags.DEFINE_integer("num_gpus", 0, "Numner of gpus.")

FLAGS = tf.flags.FLAGS

MODELS = {
  "alexnet": model.alexnet,
  "allconv": model.allconv,
  "resnet": model.resnet
}

DATASETS = {
  "cifar10": dataset.cifar10,
  "cifar100": dataset.cifar100,
  "mnist": dataset.mnist,
}

HPARAMS = {
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "momentum": 0.9,
  "warmup_steps": 1000,
  "decay_steps": 10000,
  "decay_rate": 0.5,
  "batch_size": 128,
}


def get_params():
  hparams = HPARAMS
  hparams.update(DATASETS[FLAGS.dataset].get_params())
  hparams.update(MODELS[FLAGS.model].get_params())

  hparams = tf.contrib.training.HParams(**hparams)
  hparams.parse(FLAGS.hparams)

  return hparams


def make_input_fn(mode, params):
  def _parse(*args):
    return DATASETS[FLAGS.dataset].parse(mode, *args)
  def _input_fn():
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
      dataset = DATASETS[FLAGS.dataset].read(mode)
      dataset = dataset.cache()
      if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(FLAGS.num_epochs)
        dataset = dataset.shuffle(params.batch_size * FLAGS.shuffle_batches)
      dataset = dataset.map(_parse, num_threads=8)
      dataset = dataset.batch(params.batch_size)
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
    return features, labels
  return _input_fn


def make_model_fn():
  def _model_fn(features, labels, mode, params):
    model_fn = MODELS[FLAGS.model].model

    global_step = tf.train.get_or_create_global_step()

    opt = ops.create_optimizer(
      params.optimizer, params.learning_rate, params.momentum,
      params.warmup_steps, params.decay_steps, params.decay_rate)

    if FLAGS.num_gpus:
      split_features = {k: tf.split(v, FLAGS.num_gpus)
                        for k, v in features.iteritems()}
      split_labels = {k: tf.split(v, FLAGS.num_gpus)
                      for k, v in labels.iteritems()}
      grads = []
      predictions = collections.defaultdict(list)
      losses = []

      for i in range(FLAGS.num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
          with tf.name_scope("tower_%d" % i) as name_scope:
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
              device_features = {k: v[i] for k, v in split_features.iteritems()}
              device_labels = {k:v[i] for k, v in split_labels.iteritems()}

              device_predictions, device_loss, device_metrics = model_fn(
                device_features, device_labels, mode, params)
              tf.summary.scalar("loss/main", device_loss)

              if i == 0:
                metrics = device_metrics
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
          grads = ops.average_gradients(grads)
          train_op = opt.apply_gradients(grads, global_step=global_step)
        else:
          train_op = None

        for k, v in predictions.iteritems():
          predictions[k] = tf.concat(v, axis=0)

        loss = tf.add_n(losses) if losses else None
    else:
      predictions, loss, metrics = model_fn(features, labels, mode, params)
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
      eval_metric_ops=metrics)

  return _model_fn


def experiment_fn(run_config, hparams):
  estimator = tf.estimator.Estimator(
    model_fn=make_model_fn(), config=run_config, params=hparams)
  train_hooks = [
    hooks.ExamplesPerSecondHook(
      batch_size=hparams.batch_size, 
      every_n_iter=FLAGS.save_summary_steps),
    hooks.LoggingTensorHook(
      collection="batch_logging",
      every_n_iter=FLAGS.save_summary_steps,
      batch=True),
    hooks.LoggingTensorHook(
      collection="logging",
      every_n_iter=FLAGS.save_summary_steps,
      batch=False)]
  eval_hooks = [
    hooks.SummarySaverHook(
      every_n_iter=FLAGS.save_summary_steps,
      output_dir=os.path.join(run_config.model_dir, "eval"))]
  experiment = tf.contrib.learn.Experiment(
    estimator=estimator,
    train_input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, hparams),
    eval_input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, hparams),
    eval_steps=None,
    min_eval_frequency=FLAGS.eval_frequency,
    eval_hooks=eval_hooks)
  experiment.extend_train_hooks(train_hooks)
  return experiment


def main(unused_argv):
  if FLAGS.output_dir:
    model_dir = FLAGS.output_dir
  else:
    for i in range(10):
      model_dir = "output/%s/%s/%d" % (FLAGS.dataset, FLAGS.model, i)
      if not os.path.exists(model_dir):
        break

  DATASETS[FLAGS.dataset].prepare()

  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True
  run_config = tf.contrib.learn.RunConfig(
    model_dir=model_dir,
    save_summary_steps=FLAGS.save_summary_steps,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    save_checkpoints_secs=None,
    session_config=session_config)

  estimator = tf.contrib.learn.learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule=FLAGS.schedule,
    hparams=get_params())


if __name__ == "__main__":
  tf.app.run()
