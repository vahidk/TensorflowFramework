"""Trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import matplotlib
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

from common import hooks
from common import hparams
from common import io as common_io
from common import ops
from common import optimizer
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
tf.flags.DEFINE_integer("num_reader_threads", 8, "Num reader threads.")
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


def experiment_fn(run_config, hparams):
  estimator = tf.estimator.Estimator(
    model_fn=optimizer.make_model_fn(MODELS[FLAGS.model].model, FLAGS.num_gpus), 
    config=run_config, params=hparams)
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
    train_input_fn=common_io.make_input_fn(
      DATASETS[FLAGS.dataset], tf.estimator.ModeKeys.TRAIN, hparams,
      num_epochs=FLAGS.num_epochs, 
      shuffle_batches=FLAGS.shuffle_batches,
      num_threads=FLAGS.num_reader_threads),
    eval_input_fn=common_io.make_input_fn(
      DATASETS[FLAGS.dataset], tf.estimator.ModeKeys.EVAL, hparams,
      num_epochs=FLAGS.num_epochs, 
      shuffle_batches=FLAGS.shuffle_batches,
      num_threads=FLAGS.num_reader_threads),
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
    hparams=hparams.get_params(
      MODELS[FLAGS.model], DATASETS[FLAGS.dataset], FLAGS.hparams))


if __name__ == "__main__":
  tf.app.run()
