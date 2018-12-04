"""Trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import numpy as np
import os
import tensorflow as tf

from common import dataset
from common import hooks
from common import hparams
from common import io_utils
from common import model
from common import ops
from common import optimizer

import dataset as _
import model as _

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string("model", "alexnet", "Model name.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset name.")
tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")
tf.flags.DEFINE_integer("num_epochs", None, "Number of epochs.")
tf.flags.DEFINE_integer("eval_secs", 10, "Number of seconds before running eval.")
tf.flags.DEFINE_integer("shuffle_batches", 500, "Shuffle batches.")
tf.flags.DEFINE_integer("num_reader_threads", 8, "Num reader threads.")
tf.flags.DEFINE_integer("save_summary_steps", 400, "Summary steps.")
tf.flags.DEFINE_integer("save_checkpoints_steps", 400, "Checkpoint steps.")
tf.flags.DEFINE_integer("num_gpus", 0, "Numner of gpus.")

FLAGS = tf.flags.FLAGS

MODEL = model.ModelFactory.create(FLAGS.model)
DATASET = dataset.DatasetFactory.create(FLAGS.dataset)


def main(unused_argv):
  if FLAGS.output_dir:
    model_dir = FLAGS.output_dir
  else:
    for i in range(10):
      model_dir = "output/%s/%s/%d" % (FLAGS.dataset, FLAGS.model, i)
      if not os.path.exists(model_dir):
        break

  DATASET.prepare()

  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True

  run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_summary_steps=FLAGS.save_summary_steps,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    save_checkpoints_secs=None,
    session_config=session_config)

  hp = hparams.get_params(MODEL, DATASET, FLAGS.hparams)

  estimator = tf.estimator.Estimator(
    model_fn=optimizer.make_model_fn(MODEL.model, FLAGS.num_gpus),
    config=run_config, 
    params=hp)
    
  train_spec = tf.estimator.TrainSpec(
    input_fn=io_utils.make_input_fn(
      DATASET, tf.estimator.ModeKeys.TRAIN, hp,
      num_epochs=FLAGS.num_epochs,
      shuffle_batches=FLAGS.shuffle_batches,
      num_threads=FLAGS.num_reader_threads), 
    hooks=[
      hooks.ExamplesPerSecondHook(
        batch_size=hp.batch_size,
        every_n_iter=FLAGS.save_summary_steps),
      hooks.LoggingTensorHook(
        collection="batch_logging",
        every_n_iter=FLAGS.save_summary_steps,
        batch=True),
      hooks.LoggingTensorHook(
        collection="logging",
        every_n_iter=FLAGS.save_summary_steps,
        batch=False)])

  eval_spec = tf.estimator.EvalSpec(
    input_fn=io_utils.make_input_fn(
      DATASET, tf.estimator.ModeKeys.EVAL, hp,
      num_epochs=FLAGS.num_epochs,
      shuffle_batches=FLAGS.shuffle_batches,
      num_threads=FLAGS.num_reader_threads),
    hooks=[
      hooks.SummarySaverHook(
        every_n_iter=FLAGS.save_summary_steps,
        output_dir=os.path.join(run_config.model_dir, "eval"))],
    throttle_secs=FLAGS.eval_secs)

  tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec,
    eval_spec=eval_spec)


if __name__ == "__main__":
  tf.app.run()
