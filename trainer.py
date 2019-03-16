"""Trainer module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import tensorflow as tf

from common import config
from common import dataset
from common import hooks
from common import io_utils
from common import misc_utils
from common import model
from common import ops
from common import optimizer

import dataset as _
import model as _

tf.flags.DEFINE_string("config", "", "Path of the configuration file.")
tf.flags.DEFINE_string("override", "", "Configuration override string.")
tf.flags.DEFINE_string("results", "", "Path of the results file.")
tf.flags.DEFINE_bool("overwrite_params", False, "Overwrite hypter parameters.")

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  cfg = config.get_config(FLAGS.config, FLAGS.override)
  print("Configuration loaded: ")
  print(cfg)

  if not cfg.experiment:
    if FLAGS.config:
      cfg.experiment = os.path.splitext(os.path.basename(FLAGS.config))[0]
    else:
      cfg.experiment = "default"

  model_dir = config.get_model_dir(cfg)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  
  tf.logging.set_verbosity(tf.logging.INFO)

  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True  # pylint: ignore

  run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_summary_steps=cfg.save_summary_steps,
    save_checkpoints_steps=cfg.save_checkpoints_steps,
    save_checkpoints_secs=None,
    session_config=session_config)
  
  m = model.ModelFactory.create(cfg.model)

  d = dataset.DatasetFactory.create(cfg.dataset)

  hp = config.get_params(m, d, cfg.hparams)

  d.prepare(hp)

  estimator = tf.estimator.Estimator(
    model_fn=optimizer.make_model_fn(
      m.model, d.process, cfg.num_gpus, cfg.gpu_id, 
      hp.weight_averaging_decay),
    config=run_config, 
    params=hp)

  def eval_input_fn(eval_set):  
    return io_utils.make_input_fn(
      d, eval_set, tf.estimator.ModeKeys.EVAL, hp,
      num_epochs=1,
      num_threads=cfg.num_reader_threads,
      prefetch_buffer_size=cfg.prefetch_buffer_size)

  def _predict():
    results = {}
    for eval_set in cfg.eval_sets:
      result_iterator = estimator.predict(
        input_fn=eval_input_fn(eval_set))
      result = {}
      for item in result_iterator:
        for k, v in item.items():
          result.setdefault(k, []).append(np.array(v).tolist())
      results[eval_set] = result
    return results

  def _eval():
    results = {}
    for eval_set in cfg.eval_sets:
      metrics = estimator.evaluate(
        input_fn=eval_input_fn(eval_set),
        hooks=[
          hooks.SummarySaverHook(
            every_n_iter=cfg.save_summary_steps,
            output_dir=os.path.join(run_config.model_dir, "eval_" + eval_set))],
        name=eval_set)
      results[eval_set] = metrics
      print(metrics)
    return results

  def _train():
    params_path = os.path.join(model_dir, "params.json")
    if os.path.exists(params_path) and not FLAGS.overwrite_params:
      with open(params_path, "r") as fp:
        if not fp.read() == str(hp):
          raise RuntimeError("Mismatching parameters found.")
    else:
      with open(params_path, "w") as fp:
        fp.write(str(hp))

    train_sets = (
      cfg.train_sets.to_dict() 
      if isinstance(cfg.train_sets, misc_utils.Tuple) 
      else cfg.train_sets)      
    estimator.train(
      input_fn=io_utils.make_input_fn(
        d, train_sets, 
        tf.estimator.ModeKeys.TRAIN, hp,
        num_epochs=cfg.num_epochs,
        shuffle_batches=cfg.shuffle_batches,
        num_threads=cfg.num_reader_threads,
        prefetch_buffer_size=cfg.prefetch_buffer_size),
      hooks=[
        hooks.ExamplesPerSecondHook(
          batch_size=hp.batch_size,
          every_n_iter=cfg.save_summary_steps),
        hooks.LoggingTensorHook(
          collection="batch_logging",
          every_n_iter=cfg.save_summary_steps,
          batch=True),
        hooks.LoggingTensorHook(
          collection="logging",
          every_n_iter=cfg.save_summary_steps,
          batch=False),
        tf.train.CheckpointSaverHook(
          model_dir,
          save_steps=cfg.save_checkpoints_steps,
          listeners=[
            hooks.BestCheckpointKeeper(
              model_dir,
              eval_fn=_eval,
              eval_set=cfg.checkpoint_selector.eval_set, 
              eval_metric=cfg.checkpoint_selector.eval_metric, 
              compare_fn=cfg.checkpoint_selector.compare_fn)])])

  def _export():
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        m.get_features(hp))

    estimator.export_saved_model(
      os.path.join(model_dir, "export"), 
      serving_input_fn)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
      features = serving_input_fn().features
      predictions = m.model(features, None, tf.estimator.ModeKeys.PREDICT, hp)[0]
      print("Features", {k: v.name for k, v in features.items()})
      print("Predictions", {k: v.name for k, v in predictions.items()})
      tf.train.write_graph(sess.graph_def, model_dir, 'graph_eval.pbtxt')
 
  if cfg.mode == "train":
    _train()
    if FLAGS.results:
      results = _eval()
      with open(FLAGS.results, "w") as f:
        f.write(misc_utils.serialize_json(results))
  elif cfg.mode == "eval":
    results = _eval()
    if FLAGS.results:
      with open(FLAGS.results, "w") as f:
        f.write(misc_utils.serialize_json(results))
  elif cfg.mode == "predict":
    results = _predict()
    if FLAGS.results:
      with open(FLAGS.results, "w") as f:
        f.write(misc_utils.serialize_json(results))
  elif cfg.mode == "export":
    _export()
  else:
    print("Unrecognized mode", cfg.mode)
  print("Done.")


if __name__ == "__main__":
  tf.app.run()
