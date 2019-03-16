"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import tensorflow as tf

from common import misc_utils


DEFAULT_CONFIG = {
  "mode": "train",
  "dataset": "mnist",
  "model": "alexnet",
  "experiment": "",
  "data_dir": "data",
  "output_dir": "output",
  "hparams": {},
  "num_epochs": 1000,
  "train_sets": "train",
  "eval_sets": ["eval"],
  "eval_secs": 30,
  "checkpoint_selector": {
    "eval_set": "eval",
    "eval_metric": "loss",
    "compare_fn": "less"
  },
  "shuffle_batches": 20,
  "num_reader_threads": 8,
  "prefetch_buffer_size": 1,
  "save_summary_steps": 400,
  "save_checkpoints_steps": 400,
  "num_gpus": 0,
  "gpu_id": None
}

DEFAULT_PARAMS = {
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "warmup_steps": 100,
  "constant_steps": 2500,
  "exponential_decay_steps": 10000,
  "exponential_decay_rate": 0.5,
  "cycle_decay_steps": 1000,
  "cycle_decay_rate": 0.1,
  "batch_size": 128,
  "weight_averaging_decay": 0.99
}


def get_config(path, str):
  cfg = DEFAULT_CONFIG

  if path:
    for k, v in json.load(open(path)).items():
      cfg[k] = v
  
  if str:
    for k, v in json.loads(str).items(): 
      cfg[k] = v
  
  obj = misc_utils.Tuple(cfg)

  return obj


def get_params(model, dataset, params):
  params_dict = DEFAULT_PARAMS
  params_dict.update(model.get_params())
  params_dict.update(dataset.get_params())
  params_dict.update(params.to_dict())
  hp = misc_utils.Tuple(params_dict)
  return hp


def get_model_dir(cfg):
  model_dir = "{0}/{1}/{2}/{3}".format(
    cfg.output_dir, cfg.dataset, cfg.model, cfg.experiment)
  return model_dir
