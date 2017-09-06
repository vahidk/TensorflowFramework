"""Session run hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second."""

  def __init__(
      self,
      batch_size,
      every_n_iter=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook."""
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_iter'
                       ' and every_n_secs should be provided.')
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_iter, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    del run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        tf.logging.info('Examples/sec: %g (%g), step = %g',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)


class LoggingTensorHook(tf.train.SessionRunHook):
  """Hook to print batch of tensors."""

  def __init__(self, collection, every_n_iter=None, every_n_secs=None,
               batch=False, first_k=3):
    """Initializes a `LoggingTensorHook`."""
    self._collection = collection
    self._batch = batch
    self._first_k = first_k
    self._timer = tf.train.SecondOrStepTimer(
      every_secs=every_n_secs, every_steps=every_n_iter)

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      tensors = {t.name: t for t in tf.get_collection(self._collection)}
      return tf.train.SessionRunArgs(tensors)
    else:
      return None

  def _log_tensors(self, tensor_values):
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._batch:
      self._batch_print(tensor_values)
    else:
      self._print(tensor_values)

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      self._log_tensors(run_values.results)

    self._iter_count += 1

  def _print(self, tensor_values):
    if not tensor_values:
      return
    for k, v in tensor_values.items():
      tf.logging.info("%s: %s", k, np.array_str(v))

  def _batch_print(self, tensor_values):
    if not tensor_values:
      return
    batch_size = tensor_values.values()[0].shape[0]
    for i in range(min(self._first_k, batch_size)):
      for k, v in tensor_values.items():
        tf.logging.info("%s: %s", k, np.array_str(v[i]))