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
      every_n_secs=None):
    """Initializer for ExamplesPerSecondHook."""
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
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


class SummarySaverHook(tf.train.SessionRunHook):
  """Saves summaries every N steps."""

  def __init__(self,
               every_n_iter=None, 
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    """Initializes a `SummarySaverHook`."""
    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._timer = tf.train.SecondOrStepTimer(
      every_secs=every_n_iter, every_steps=every_n_secs)

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
    self._next_step = None
    self._global_step_tensor = tf.train.get_global_step()
    self._summaries = tf.summary.merge_all()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._request_summary = (
        self._next_step is None or
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    if self._request_summary and self._summaries is not None:
      requests["summary"] = self._summaries

    return tf.train.SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    _ = run_context
    if not self._summary_writer:
      return

    global_step = run_values.results["global_step"]

    if self._request_summary:
      self._timer.update_last_triggered_step(global_step)
      if "summary" in run_values.results:
        summary = run_values.results["summary"]
        self._summary_writer.add_summary(summary, global_step)

    self._next_step = global_step + 1

  def end(self, session=None):
    if self._summary_writer:
      self._summary_writer.flush()
