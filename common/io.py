"""I/O utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import tensorflow as tf


def parallel_record_writer(iterator, create_example, path, num_threads=6):
  """Create a RecordIO file from data for efficient reading."""
  if num_threads == 1:
    writer = tf.python_io.TFRecordWriter(path)
    for item in iterator:
      example = create_example(item)
      writer.write(example.SerializeToString())
    writer.close()
    return

  def _queue(inputs):
    for item in iterator:
      inputs.put(item)
    for _ in range(num_threads):
      inputs.put(None)

  def _map_fn(inputs, outputs):
    while True:
      item = inputs.get()
      if item is None:
        break
      example = create_example(item)
      outputs.put(example)
    outputs.put(None)

  # Read the inputs.
  inputs = mp.Queue()
  process = mp.Process(target=_queue, args=(inputs,))
  process.daemon = True
  process.start()

  # Convert to tf.Example
  outputs = mp.Queue()
  for _ in range(num_threads):
    process = mp.Process(target=_map_fn, args=(inputs, outputs))
    process.daemon = True
    process.start()

  # Write the output to file.
  writer = tf.python_io.TFRecordWriter(path)
  counter = 0
  while True:
    example = outputs.get()
    if example is None:
      counter += 1
      if counter == num_threads:
        break
      else:
        continue
    writer.write(example.SerializeToString())
  writer.close()


def make_input_fn(dataset, mode, params, 
                  num_epochs=None, 
                  shuffle_batches=10,
                  num_threads=8,
                  initializable_iterator=False):
  """Make input function."""
  def _parse(*args):
    return dataset.parse(mode, *args)
  def _input_fn():
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
      d = dataset.read(mode)
      d = d.cache()
      if mode == tf.estimator.ModeKeys.TRAIN:
        d = d.repeat(num_epochs)
        d = d.shuffle(params.batch_size * shuffle_batches)
      d = d.map(_parse, num_threads=num_threads)
      d = d.batch(params.batch_size)
      if initializable_iterator:
        it = d.make_initializable_iterator()
        features, labels = it.get_next()
        return it.initializer, features, labels
      else:
        it = d.make_one_shot_iterator()
        features, labels = it.get_next()
        return features, labels
  return _input_fn
