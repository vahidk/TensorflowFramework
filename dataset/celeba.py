"""Helen dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import sys
import tensorflow as tf

from common import dataset
from common import io_utils

LOCAL_DIR = "data/celeba/"
PARTITIONS = LOCAL_DIR + "Eval/list_eval_partition.txt"
ATTRIBUTES = LOCAL_DIR + "Anno/list_attr_celeba.txt"
LANDMARKS = LOCAL_DIR + "Anno/list_landmarks_align_celeba.txt"
IMAGE_DIR = LOCAL_DIR + "img_align_celeba/"
RECORDS_PATH = os.path.join(LOCAL_DIR, "data_%s.tfrecord")

IMAGE_SIZE = [218, 178]
NUM_LANDMARKS = 5
NUM_ATTRIBUTES = 40


class CelebA(dataset.AbstractDataset):

  def get_params(self):
    return {
      "num_landmarks": NUM_LANDMARKS,
      "left_eye": [0],
      "right_eye": [1],
    }

  def prepare(self):
    _convert_data(tf.estimator.ModeKeys.TRAIN)
    _convert_data(tf.estimator.ModeKeys.EVAL)
    _convert_data(tf.estimator.ModeKeys.PREDICT)

  def read(self, mode):
    """Create an instance of the dataset object."""
    return tf.contrib.data.TFRecordDataset(RECORDS_PATH % mode)


  def parse(self, mode, record):
    """Parse input record to features and labels."""
    features = tf.parse_single_example(record, {
      "image": tf.FixedLenFeature([], tf.string),
      "label": tf.FixedLenFeature([NUM_LANDMARKS * 2], tf.float32),
    })

    image = tf.to_float(tf.image.decode_image(features["image"], 3)) / 255.0
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    landmark = tf.reshape(features["landmark"], [NUM_LANDMARKS, 2])
    attribute = tf.reshape(features["attribute"], [NUM_LANDMARKS, 2])

    return {"image": image}, {"landmark": landmark, "attribute": attribute}


dataset.DatasetFactory.register("celeba", CelebA)


def _read_annotations(path, dtype):
  file = open(path)
  rows = int(next(file))
  names = next(file).split()
  cols = len(names)
  data = np.zeros([rows, cols], dtype=dtype)
  for i in range(cols):
    data[i] = next(file).split()[1:]
  return data


def _read_paritions(path):
  lines = open(path).readlines()
  rows = len(lines)
  data = np.zeros([rows], dtype=np.int32)
  for i, line in enumerate(lines):
    data[i] = line.split()[1]
  train = np.where(data==0)[0]
  val = np.where(data==1)[0]
  test = np.where(data==2)[0]
  return train, val, test


def _iterator(mode):
  train, val, test = _read_paritions(PARTITIONS)
  ids = {
    tf.estimator.ModeKeys.TRAIN: train,
    tf.estimator.ModeKeys.EVAL: val,
    tf.estimator.ModeKeys.PREDICT: test,
  }[mode]

  landmarks = _read_annotations(LANDMARKS, np.float32)
  attributes = _read_annotations(ATTRIBUTES, np.int64)

  for id in ids:
    name = IMAGE_DIR + "%06d.jpg" % (id + 1)
    landmark = landmarks[id]
    attribute = attributes[id]
    yield name, landmark, attribute


def _convert_data(mode):
  def _create_example(item):
    name, landmark, attribute = item
    print("Processing", name)

    # Load the image.
    image = open(name).read()

    # Write record
    example = tf.train.Example(features=tf.train.Features(
      feature={
        "image": tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[image])),
        "landmark": tf.train.Feature(
          float_list=tf.train.FloatList(value=landmark)),
        "attribute": tf.train.Feature(
          int64_list=tf.train.Int64List(value=attribute))
      }))
    return example

  io_utils.parallel_record_writer(
    _iterator(mode), _create_example, RECORDS_PATH % mode)


def _visulize_data(mode):
  path = RECORDS_PATH % mode
  iterator = tf.python_io.tf_record_iterator(path)
  for i in range(5):
    item = next(iterator)

    example = tf.train.Example()
    example.ParseFromString(item)

    image = PIL.Image.open(io.BytesIO(
      example.features.feature["image"].bytes_list.value[0]))
    landmark = np.reshape(np.array(
      example.features.feature["landmark"].float_list.value),
      [NUM_LANDMARKS, 2])

    plt.imshow(image)
    plt.plot(landmark[:, 0], landmark[:, 1], ".")
    plt.show()


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python helen.py <convert|visualize>")
    sys.exit(1)

  if sys.argv[1] == "convert":
    _convert_data(tf.estimator.ModeKeys.TRAIN)
    _convert_data(tf.estimator.ModeKeys.EVAL)
    _convert_data(tf.estimator.ModeKeys.PREDICT)
  elif sys.argv[1] == "visualize":
    _visulize_data(tf.estimator.ModeKeys.TRAIN)
  else:
    print("Unknown command", sys.argv[1])
