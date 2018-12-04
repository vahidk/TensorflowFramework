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
from common import image_utils
from common import io_utils
from common import ops

LOCAL_DIR = "data/helen/"
IMAGES_DIR = os.path.join(LOCAL_DIR, "images")
BBOXES_DIR = os.path.join(LOCAL_DIR, "bboxes")
LABELS_DIR = os.path.join(LOCAL_DIR, "annotation")
TRAIN_DATA = os.path.join(LOCAL_DIR, "trainnames.txt")
TEST_DATA = os.path.join(LOCAL_DIR, "testnames.txt")
RECORDS_PATH = os.path.join(LOCAL_DIR, "data_%s.tfrecord")

IMAGE_SIZE = 128
NUM_LANDMARKS = 194


class Helen(dataset.AbstractDataset):

  def get_params(self):
    return {
      "image_size": IMAGE_SIZE,
      "num_landmarks": NUM_LANDMARKS,
      "left_eye": np.arange(134, 154),
      "right_eye": np.arange(114, 134),
    }

  def prepare(self):
    _convert_data(tf.estimator.ModeKeys.TRAIN)
    _convert_data(tf.estimator.ModeKeys.EVAL)

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
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    label = tf.reshape(features["label"], [NUM_LANDMARKS, 2])

    image, label = ops.random_crop_image(
      image, label, (8, 8), mode==tf.estimator.ModeKeys.TRAIN)

    return {"image": image}, {"label": label}


dataset.DatasetFactory.register("helen", Helen)


def _iterator(mode):
  train_paths = open(TRAIN_DATA).read().strip().splitlines()
  test_paths = open(TEST_DATA).read().strip().splitlines()
  names = {
    tf.estimator.ModeKeys.TRAIN: train_paths,
    tf.estimator.ModeKeys.EVAL: test_paths,
  }[mode]

  paths = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
  labels_map = {}
  for path in paths:
    content = open(path).read().strip().splitlines()
    name = content[0]
    labels = np.array([row.split(" , ") for row in content[1:]], dtype=np.float)
    labels_map[name] = labels

  for name in names:
    labels = labels_map[name]
    yield name, labels


def _convert_data(mode):
  if not os.path.exists(BBOXES_DIR):
    os.mkdir(BBOXES_DIR)

  def _create_example(item):
    name, labels = item
    print("Processing", name)

    image_path = os.path.join(IMAGES_DIR, "%s.jpg" % name)
    image = np.asarray(PIL.Image.open(image_path))

    bbox_path = "%s/%s_bbox.txt" % (BBOXES_DIR, name)
    if os.path.exists(bbox_path):
      bbox = np.loadtxt(bbox_path).astype(np.int32)
    else:
      bbox = image_utils.detect_face(image, labels)
      np.savetxt(bbox_path, bbox)

    image, labels = image_utils.crop_and_scale_image(
      image, labels, bbox, IMAGE_SIZE)

    image = image_utils.encode_image(image)
    labels = np.reshape(labels, [-1])

    example = tf.train.Example(features=tf.train.Features(
      feature={
        "image": tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[image])),
        "label": tf.train.Feature(
          float_list=tf.train.FloatList(value=labels))
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
    label = np.reshape(np.array(
      example.features.feature["label"].float_list.value),
      [NUM_LANDMARKS, 2])

    plt.imshow(image)
    plt.plot(label[:,0], label[:,1], ".")
    plt.show()


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python helen.py <convert|visualize>")
    sys.exit(1)

  if sys.argv[1] == "convert":
    _convert_data(tf.estimator.ModeKeys.TRAIN)
    _convert_data(tf.estimator.ModeKeys.EVAL)
  elif sys.argv[1] == "visualize":
    _visulize_data(tf.estimator.ModeKeys.TRAIN)
  else:
    print("Unknown command", sys.argv[1])
