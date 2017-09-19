"""Cifar100 dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import cPickle
from six.moves import urllib
import struct
import sys
import tarfile
import tensorflow as tf

from common import utils

REMOTE_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
LOCAL_DIR = os.path.join("data/cifar100/")
ARCHIVE_NAME = "cifar-100-python.tar.gz"
DATA_DIR = "cifar-100-python/"
TRAIN_BATCHES = ["train"]
TEST_BATCHES = ["test"]

IMAGE_SIZE = 32
NUM_CLASSES = 100


def get_params():
  """Return dataset parameters."""
  return {
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
  }


def prepare():
  """Download the cifar 100 dataset."""
  if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)
  if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
    print("Downloading...")
    urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
  if not os.path.exists(LOCAL_DIR + DATA_DIR):
    print("Extracting files...")
    tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
    tar.extractall(LOCAL_DIR)
    tar.close()


def read(mode):
  """Create an instance of the dataset object."""
  batches = {
    tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
    tf.estimator.ModeKeys.EVAL: TEST_BATCHES
  }[mode]

  all_images = []
  all_labels = []

  for batch in batches:
    with open("%s%s%s" % (LOCAL_DIR, DATA_DIR, batch), "rb") as fo:
      dict = cPickle.load(fo)
      images = np.array(dict["data"])
      labels = np.array(dict["fine_labels"])

      num = images.shape[0]
      images = np.reshape(images, [num, 3, IMAGE_SIZE, IMAGE_SIZE])
      images = np.transpose(images, [0, 2, 3, 1])
      print("Loaded %d examples." % num)

      all_images.append(images)
      all_labels.append(labels)

  all_images = np.concatenate(all_images)
  all_labels = np.concatenate(all_labels)

  return tf.contrib.data.Dataset.from_tensor_slices((all_images, all_labels))


def parse(mode, image, label):
  """Parse input record to features and labels."""
  image = tf.to_float(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

  if mode == tf.estimator.ModeKeys.TRAIN:
    image = tf.image.resize_image_with_crop_or_pad(
      image, IMAGE_SIZE + 4, IMAGE_SIZE + 4)
    image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)

  image = tf.image.per_image_standardization(image)

  return {"image": image}, {"label": label}
