"""Mnist dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import urllib
import struct
import sys
import tensorflow as tf

from common import dataset
from common import misc_utils

REMOTE_URL = "http://yann.lecun.com/exdb/mnist/"
LOCAL_DIR = "data/mnist/"
TRAIN_IMAGE_URL = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_URL = "train-labels-idx1-ubyte.gz"
TEST_IMAGE_URL = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_URL = "t10k-labels-idx1-ubyte.gz"

IMAGE_SIZE = 28
NUM_CLASSES = 10


class Mnist(dataset.AbstractDataset):

  def get_params(self):
    return {
      "image_size": IMAGE_SIZE,
      "num_classes": NUM_CLASSES,
    }

  def prepare(self, params):
    """This function will be called once to prepare the dataset."""
    if not os.path.exists(LOCAL_DIR):
      os.makedirs(LOCAL_DIR)
    for name in [
      TRAIN_IMAGE_URL,
      TRAIN_LABEL_URL,
      TEST_IMAGE_URL,
      TEST_LABEL_URL]:
      if not os.path.exists(LOCAL_DIR + name):
        urllib.request.urlretrieve(REMOTE_URL + name, LOCAL_DIR + name)

  def read(self, split, params):
    """Create an instance of the dataset object."""
    image_urls = {
      "train": TRAIN_IMAGE_URL,
      "eval": TEST_IMAGE_URL
    }[split]
    label_urls = {
      "train": TRAIN_LABEL_URL,
      "eval": TEST_LABEL_URL
    }[split]

    with gzip.open(LOCAL_DIR + image_urls, "rb") as f:
      _, num, rows, cols = struct.unpack(">IIII", f.read(16))
      images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
      images = np.reshape(images, [num, rows, cols, 1])
      print("Loaded %d images of size [%d, %d]." % (num, rows, cols))

    with gzip.open(LOCAL_DIR + label_urls, "rb") as f:
      _, num = struct.unpack(">II", f.read(8))
      labels = np.frombuffer(f.read(num), dtype=np.int8).astype(np.int32)
      print("Loaded %d labels." % num)

    return tf.data.Dataset.from_tensor_slices((images, labels))


  def parse(self, mode, params, image, label):
    """Parse input record to features and labels."""
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])

    # image = tf.image.per_image_standardization(image)

    return {"image": image}, {"label": label}


dataset.DatasetFactory.register("mnist", Mnist)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python dataset.mnist <convert|visualize>")
    sys.exit(1)

  if sys.argv[1] == "download":
    d = Mnist()
    d.prepare(misc_utils.Tuple(d.get_params()))
  else:
    print("Unknown command", sys.argv[1])
