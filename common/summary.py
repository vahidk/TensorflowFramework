"""Summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from common import image_utils
from common import ops


def image_grid(images, rows, cols, name, clip=False):
  """Visualize image grid."""
  n = rows * cols
  images = images[:n]
  bs, h, w, c = ops.get_shape(images)
  images = tf.reshape(images, [rows, cols, h, w, c])
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * h, cols * w, c])
  if clip:
    images = tf.clip_by_value(images, 0, 255)
  return tf.summary.image(name, images)


def labeled_face(images, labels, predictions, max_outputs=3, 
                 flip_vertical=False, name="face"):
  """Writes a summary visualizing given face images and corresponding labels."""
  def _visualize_face(image, labels, predictions):
    fig = plt.figure(figsize=(3, 3), dpi=80)
    ax = fig.add_subplot(111)
    if flip_vertical:
      image = image[::-1,...]
    ax.imshow(image.squeeze())
    w = image.shape[1]
    h = image.shape[0]
    ax.plot(labels[:, 0], labels[:, 1], "b.", 
            markersize=1, alpha=0.5)
    ax.plot(predictions[:, 0], predictions[:, 1], "r.", 
            markersize=1, alpha=0.5)
    for i in range(labels.shape[0]):
      ax.plot([predictions[i, 0], labels[i, 0]],
              [predictions[i, 1], labels[i, 1]], 'r-', 
              linewidth=0.5, alpha=0.5)
    fig.canvas.draw()
    buf = io.BytesIO()
    data = fig.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img = image_utils.decode_image(buf)
    return img

  def _visualize_faces(images, labels, predictions):
    outputs = []
    for i in range(max_outputs):
      output = _visualize_face(images[i], labels[i], predictions[i])
      outputs.append(output)
    return np.array(outputs, dtype=np.uint8)

  figs = tf.py_func(_visualize_faces, [images, labels, predictions], tf.uint8)
  return tf.summary.image(name, figs)


def labeled_image(name, images, labels, max_outputs=3, flip_vertical=False,
                  color="pink", font_size=15):
  """Writes a summary visualizing given images and corresponding labels."""
  def _visualize_image(image, label):
    # Do the actual drawing in python
    fig = plt.figure(figsize=(3, 3), dpi=80)
    ax = fig.add_subplot(111)
    if flip_vertical:
      image = image[::-1,...]
    ax.imshow(image.squeeze())
    ax.text(0, 0, str(label),
      horizontalalignment="left",
      verticalalignment="top",
      color=color,
      fontsize=font_size)
    fig.canvas.draw()

    # Write the plot as a memory file.
    buf = io.BytesIO()
    data = fig.savefig(buf, format="png")
    buf.seek(0)

    # Read the image and convert to numpy array
    return image_utils.decode_image(buf)

  def _visualize_images(images, labels):
    # Only display the given number of examples in the batch
    outputs = []
    for i in range(max_outputs):
      output = _visualize_image(images[i], labels[i])
      outputs.append(output)
    return np.array(outputs, dtype=np.uint8)

  # Run the python op.
  figs = tf.py_func(_visualize_images, [images, labels], tf.uint8)
  return tf.summary.image(name, figs)
