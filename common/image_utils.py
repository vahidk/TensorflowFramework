"""Image utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.ndimage as ndimage
import tensorflow as tf


face_cascade = cv2.CascadeClassifier(
  "../app/Avatar/data/models/haarcascade_frontalface_alt.xml")


def pad_or_crop_image(image, padding, mode):
  """Pad or crop an annotated image."""
  p_pad = [(max(p[0], 0), max(p[1], 0)) for p in padding]
  n_pad = [(max(-p[0], 0), max(-p[1], 0)) for p in padding]
  min_p = [p[0] for i, p in enumerate(n_pad)]
  max_p = [image.shape[1-i]-p[1] for i, p in enumerate(n_pad)]
  image = image[min_p[1]:max_p[1], min_p[0]:max_p[0]]
  image = np.pad(image, [p_pad[1], p_pad[0], (0, 0)], mode)
  return image


def crop_image(image, labels, bbox):
  """Crop an annotated image."""
  image_w = image.shape[1]
  image_h = image.shape[0]
  pad_x = (bbox[2] // 4) - bbox[0], (bbox[0] + bbox[2] * 5 // 4) - image_w
  pad_y = (bbox[3] // 4) - bbox[1], (bbox[1] + bbox[3] * 5 // 4) - image_h
  image = pad_or_crop_image(image, (pad_x, pad_y), "reflect")
  labels = labels + np.array([[pad_x[0], pad_y[0]]])
  return image, labels


def scale_image(image, labels, size):
  """Scale an annotated image."""
  zoom = np.array([size, size]) / np.array([image.shape[0], image.shape[1]])
  image = ndimage.interpolation.zoom(image, [zoom[0], zoom[1], 1])
  labels = labels * zoom
  return image, labels


def crop_and_scale_image(image, labels, bbox, size):
  """Crop and scale an annotated image."""
  image, labels = crop_image(image, labels, bbox)
  image, labels = scale_image(image, labels, size)
  return image, labels


def overlap(bbox1, bbox2):
  """Compute the overlap beween two boxes."""
  tl = np.maximum(bbox1[0:2], bbox2[0:2])
  br = np.minimum(bbox1[0:2] + bbox1[2:4], bbox2[0:2] + bbox2[2:4])
  size = br - tl
  ratio = float(np.prod(size)) / np.prod(bbox1[2:4])
  return ratio


def detect_face(image, labels, min_overlap=0.8):
  """Detect face."""
  cv2.ocl.setUseOpenCL(False)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
  center = labels.mean(axis=0).astype(np.int)
  size = np.int((labels.max(axis=0) - labels.min(axis=0)).mean())
  size = np.array([size, size])
  pos = center - size // 2
  init_face = np.concatenate((pos, size))
  sel_face = init_face
  detected = False
  for face in faces:
    cur_overlap = overlap(init_face, face)
    if cur_overlap > min_overlap:
      sel_face = face
      min_overlap = cur_overlap
      detected = True
  print(["Not detected", "Detected"][detected])
  return sel_face.astype(np.int)


def encode_image(data, format="png"):
  """Encodes a numpy array to string."""
  im = PIL.Image.fromarray(data)
  buf = io.BytesIO()
  data = im.save(buf, format=format)
  buf.seek(0)
  return buf.getvalue()


def decode_image(data):
  """Decode the given image to a numpy array."""
  buf = io.BytesIO(data)
  im = PIL.Image.open(buf)
  data = np.array(im.getdata()).reshape([im.height, im.width, -1])
  return data


def visualize(image, labels):
  """Visualize image."""
  plt.figure(figsize=(16, 12))
  plt.imshow(image)
  plt.plot(labels[:, 0], labels[:, 1], ".")
  plt.axis("off")
  plt.show()
