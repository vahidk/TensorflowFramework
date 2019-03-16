"""Image utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.ExifTags
import scipy.ndimage as ndimage
import tensorflow as tf


face_cascade = cv2.CascadeClassifier(
  "data/opencv/haarcascade_frontalface_alt.xml")


def normalize_translation_and_scale(labels):
  center = labels.mean(axis=-2, keepdims=True)
  labels_c = labels - center
  norm = np.linalg.norm(labels_c, axis=-1, keepdims=True)
  scale = norm.mean(axis=-2, keepdims=True)
  labels_cs = labels_c / scale
  return labels_cs, center, scale


def crop_image(image, labels, bbox):
  """Crop an annotated image."""
  image_size = np.array([image.shape[1], image.shape[0]])
  bbox = np.array(bbox, dtype=np.int32)
  tl = bbox[0:2]
  br = bbox[0:2] + bbox[2:4]
  if np.any(tl < 0) or np.any(br > image_size - 1):
    pad = np.maximum(
      np.maximum(-tl, 0),
      np.maximum(br - image_size + 1, 0))
    image = np.pad(image, ((pad[1], pad[1]), 
                           (pad[0], pad[0]), 
                           (0, 0)), "constant")
    labels += pad[np.newaxis, :]
    tl += pad
    br += pad
  image = image[tl[1]:br[1], tl[0]:br[0], :]
  labels -= tl[np.newaxis, :]
  return image, labels


def compact_crop(image, labels, margin=0):
  image = np.array(image)
  labels = np.array(labels).astype(np.int32)
  minimum = np.amin(labels, axis=0)
  maximum = np.amax(labels, axis=0)
  center = ((minimum + maximum) / 2).astype(np.int32)
  half_size = np.amax((1 + margin) * (maximum - minimum) / 2).astype(np.int32)
  tl = center - half_size
  bbox = tl[0], tl[1], half_size * 2, half_size * 2
  image, labels = crop_image(image, labels, bbox)
  return image, labels


def rotate_landmarks(labels, center, angle):
  labels = np.array(labels, dtype=np.float32)
  center = np.array(center, dtype=np.float32)
  transform = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]])
  labels -= center[np.newaxis]
  labels = np.dot(labels, transform)
  labels += center[np.newaxis]
  return labels


def rotate_image(image, labels, angle):
  h, w = image.shape[0], image.shape[1]
  image = ndimage.rotate(image, np.degrees(angle), reshape=False)
  labels = rotate_landmarks(labels, [w / 2, h / 2], angle)
  return image, labels


def align_to_shape(image, labels, target, extend=1.0, rotate=False):
  image = np.array(image)
  labels = np.array(labels).astype(np.float32)
  target = np.array(target).astype(np.float32)
  m, _ = cv2.estimateAffinePartial2D(labels, target)
  image_t = cv2.warpAffine(image, m, (128, 128))
  labels_t = np.dot(labels, m[:,:2].T) + m[np.newaxis,:,2]
  return image_t, labels_t


def scale_image(image, labels, size):
  """Scale an annotated image."""
  image = PIL.Image.fromarray(image)
  zoom = np.array([size, size], dtype=np.float32) / np.array(image.size, np.float32)
  image = image.resize([size, size], resample=PIL.Image.ANTIALIAS)
  image = np.array(image)
  labels = labels * zoom
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


def decode_image(fp, force_rgb=True):
  """Decode the given image to a numpy array."""
  im = PIL.Image.open(fp)

  # correct rotation
  orientation_key = 0x0112
  if hasattr(im, "_getexif") and im._getexif():
    orientation = im._getexif().get(orientation_key, 0)
    if orientation == 3:
      im = im.rotate(180, expand=True)
    elif orientation == 6:
      im = im.rotate(270, expand=True)
    elif orientation == 8:
      im = im.rotate(90, expand=True)

  if force_rgb:
    im = im.convert(mode='RGB')

  im = np.array(im)
  return im


def visualize(image, labels):
  """Visualize image."""
  plt.figure(figsize=(16, 12))
  plt.imshow(image)
  plt.plot(labels[:, 0], labels[:, 1], ".")
  plt.axis("off")
  plt.show()
