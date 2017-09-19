"""Loss ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gaussian_kl(q, p=(0., 0.)):
  """Computes the KL divergence between two isotropic Gaussian distributions.

  Args:
    q: A tuple (mu, log_sigma_sq) representing a multi-variatie Gaussian.
    p: A tuple (mu, log_sigma_sq) representing a multi-variatie Gaussian.
  Returns:
    A tensor representing KL(q, p).
  """
  mu1, log_sigma1_sq = q
  mu2, log_sigma2_sq = p
  return tf.reduce_sum(
    0.5 * (log_sigma2_sq - log_sigma1_sq +
           tf.exp(log_sigma1_sq - log_sigma2_sq) +
           tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) -
           1), axis=-1)


def gan_loss(x, gz, discriminator):
  """Original GAN loss.

  Args:
    x: Batch of real samples.
    gz: Batch of generated samples.
    discriminator: Discriminator function.
  Returns:
    d_loss: Discriminator loss.
    g_loss: Generator loss.
  """
  dx = discriminator(x)
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    dgz = discriminator(gz)
  d_loss = -tf.reduce_mean(tf.log_sigmoid(dx) + tf.log_sigmoid(-dgz))
  g_loss = -tf.reduce_mean(tf.log_sigmoid(dgz))
  return d_loss, g_loss


def lsgan_loss(x, gz, discriminator):
  """LS-GAN loss.

  Args:
    x: Batch of real samples.
    gz: Batch of generated samples.
    discriminator: Discriminator function.
  Returns:
    d_loss: Discriminator loss.
    g_loss: Generator loss.
  """
  dx = discriminator(x)
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    dgz = discriminator(gz)
  d_loss = tf.reduce_mean(tf.square(dx - 1.0) + tf.square(dgz))
  g_loss = tf.reduce_mean(tf.square(dgz - 1.0))
  return d_loss, g_loss


def wgan_loss(x, gz, discriminator, beta=10.0):
  """Improved Wasserstein GAN loss.

  Args:
    x: Batch of real samples.
    gz: Batch of generated samples.
    discriminator: Discriminator function.
    beta: Regualarizer factor.
  Returns:
    d_loss: Discriminator loss.
    g_loss: Generator loss.
  """
  dx = discriminator(x)
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    dgz = discriminator(gz)
  batch_size = tf.shape(x)[0]
  alpha = tf.random_uniform([batch_size])
  xhat = x * alpha + gz * (1 - alpha)
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    dxhat = discriminator(xhat)
  gnorm = tf.norm(tf.gradients(dxhat, xhat)[0])
  d_loss = -tf.reduce_mean(dx - dgz - beta * tf.square(gnorm - 1))
  g_loss = -tf.reduce_mean(dgz)
  return d_loss, g_loss
