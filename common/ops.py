"""Common TensorFlow ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def reshape(tensor, dims_list):
  """Reshape the given tensor by collapsing dimensions."""
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor


def dense_layers(tensor,
                 sizes,
                 activation=tf.nn.relu,
                 linear_top_layer=False,
                 drop_rate=0.0,
                 name=None,
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  with tf.variable_scope(name, default_name="dense_layers"):
    for i, size in enumerate(sizes):
      if i == len(sizes) - 1 and linear_top_layer:
        activation = None
      tensor = tf.layers.dropout(tensor, drop_rate)
      tensor = tf.layers.dense(
          tensor,
          size,
          name="dense_layer_%d" % i,
          activation=activation,
          **kwargs)
  return tensor


def conv_layers(tensor,
                filters,
                kernels,
                pools,
                padding="same",
                activation=tf.nn.relu,
                drop_rate=0.0,
                **kwargs):
  """Builds a stack of convolutional layers with dropout and max pooling."""
  for fs, ks, ps in zip(filters, kernels, pools):
    tensor = tf.layers.dropout(tensor, drop_rate)
    tensor = tf.layers.conv2d(
      tensor,
      filters=fs,
      kernel_size=ks,
      padding=padding,
      activation=activation,
      **kwargs)
    if ps and ps > 1:
      tensor = tf.layers.max_pooling2d(
        inputs=tensor, pool_size=ps, strides=ps, padding=padding)
  return tensor


def log_prob_from_logits(logits, axis=-1):
  """Normalize the log-probabilities so that probabilities sum to one."""
  return logits - tf.reduce_logsumexp(logits, axis=axis, keep_dims=True)


def batch_gather(tensor, indices):
  """Gather in batch from a tensor of arbitrary size.

  In pseduocode this module will produce the following:
  output[i] = tf.gather(tensor[i], indices[i])

  Args:
    tensor: Tensor of arbitrary size.
    indices: Vector of indices.
  Returns:
    output: A tensor of gathered values.
  """
  shape = get_shape(tensor)
  flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
  indices = tf.convert_to_tensor(indices)
  offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
  offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
  output = tf.gather(flat_first, indices + offset)
  return output


def rnn_beam_search(update_fn, initial_state, sequence_length, beam_width,
                    begin_token_id, end_token_id, name="rnn"):
  """Beam-search decoder for recurrent models.

  Args:
    update_fn: Function to compute the next state and logits given the current
               state and ids.
    initial_state: Recurrent model states.
    sequence_length: Length of the generated sequence.
    beam_width: Beam width.
    begin_token_id: Begin token id.
    end_token_id: End token id.
    name: Scope of the variables.
  Returns:
    ids: Output indices.
    logprobs: Output log probabilities probabilities.
  """
  batch_size = initial_state.shape.as_list()[0]

  state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])

  sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])

  ids = tf.tile([[begin_token_id]], [batch_size, beam_width])
  sel_ids = tf.expand_dims(ids, axis=2)

  mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

  for i in range(sequence_length):
    with tf.variable_scope(name, reuse=True if i > 0 else None):

      state, logits = update_fn(state, ids)
      logits = log_prob_from_logits(logits)

      sum_logprobs = (
          tf.expand_dims(sel_sum_logprobs, axis=2) +
          (logits * tf.expand_dims(mask, axis=2)))

      num_classes = logits.shape.as_list()[-1]

      sel_sum_logprobs, indices = tf.nn.top_k(
          tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
          k=beam_width)

      ids = indices % num_classes

      beam_ids = indices // num_classes

      state = batch_gather(state, beam_ids)

      sel_ids = tf.concat([batch_gather(sel_ids, beam_ids),
                           tf.expand_dims(ids, axis=2)], axis=2)

      mask = (batch_gather(mask, beam_ids) *
              tf.to_float(tf.not_equal(ids, end_token_id)))

  return sel_ids, sel_sum_logprobs


def merge(tensors, units, activation=tf.nn.relu, name=None, **kwargs):
  """Merge features with broadcasting support.

  This operation concatenates multiple features of varying length and applies
  non-linear transformation to the outcome.

  Example:
    a = tf.zeros([m, 1, d1])
    b = tf.zeros([1, n, d2])
    c = merge([a, b], d3)  # shape of c would be [m, n, d3].

  Args:
    tensors: A list of tensor with the same rank.
    units: Number of units in the projection function.
  """
  with tf.variable_scope(name, default_name="merge"):
    # Apply linear projection to input tensors.
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.layers.dense(
          tensor, units, activation=None,
          name="proj_%d" % i,
          **kwargs)
      projs.append(proj)

    # Compute sum of tensors.
    result = projs.pop()
    for proj in projs:
      result = result + proj

    # Apply nonlinearity.
    if activation:
      result = activation(result)
  return result


def softmax_entropy(logits, dim=-1):
  """Compute softmax entropy from logits."""
  plogp = tf.nn.softmax(logits, dim) * tf.nn.log_softmax(logits, dim)
  return -tf.reduce_sum(plogp, dim)


def create_optimizer(optimizer, learning_rate, decay_steps=None, **kwargs):
  """Create an optimizer object."""
  global_step = tf.train.get_or_create_global_step()

  if decay_steps:
    learning_rate = tf.train.exponential_decay(
      learning_rate, global_step, decay_steps, 0.5, staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)

  return tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer](
    learning_rate, **kwargs)


def average_gradients(tower_grads):
  """Compute average gradients."""
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = [g for g, _ in grad_and_vars]
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)
    v = grad_and_vars[0][1]
    average_grads.append((grad, v))
  return average_grads
