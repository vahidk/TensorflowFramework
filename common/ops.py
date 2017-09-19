"""Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numbers
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
    if isinstance(dims, numbers.Number):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor


def expand_dims(tensor, dims):
  """Expand the rank of a tensor by inserting singular dimensions."""
  if isinstance(dims, numbers.Number):
    dims = [dims]
  for dim in dims:
    tensor = tf.expand_dims(tensor, dim)
  return tensor


def tile_like(tensor, like):
  """Tile a tensor to match another."""
  tensor = tf.tile(tensor, tf.shape(like) / tf.shape(tensor))
  return tensor


def batch_normalization(tensor, training=False, epsilon=0.001, momentum=0.9, 
                        fused_batch_norm=False, name=None):
  """Performs batch normalization on given 4-D tensor.
  
  The features are assumed to be in NHWC format. Noe that you need to 
  run UPDATE_OPS in order for this function to perform correctly, e.g.:

  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(loss)

  Based on: https://arxiv.org/abs/1502.03167
  """
  with tf.variable_scope(name, default_name="batch_normalization"):
    channels = tensor.shape.as_list()[-1]
    axes = list(range(tensor.shape.ndims - 1))

    beta = tf.get_variable(
      'beta', channels, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
      'gamma', channels, initializer=tf.ones_initializer())

    avg_mean = tf.get_variable(
      "avg_mean", channels, initializer=tf.zeros_initializer(),
      trainable=False)
    avg_variance = tf.get_variable(
      "avg_variance", channels, initializer=tf.ones_initializer(),
      trainable=False)

    if training:
      if fused_batch_norm:
        mean, variance = None, None
      else:
        mean, variance = tf.nn.moments(tensor, axes=axes)
    else:
      mean, variance = avg_mean, avg_variance
   
    if fused_batch_norm:
      tensor, mean, variance = tf.nn.fused_batch_norm(
        tensor, scale=gamma, offset=beta, mean=mean, variance=variance, 
        epsilon=epsilon, is_training=training)
    else:
      tensor = tf.nn.batch_normalization(
        tensor, mean, variance, beta, gamma, epsilon)

    if training:
      update_mean = tf.assign(
        avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
      update_variance = tf.assign(
        avg_variance, avg_variance * momentum + variance * (1.0 - momentum))

      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

  return tensor


def squeeze_and_excite(tensor, ratio, name=None):
  """Apply squeeze/excite on given 4-D tensor.
  
  Based on: https://arxiv.org/abs/1709.01507
  """
  with tf.variable_scope(name, default_name="squeeze_and_excite"):
    original = tensor
    units = tensor.shape.as_list()[-1]
    tensor = tf.reduce_mean(tensor, [1, 2], keep_dims=True)
    tensor = dense_layers(
      tensor, [units / ratio, units], use_bias=False, linear_top_layer=True)
    tensor = tf.nn.sigmoid(tensor)
    tensor = original * tensor
  return tensor


def dense_layers(tensor,
                 units,
                 activation=tf.nn.relu,
                 use_bias=True,
                 linear_top_layer=False,
                 drop_rates=None,
                 batch_norm=False,
                 training=False,
                 weight_decay=0.0002,
                 **kwargs):
  """Builds a stack of fully connected layers with optional dropout."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  for i, (size, drp) in enumerate(zip(units, drop_rates)):
    if i == len(units) - 1 and linear_top_layer:
      activation = None
    with tf.variable_scope("dense_block_%d" % i):
      tensor = tf.layers.dropout(tensor, drp)
      tensor = tf.layers.dense(
        tensor, size, use_bias=use_bias,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        **kwargs)
      if activation:
        if batch_norm:
          tensor = batch_normalization(tensor, training=training)
        tensor = activation(tensor)
  return tensor


def conv_layers(tensor,
                filters,
                kernels,
                strides=None,
                pool_sizes=None,
                pool_strides=None,
                padding="same",
                activation=tf.nn.relu,
                use_bias=False,
                linear_top_layer=False,
                drop_rates=None,
                conv_method="conv",
                pool_method="conv",
                pool_activation=None,
                batch_norm=False,
                training=False,
                weight_decay=0.0002,
                **kwargs):
  """Builds a stack of convolutional layers with dropout and max pooling."""
  if pool_sizes is None:
    pool_sizes = [1] * len(filters)
  if pool_strides is None:
    pool_strides = pool_sizes
  if strides is None:
    strides = [1] * len(filters)
  if drop_rates is None:
    drop_rates = [0.] * len(filters)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(filters)

  if conv_method == "conv":
    conv = functools.partial(
      tf.layers.conv2d,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "transposed":
    conv = functools.partial(
      tf.layers.conv2d_transpose,
      kernel_initializer=tf.glorot_uniform_initializer(),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
  elif conv_method == "separable":
    conv = functools.partial(
      tf.layers.separable_conv2d,
      depthwise_initializer=tf.glorot_uniform_initializer(),
      pointwise_initializer=tf.glorot_uniform_initializer(),
      depthwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      pointwise_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

  for i, (fs, ks, ss, pz, pr, drp) in enumerate(
    zip(filters, kernels, strides, pool_sizes, pool_strides, drop_rates)):
    with tf.variable_scope("conv_block_%d" % i):
      if i == len(filters) - 1 and linear_top_layer:
        activation = None
        pool_activation = None
      tensor = tf.layers.dropout(tensor, drp)
      tensor = conv(
        tensor, fs, ks, ss, padding, use_bias=use_bias, name="conv2d",
        **kwargs)
      if activation:
        if batch_norm:
          tensor = batch_normalization(tensor, training=training)
        tensor = activation(tensor)
      if pz > 1:
        if pool_method == "max":
          tensor = tf.layers.max_pooling2d(
            tensor, pz, pr, padding, name="max_pool")
        elif pool_method == "std":
          tensor = tf.space_to_depth(tensor, pz, name="space_to_depth")
        elif pool_method == "dts":
          tensor = tf.depth_to_space(tensor, pz, name="depth_to_space")
        else:
          tensor = conv(
            tensor, fs, pz, pr, padding, use_bias=use_bias,
            name="strided_conv2d", **kwargs)
          if pool_activation:
            if batch_norm:
              tensor = batch_normalization(tensor, training=training)
            tensor = pool_activation(tensor)
  return tensor


def merge(tensors, units, activation=tf.nn.relu, name=None, **kwargs):
  """Merge tensors with broadcasting support."""
  with tf.variable_scope(name, default_name="merge"):
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.layers.dense(
          tensor, units, name="proj_%d" % i, **kwargs)
      projs.append(proj)

    result = projs.pop()
    for proj in projs:
      result = result + proj

    if activation:
      result = activation(result)
  return result


def merge_layers(tensors, units, activation=tf.nn.relu,
                 linear_top_layer=False, drop_rates=None,
                 name=None, **kwargs):
  """Merge followed by a stack of dense layers."""
  if drop_rates is None:
    drop_rates = [0.] * len(units)
  elif isinstance(drop_rates, numbers.Number):
    drop_rates = [drop_rates] * len(units)
  with tf.variable_scope(name, default_name="merge_layers"):
    result = tf.layers.dropout(result, drop_rates[0])
    result = merge(tensors, units[0], activation, **kwargs)
    result = dense_layers(result, units[1:],
                          activation=activation,
                          drop_rates=drop_rates[1:],
                          linear_top_layer=linear_top_layer,
                          **kwargs)
  return result


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


def softmax_entropy(logits, dim=-1):
  """Softmax entropy from logits."""
  plogp = tf.nn.softmax(logits, dim) * tf.nn.log_softmax(logits, dim)
  return -tf.reduce_sum(plogp, dim)


def leaky_relu(tensor, alpha=0.1):
  """Computes the leaky rectified linear activation."""
  return tf.maximum(tensor, alpha * tensor)
