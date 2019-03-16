"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import collections
import json
import random
import tensorflow as tf


class Tuple(object):

  def __init__(self, init):
    if isinstance(init, dict):
      dic = {}
      for k, v in init.items():
        if isinstance(v, dict):
          v = Tuple(v)
        dic[k] = v
      self._dict = dic
    elif isinstance(init, list):
      self._dict = {k: None for k in init}
    else:
      raise ValueError("Invalid init value {}.".format(init))
  
  def __getattr__(self, name):
    if not name in self._dict:
      raise ValueError("Field {} doesn't exist.".format(name))
    return self._dict[name]

  def __repr__(self):
    return json.dumps(self.to_dict())
  
  def __str__(self):
    return json.dumps(self.to_dict(), indent=4, sort_keys=True)

  def __eq__(self, other):
    return repr(self) == repr(other)

  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __deepcopy__(self, memo):
    return Tuple(self._dict.copy())

  def to_dict(self):
    dic = {}
    for k, v in self._dict.items():
      if isinstance(v, Tuple):
        v = v.to_dict()
      dic[k] = v
    return dic
  
  @staticmethod
  def from_string(text):
    return Tuple(deserialize_json(text))

  @staticmethod
  def from_file(path):
    return Tuple.from_string(open(path).read())


def deserialize_json(text):
  try:
    return ast.literal_eval(str(text))
  except:
    return json.loads(str(text))


def serialize_json(data, indent=4, sort_keys=True):
  return json.dumps(
    deserialize_json(data), indent=indent, sort_keys=sort_keys)


def lookup_flat(dic, flat_key, sep="."):
  p = dic
  for k in flat_key.split(sep):
    p = p[k]
  return p


def shuffle(lst):
  random.shuffle(lst)
  return lst
