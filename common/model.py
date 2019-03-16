"""Abstract model class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class AbstractModel(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_params(self):
    """Return model parameters."""
    return {}

  @abc.abstractmethod
  def get_features(self):
    """Return dictionary of feature names to placeholders."""
    return {}

  @abc.abstractmethod
  def model(self, features, labels, mode, params):
    """Main model class."""
    pass


class ModelFactory(object):

  models = {}

  @staticmethod
  def register(name, modelClass):
    """Register model."""
    ModelFactory.models[name] = modelClass

  @staticmethod
  def create(name):
    """Create model."""
    return ModelFactory.models[name]()
