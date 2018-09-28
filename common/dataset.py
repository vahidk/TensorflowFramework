"""Abstract dataset class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class AbstractDataset(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_params(self):
    """Return dataset parameters."""
    return {}

  @abc.abstractmethod
  def prepare(self):
    """This function will be called once to prepare the dataset."""
    pass


  @abc.abstractmethod
  def read(self, mode):
    """Create an instance of the dataset object."""
    pass


  @abc.abstractmethod
  def parse(self, mode, image, label):
    """Parse input record to features and labels."""
    pass


class DatasetFactory(object):

  datasets = {}

  @staticmethod
  def register(name, datasetClass):
    """Register dataset."""
    DatasetFactory.datasets[name] = datasetClass

  @staticmethod
  def create(name):
    """Create dataset."""
    return DatasetFactory.datasets[name]()
