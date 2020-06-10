import copy
import datetime
import json
import operator
import time
import sys
import os
import os.path
import multiprocessing
#import multiprocessing.dummy as multiprocessing
import queue
import gzip
import unittest
import signal
from collections import defaultdict
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass


from tqdm import trange, tqdm
import numpy as np


def make_timestamp():
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H.%M.%S")  # We choose this format to make the filename compatible with windows environmnets


class HyperParameter(object):
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def __repr__(self):
        raise NotImplementedError()

    def random_sample(self):
        ...


class IntegerRangeHyperParameter(HyperParameter):
    def __init__(self, low, high=None, rng=None):
        super().__init__(rng=rng)
        if high is None:
            high = low
            low = 0
        self.low = low
        self.high = high
        #self.current_item = low  # We'll see how we implement grid search, if it's done with an iterator this variable
                                  # will not be needed
    def __repr__(self):
        return "<{} low:{},high:{}>".format(self.__class__.__name__, self.low, self.high)

    def random_sample(self):
        return self.rng.randint(self.low, self.high)


class DiscreteHyperParameter(HyperParameter):
    def __init__(self, values, rng=None):
        super().__init__(rng=rng)
        self.values = list(values)
        self.current_item = 0

    def random_sample(self):
        return self.rng.choice(self.values)

    def __repr__(self):
        return "<{} values:{}>".format(self.__class__.__name__, self.values)


class LinearHyperParameter(HyperParameter):
    def __init__(self, low, high=None, num=None, rng=None):
        super().__init__(rng=rng)
        if high is None:
            high = low
            low = 0
        self.low = low
        self.high = high
        self.num = num

    def random_sample(self):
        return (self.high - self.low) * self.rng.random_sample() + self.low

    def __repr__(self):
        return "<{} low:{},high:{},num:{}>".format(self.__class__.__name__, self.low, self.high, self.num)


class GeometricHyperParameter(LinearHyperParameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = np.log10(self.low)
        self.high = np.log10(self.high)

    def random_sample(self):
        # If we transform the space from low to high to a log-scale and then draw uniform samples in that space,
        # by exponentiating them we should get the right value
        sample = LinearHyperParameter.random_sample(self)
        return np.power(10, sample)


def materialize_hyper_parameters(obj, search_method='random', non_collection_types=(str, bytes, bytearray, np.ndarray)):
    """Make any HyperParameter a concrete object"""
    if isinstance(obj, HyperParameter):
        if search_method == 'random':
            sample = obj.random_sample()
            return materialize_hyper_parameters(sample)
        else:
            raise NotImplementedError('Search method {} is not implemented'.format(search_method))
    elif isinstance(obj, Mapping):
        return type(obj)((k, materialize_hyper_parameters(v, search_method)) for k, v in obj.items())
    elif isinstance(obj, Collection) and not isinstance(obj, non_collection_types):
        return type(obj)(materialize_hyper_parameters(x, search_method) for x in obj)
    elif hasattr(obj, '__dict__'):
        obj_copy = copy.copy(obj)
        obj_copy.__dict__ = materialize_hyper_parameters(obj.__dict__)
        return obj_copy
    else:
        return obj


class HyperParameterManager(object):
    def __init__(self, base_args, base_kwargs, search_method='random'):
        self.base_args = base_args
        self.base_kwargs = base_kwargs
        self.search_method = search_method
        self.search_space = []
        self.hyper_parameters = dict()
        self.history = defaultdict(list)
        self.n_iter = 0
        self.setup_search_space()

    def setup_search_space(self):
        # for i, arg in enumerate(self.base_args):
        #     if isinstance(arg, HyperParameter):
        #         self.search_space.append((arg, 'args', i))
        # for k, v in self.base_kwargs.items():
        #     if isinstance(v, HyperParameter):
        #         self.search_space.append((v, 'kwargs', k))
        pass

    def get_hyper_parameters(self):
        args = list(self.base_args)
        kwargs = dict(self.base_kwargs.items())
        self.n_iter += 1
        hp_id = self.n_iter  ## When we implement smarter search methods, this should be a reference to
                             # the hp-point produced
        args = self.materialize_hyper_params(args)
        kwargs = self.materialize_hyper_params(kwargs)
        self.hyper_parameters[hp_id] = (args, kwargs)
        return hp_id, args, kwargs

    def report(self, hp_id, performance):
        # The idea is that the manager can do things with this history. Since we will probably not have a lot of
        # samples, just having a flat structure works for now. The argument is that if you need to do smart HP
        # optimization, the cost of producing a sample is high, and you will be in a data limited regime. Having to
        # iterate over a list will be a small cost compared to evaluating each sample.
        self.history[hp_id].append(performance)

    def best_hyper_params(self):
        best_performance = None
        best_args = None
        best_kwargs = None
        for hp_id, performances in self.history.items():
            for performance in performances:
                if best_performance is None or performance.cmp(best_performance):
                    best_performance = performance
                    best_args, best_kwargs = self.hyper_parameters[hp_id]
        return best_args, best_kwargs

    def get_any_hyper_params(self):
        hp_id, args, kwargs = self.get_hyper_parameters()
        return args, kwargs


class ObjectHyperParameterManager(object):
    def __init__(self, base_obj, search_method='random'):
        self.base_obj = base_obj
        self.search_method = search_method
        self.search_space = []
        self.hyper_parameters = dict()
        self.history = defaultdict(list)
        self.n_iter = 0
        self.setup_search_space()

    def setup_search_space(self):
        # for i, arg in enumerate(self.base_args):
        #     if isinstance(arg, HyperParameter):
        #         self.search_space.append((arg, 'args', i))
        # for k, v in self.base_kwargs.items():
        #     if isinstance(v, HyperParameter):
        #         self.search_space.append((v, 'kwargs', k))
        pass

    def get_hyper_parameters(self):
        materialized_obj = materialize_hyper_parameters(self.base_obj)
        self.n_iter += 1
        hp_id = self.n_iter  ## When we implement smarter search methods, this should be a reference to
                             # the hp-point produced
        self.hyper_parameters[hp_id] = materialized_obj
        return hp_id, materialized_obj

    def report(self, hp_id, performance):
        # The idea is that the manager can do things with this history. Since we will probably not have a lot of
        # samples, just having a flat structure works for now. The argument is that if you need to do smart HP
        # optimization, the cost of producing a sample is high, and you will be in a data limited regime. Having to
        # iterate over a list will be a small cost compared to evaluating each sample.
        self.history[hp_id].append(performance)

    def best_hyper_params(self):
        best_performance = None
        best_args = None
        best_kwargs = None
        for hp_id, performances in self.history.items():
            for performance in performances:
                if best_performance is None or performance.cmp(best_performance):
                    best_performance = performance
                    best_args, best_kwargs = self.hyper_parameters[hp_id]
        return best_args, best_kwargs

    def get_any_hyper_params(self):
        return self.get_hyper_parameters()

    def get_next(self):
        #TODO: This assumes random sampling for the moment
        return self.get_hyper_parameters()



class HyperParameterTrainer(object):
    def __init__(self, *, base_model, base_args, base_kwargs,
                 search_method='random'):
        self.base_model = base_model
        self.base_args = base_args
        self.base_kwargs = base_kwargs
        self.hp_manager = HyperParameterManager(base_args, base_kwargs,
                                                search_method=search_method)
        self.search_method = search_method

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train(self, n, **train_kwargs):
        try:
            for i in trange(n, desc='Hyper parameter'):
                    hp_id, args, kwargs = self.hp_manager.get_hyper_parameters()
                    model = self.base_model(*args, **kwargs)
                    performance, best_model_path = train(model=model,
                                                         **train_kwargs)
                    self.hp_manager.report(hp_id, performance)
        except StopIteration:
            return self.hp_manager.best_hyper_params()

    def get_best_hyper_params(self):
        return self.hp_manager.best_hyper_params()

    def get_any_hyper_params(self):
        return self.hp_manager.get_any_hyper_params()

    def get_next(self):
        pass


def add_parser_args(parser):
    """ Add common command line arguments used by the training function.
    """
    parser.add_argument('--output-dir',
                        help=("Directory to write output to."),
                        type=Path)
    parser.add_argument('--max-epochs', help="Maximum number of epochs to train for.", type=int, default=100)
    parser.add_argument('--eval-time', help="How often to run the model on the validation set in seconds.", type=float)
    parser.add_argument('--eval-epochs', help="How often to run the model on the validation set in epochs. 1 means at the end of every epoch.", type=int)
    parser.add_argument('--eval-iterations', help="How often to run the model on the validation set in number of training iterations.", type=int)

    parser.add_argument('--do-pre-eval', help="If flag is set, the model will be evaluated once before training starts",
                        action='store_true')
    parser.add_argument('--keep-snapshots', help="If flag is set, all snapshots will be kept. otherwise only the best and the latest are kept.",
                        action='store_true')

