import copy
import datetime
import types
from collections import defaultdict
from pathlib import Path
from collections.abc import Collection, Mapping
from typing import Callable, Any

from mltrain.train import train, TrainingConfig, TrainingArguments

from tqdm import trange, tqdm
import numpy as np


class HyperParameter(object):
    def __repr__(self):
        raise NotImplementedError()

    def random_sample(self, rng=None):
        ...


class IntegerRangeHyperParameter(HyperParameter):
    def __init__(self, low, high=None):
        super().__init__()
        if high is None:
            high = low
            low = 0
        self.low = low
        self.high = high
        #self.current_item = low  # We'll see how we implement grid search, if it's done with an iterator this variable
                                  # will not be needed
    def __repr__(self):
        return "<{} low:{},high:{}>".format(self.__class__.__name__, self.low, self.high)

    def random_sample(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        return rng.randint(self.low, self.high)


class DiscreteHyperParameter(HyperParameter):
    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.current_item = 0

    def random_sample(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        return rng.choice(self.values)

    def __repr__(self):
        return "<{} values:{}>".format(self.__class__.__name__, self.values)


class LinearHyperParameter(HyperParameter):
    def __init__(self, low, high=None, num=None):
        super().__init__()
        if high is None:
            high = low
            low = 0
        self.low = low
        self.high = high
        self.num = num

    def random_sample(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        return (self.high - self.low) * rng.random_sample() + self.low

    def __repr__(self):
        return "<{} low:{},high:{},num:{}>".format(self.__class__.__name__, self.low, self.high, self.num)


class GeometricHyperParameter(LinearHyperParameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = np.log10(self.low)
        self.high = np.log10(self.high)

    def random_sample(self, rng=None):
        # If we transform the space from low to high to a log-scale and then draw uniform samples in that space,
        # by exponentiating them we should get the right value
        sample = LinearHyperParameter.random_sample(self, rng=rng)
        return np.power(10, sample)


class MaterializeError(Exception):
    def __init__(self, obj, message):
        self.obj = obj
        self.message = message

    def __str__(self):
        return f"{self.message}: {self.obj}"


class HyperParameterManager(object):
    def __init__(self, base_obj, search_method='random', n=None, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.base_obj = base_obj
        self.search_method = search_method
        if search_method == 'random' and n is None:
            raise ValueError("When random search is specified, number of iterations must be set")
        self.n = n
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

    def materialize_hyper_parameters(self,
                                     obj):
        """Make any HyperParameter a concrete object"""
        non_collection_types = (str, bytes, bytearray, np.ndarray)
        try:
            if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
                return obj
            if isinstance(obj, HyperParameter):
                if self.search_method == 'random':
                    sample = obj.random_sample(self.rng)
                    return self.materialize_hyper_parameters(sample)
                else:
                    raise NotImplementedError('Search method {} is not implemented'.format(self.search_method))
            elif isinstance(obj, Mapping):
                return type(obj)({k: self.materialize_hyper_parameters(v) for k, v in obj.items()})
            elif isinstance(obj, Collection) and not isinstance(obj, non_collection_types):
                return type(obj)(self.materialize_hyper_parameters(x) for x in obj)
            elif hasattr(obj, '__dict__'):
                try:
                    obj_copy = copy.copy(obj)
                    obj_copy.__dict__ = self.materialize_hyper_parameters(obj.__dict__)
                    return obj_copy
                except TypeError:
                    return obj
            else:
                return obj
        except TypeError as e:
            raise MaterializeError(obj, "Failed to materialize") from e

    def get_hyper_parameters(self):
        materialized_obj = self.materialize_hyper_parameters(self.base_obj)
        self.n_iter += 1
        hp_id = self.n_iter  ## When we implement smarter search methods, this should be a reference to
                             # the hp-point produced
        self.hyper_parameters[hp_id] = materialized_obj
        return hp_id, materialized_obj

    def report(self, hp_id, performance, metadata=None):
        # The idea is that the manager can do things with this history. Since we will probably not have a lot of
        # samples, just having a flat structure works for now. The argument is that if you need to do smart HP
        # optimization, the cost of producing a sample is high, and you will be in a data limited regime. Having to
        # iterate over a list will be a small cost compared to evaluating each sample.
        self.history[hp_id].append((performance, metadata))

    def best_hyper_params(self):
        best_performance = None
        best_param = None
        best_metadata = None
        for hp_id, performances in self.history.items():
            for performance, metadata in performances:
                if best_performance is None or performance.cmp(best_performance):
                    best_performance = performance
                    best_params = self.hyper_parameters[hp_id]
                    best_metadata = metadata
        return best_params, metadata

    def get_any_hyper_params(self):
        return self.get_hyper_parameters()

    def get_next(self):
        #TODO: This assumes random sampling for the moment
        return self.get_hyper_parameters()

    def __iter__(self):
        if self.search_method == 'random':
            for i in range(self.n):
                yield self.get_hyper_parameters()

    def len(self):
        if self.search_method == 'random':
            return self.n


class HyperParameterTrainer(object):
    def __init__(self, *, setting_interpreter: Callable[[Any], TrainingArguments], hp_manager: HyperParameterManager):
        self.hp_manager = hp_manager
        self.settings_interpreter = setting_interpreter

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def train(self):
        for hp_id, settings in tqdm(self.hp_manager, 'Hyper parameters'):
            # We need to use the settings to prepare model, dataset and training arguments
            prepared_settings = self.settings_interpreter(settings)
            performance, best_model_path = train(training_args=prepared_settings)
            self.hp_manager.report(hp_id, performance, dict(best_model_path=best_model_path))

    def get_best_hyper_params(self):
        best_params, metadata = self.hp_manager.best_hyper_params()
        return best_params

    def get_any_hyper_params(self):
        return self.hp_manager.get_any_hyper_params()

    def best_model(self):
        best_params, metadata = self.hp_manager.best_hyper_params()
        return metadata['best_model_path']

    def best_hyper_params(self):
        best_params, metadata = self.hp_manager.best_hyper_params()
        return best_params


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

