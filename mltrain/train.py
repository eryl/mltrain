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

from tqdm import trange, tqdm
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, batch):
        pass

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def evaluation_metrics(self):
        pass

    @abstractmethod
    def evaluate(self, batch):
        pass

    @abstractmethod
    def save(self, save_path):
        pass


class JSONEncoder(json.JSONEncoder):
    "Custom JSONEncoder which tries to encode filed types (like pathlib Paths) as strings"
    def default(self, o):
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError:
            return str(o)


class TrainingError(Exception):
    def __init__(self, metadata, message):
        self.metadata = metadata
        self.message = message

    def __str__(self):
        return f"{self.message}\n Metadata was: {self.metadata}"



def run_experiments(num_experiments, experiment_kwargs_list, experiment_function, kwargs, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    total_num_experiments = len(experiment_kwargs_list)*num_experiments
    try:
        current_experiment = 0
        for i in range(num_experiments):
            for task_params in experiment_kwargs_list:
                current_experiment += 1
                print('Starting experiment {}/{}'.format(current_experiment, total_num_experiments))
                print('Starting task with params {}'.format(task_params))
                kwargs.update(task_params)
                kwargs['random_seed'] = rng.randint(0, 2**32)
                kwargs['experiment_function'] = experiment_function
                p = multiprocessing.Process(target=worker, kwargs=kwargs)
                p.start()
                p.join()
                if p.exitcode != 0:
                    # For now we assume that the process died because of out of memory exceptions
                    print("Process died with exit code 1.")
                    sys.exit(0)

    except KeyboardInterrupt:
        pass


def worker(*, experiment_function, device='cpu', backend='theano', **kwargs):
    if not device == 'cpu':
        if backend == 'pytorch':
            kwargs['device'] = device
        elif backend == 'theano':
            print("Setting device {}".format(device))
            import pygpu.gpuarray
            import theano.gpuarray
            theano.gpuarray.use(device)

    print("Starting new training with parameters:")
    metadata = dict()
    metadata['command_line_params'] = kwargs
    for param, value in sorted(kwargs.items()):
        print("  {}: {}".format(param, value))
    experiment_function(metadata=metadata, backend=backend, **kwargs)


def make_timestamp():
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H.%M.%S")  # We choose this format to make the filename compatible with windows environmnets


def run_experiment(*, hyper_parameters=None, model_factory=None, **kwargs):
    if hyper_parameters is not None:
        for hp in hyper_parameters:
            model = model_factory(hyper_parameters)
            train(model=model, **kwargs)


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

    def materialize_hyper_params(self, obj):
        """Make any HyperParameter a concrete object"""
        if isinstance(obj, Mapping):
            return type(obj)((k, self.materialize_hyper_params(v)) for k, v in obj.items())
        elif isinstance(obj, Collection) and not isinstance(obj, (str, bytes, bytearray, np.ndarray)):
            return type(obj)(self.materialize_hyper_params(x) for x in obj)
        elif isinstance(obj, HyperParameter):
            if self.search_method == 'random':
                return obj.random_sample()
            else:
                raise NotImplementedError('Search method {} is not implemented'.format(self.search_method))
        else:
            return obj

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

def train(*,
          model,
          training_dataset,
          evaluation_dataset,
          max_epochs,
          output_dir: Path,
          metadata=None,
          keep_snapshots=False,
          eval_time=None,
          eval_iterations=None,
          eval_epochs=1,
          checkpoint_suffix='.pkl',
          model_format_string=None,
          do_pre_eval=False):
    best_performance, model_format_string, output_dir = setup_training(model=model,
                                                                       output_dir=output_dir,
                                                                       metadata=metadata,
                                                                       keep_snapshots=keep_snapshots,
                                                                       eval_time=eval_time,
                                                                       eval_iterations=eval_iterations,
                                                                       eval_epochs=eval_epochs,
                                                                       checkpoint_suffix=checkpoint_suffix,
                                                                       model_format_string=model_format_string)
    try:
        with Monitor(output_dir / 'logs') as monitor:
            best_performance, best_model_path = training_loop(model=model,
                                                              training_dataset=training_dataset,
                                                              evaluation_dataset=evaluation_dataset,
                                                              max_epochs=max_epochs,
                                                              monitor=monitor,
                                                              best_performance=best_performance,
                                                              model_checkpoint_format=model_format_string,
                                                              eval_time=eval_time,
                                                              eval_iterations=eval_iterations,
                                                              eval_epochs=eval_epochs,
                                                              do_pre_eval=do_pre_eval,
                                                              keep_snapshots=keep_snapshots)
            return best_performance, best_model_path
    except Exception as e:
        raise TrainingError(metadata, "Error during training") from e

def setup_training(*,
          model,
          output_dir: Path,
          metadata=None,
          keep_snapshots=False,
          eval_time=None,
          eval_iterations=None,
          eval_epochs=1,
          checkpoint_suffix='.pkl',
          model_format_string=None):
    if model_format_string is None:
        model_format_string = model.__class__.__name__ + '_epoch-{epoch:.04f}_{metrics}' + checkpoint_suffix

    output_dir = output_dir / make_timestamp()
    while output_dir.exists():
        time.sleep(1)
        output_dir = output_dir / make_timestamp()
    model_format_string = output_dir / model_format_string

    setup_directory(output_dir)

    if metadata is None:
        metadata = dict()
    try:
        model_metadata = model.get_metadata()
        metadata['model_metadata'] = model_metadata
        #print("Model parameters are: ")
        #print('\n'.join(list(sorted('{}: {}'.format(k, v) for k, v in model_metadata.items()))))
    except AttributeError:
        print("Couldn't get model parameters, skipping model_params for the metadata")

    training_params = dict(eval_time=eval_time,
                           eval_iterations=eval_iterations,
                           eval_epochs=eval_epochs,
                           model_format_string=model_format_string,
                           output_dir=output_dir,
                           keep_snapshots=keep_snapshots)
    metadata['training_params'] = training_params

    json_encoder = JSONEncoder(sort_keys=True, indent=4, separators=(',', ': '))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as metadata_fp:
        metadata_fp.write(json_encoder.encode(metadata))

    best_performance = setup_metrics(model.evaluation_metrics())
    return best_performance, model_format_string, output_dir


class EvaluationMetric(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def cmp(self, a, b):
        raise NotImplementedError()


class HigherIsBetterMetric(EvaluationMetric):
    def __init__(self, name):
        super().__init__(name)
        self.worst_value = -np.inf

    def cmp(self, a, b):
        return a > b


class LowerIsBetterMetric(EvaluationMetric):
    def __init__(self, name):
        super().__init__(name)
        self.worst_value = np.inf

    def cmp(self, a, b):
        return a < b


class Performance(object):
    def __init__(self, metric, value=None):
        self.metric = metric
        if value is None:
            value = metric.worst_value
        self.value = value

    def cmp(self, other):
        return self.metric.cmp(self.value, other.value)

    def __str__(self):
        return str(self.value)


class PerformanceCollection(object):
    def __init__(self, performances):
        self.metrics = [p.metric for p in performances] # Keeps the order of the performance objects
        self.performances = { p.metric:p for p in performances}

    def cmp(self, other):
        for metric in self.metrics:
            performance = self.performances[metric]
            other_performance = other.get_performance(metric)
            if performance.cmp(other_performance):
                # This performance is better than the other
                return True
            elif other_performance.cmp(performance):
                # This performance is equal to the other, we need to look at the next metric
                return False
            else:
                # This performance is worse than the other
                continue
        return True  # If two performance collections are exactly the same, we return this one as the better

    def get_performance(self, metric):
        return self.performances[metric]

    def update(self, value_map):
        """
        Return a new PerformanceCollection where the metrics are updated according to the
        values in the value map
        :param value_map:
        :return:
        """
        performances = [Performance(metric, value_map[metric.name]) for metric in self.metrics]
        return PerformanceCollection(performances)

    def get_metrics(self):
        return self.metrics

    def __str__(self):
        return '_'.join('{}:{}'.format(metric.name, self.performances[metric].value) for metric in self.metrics)

    def items(self):
        yield from self.performances.items()

def setup_metrics(evaluation_metrics):
    base_performances = []
    for evaluation_metric in evaluation_metrics:
        if isinstance(evaluation_metric, str):
            if 'loss' in evaluation_metric:
                evaluation_metric = LowerIsBetterMetric(evaluation_metric)
            elif 'accuracy' in evaluation_metric:
                evaluation_metric = HigherIsBetterMetric(evaluation_metric)
            else:
                raise RuntimeError(
                    "Metric {} is not implemented, please supply an EvaluationMetric object instead.".format(evaluation_metric))
        if isinstance(evaluation_metric, EvaluationMetric):
            base_performance = Performance(evaluation_metric)
            base_performances.append(base_performance)
    best_performance = PerformanceCollection(base_performances)
    return best_performance


def training_loop(*,
                  model,
                  training_dataset,
                  evaluation_dataset,
                  max_epochs,
                  monitor,
                  best_performance,
                  model_checkpoint_format,
                  eval_time=None,
                  eval_iterations=None,
                  eval_epochs=1,
                  do_pre_eval=True,
                  keep_snapshots=False):

    epoch = 0
    best_model_path = None

    def sigint_handler(signal, frame):
        checkpoint(model, model_checkpoint_format, np.nan, {}, is_best=False, remove_models=False)
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    # Since we call evaluate_models from som many places below, we summarize the common arguments in a dict
    eval_kwargs = dict(model=model,
                       evaluation_dataset=evaluation_dataset,
                       model_checkpoint_format=model_checkpoint_format,
                       monitor=monitor,
                       keep_snapshots=keep_snapshots)
    # These variables will be used to control when to do evaluation
    eval_timestamp = time.time()
    eval_epoch = 0
    eval_iteration = 0
    needs_final_eval = True

    if do_pre_eval:
        best_metrics, best_model_path = evaluate_model(best_performance=best_performance, epoch=0, **eval_kwargs)

    for epoch in trange(max_epochs, desc='Epochs'):
        ## This is the main training loop
        for i, batch in enumerate(tqdm(training_dataset, desc='Training batch')):
            needs_final_eval = True
            epoch_fraction = epoch + i / len(training_dataset)
            training_results = model.fit(batch)
            monitor.log_one_now('epoch', epoch_fraction)
            if training_results is not None:
                monitor.log_now(training_results)

            # eval_time and eval_iterations allow the user to control how often to run evaluations
            eval_time_dt = time.time() - eval_timestamp
            eval_iteration += 1

            if ((eval_time is not None and eval_time > 0 and eval_time_dt >= eval_time) or
                (eval_iterations is not None and eval_iterations > 0 and eval_iteration >= eval_iterations)):
                best_metrics, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch_fraction, **eval_kwargs)
                eval_timestamp = time.time()
                eval_iteration = 0
                needs_final_eval = False

            monitor.tick()
            # End of training loop

        eval_epoch += 1
        if (eval_epochs is not None and eval_epochs > 0 and eval_epoch >= eval_epochs):
            best_metrics, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch, **eval_kwargs)
            eval_epoch = 0
            needs_final_eval = False
        # End of epoch

    # Done with the whole training loop. If we ran the evaluate_model at the end of the last epoch, we shouldn't do
    # it again
    if needs_final_eval:
        best_metrics, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch, **eval_kwargs)
    return best_metrics, best_model_path


def evaluate_model(*,
                   model,
                   evaluation_dataset,
                   best_performance,
                   model_checkpoint_format,
                   epoch,
                   monitor=None,
                   keep_snapshots=False):
    gathered_evaluation_results = defaultdict(list)
    for batch in evaluation_dataset:
        for k, v in model.evaluate(batch).items():
            gathered_evaluation_results[k].append(v)
    evaluation_results = {}
    for k, v in gathered_evaluation_results.items():
        if v:
            try:
                evaluation_results[k] = np.mean(v)
            except TypeError:
                print("Not logging result {}, can't aggregate data type".format(k))
    new_performance = best_performance.update(evaluation_results)

    is_best = new_performance.cmp(best_performance)

    if monitor is not None:
        monitor.log_now({k: v for k,v in evaluation_results.items()})

    best_model_path = checkpoint(model, model_checkpoint_format, epoch,
                                 new_performance, is_best, remove_models=not keep_snapshots)
    if is_best:
        best_performance = new_performance
        if monitor is not None:
            monitor.log_now({'best_{}'.format(k):v for k,v in best_performance.items()})
        best_performance_file = model_checkpoint_format.with_name('best_performance.csv')
        with open(best_performance_file, 'w') as fp:
            items = [(k.name, v) for k,v in best_performance.items()]
            keys, vals = zip(*sorted(items))
            fp.write(','.join(str(k) for k in keys) + '\n')
            fp.write(','.join(str(v) for v in vals) + '\n')
    return best_performance, best_model_path


def setup_directory(output_dir: Path):
    # Create directory and set up symlinks if it doesn't already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = output_dir.parent
    symlink_name = parent_dir / 'latest_experiment'
    if symlink_name.is_symlink() or symlink_name.exists():
        symlink_name.unlink()

    symlink_name.symlink_to(output_dir.relative_to(parent_dir))


def checkpoint(model,
               checkpoint_format: Path,
               epoch,
               performances,
               is_best, latest_model_name='latest_model', best_model_name='best_model',
               remove_models=True):
    model_directory = checkpoint_format.parent.resolve(strict=False)
    model_name = checkpoint_format.name.format(epoch=epoch, metrics=performances)
    checkpoint_path = checkpoint_format.with_name(model_name).resolve()
    model_directory.mkdir(exist_ok=True)
    model.save(checkpoint_path)
    model_suffix = checkpoint_path.suffix
    latest_model_symlink = (model_directory / latest_model_name).with_suffix(model_suffix)
    best_model_symlink = (model_directory / best_model_name).with_suffix(model_suffix)

    if remove_models and latest_model_symlink.exists():
        latest_model = latest_model_symlink.resolve(strict=True)
        if not best_model_symlink.exists() or latest_model != best_model_symlink.resolve(strict=True):
            latest_model.unlink()

    if os.path.lexists(latest_model_symlink):
        latest_model_symlink.unlink()

    relative_checkpoint = checkpoint_path.relative_to(latest_model_symlink.absolute().parent)
    latest_model_symlink.symlink_to(relative_checkpoint)

    if is_best:
        # Path.exists() on a symlink will return True if what the symlink points to exists, not if the symlink exists
        # To check if the symlink exist, we call is_symlink() as well a exists(), if the path is a symlink, is_symlink()
        # will only return true if it exists, if either returns True, than the file exists and we should remove it
        # whether it's a symlink or not
        if best_model_symlink.exists():
            if remove_models:
                # The previous best model can't also be the latest model since we take care of that above, so it's safe
                # to remove
                previous_best_model = best_model_symlink.resolve(strict=True)
                previous_best_model.unlink()
        if best_model_symlink.is_symlink():
            best_model_symlink.unlink()
        relative_checkpoint = checkpoint_path.relative_to(best_model_symlink.absolute().parent)
        best_model_symlink.symlink_to(relative_checkpoint)

    return best_model_symlink.resolve()


class Monitor(object):
    def __init__(self, monitor_dir, save_interval=20, buffer_size=100):
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.channel_values = multiprocessing.Queue(maxsize=100)
        self.save_interval = save_interval
        self.buffer_size = buffer_size
        self.monitor_process = MonitorProcess(monitor_dir,
                                              self.channel_values,
                                              buffer_size=buffer_size,
                                              save_interval=save_interval)
        self.monitor_process.start()
        signal.signal(signal.SIGINT, original_sigint_handler)
        self.time = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor_process.exit.set()
        print("Waiting for monitor process to exit cleanly")
        self.monitor_process.join()
        print("Monitor exiting")

    def tick(self):
        """
        Progress the time one step.
        """
        self.time += 1

    def log_now(self, values):
        for channel_name, value in values.items():
            update_command = (self.time, channel_name, value)
            self.channel_values.put(update_command)

    def log_one_now(self, channel, value):
        update_command = (self.time, channel, value)
        self.channel_values.put(update_command)


class MonitorProcess(multiprocessing.Process):
    def __init__(self, store_directory, command_queue, *args, buffer_size=100, save_interval=None, compress_log=False, **kwargs):
        super(MonitorProcess, self).__init__(*args, **kwargs)
        self.store_directory = store_directory
        if not os.path.exists(store_directory):
            os.makedirs(store_directory)
        self.channel_files = dict()
        self.command_queue = command_queue
        self.buffer_size = buffer_size
        self.channels = defaultdict(list)
        self.save_interval = save_interval
        self.compress_log = compress_log
        if save_interval is not None:
            self.tm1 = time.time()
        self.exit = multiprocessing.Event()

    def run(self):
        while not self.exit.is_set():
            try:
                command = self.command_queue.get(False, 10)
                self.update_channel(command)
            except queue.Empty:
                pass
            if self.save_interval is not None and time.time() - self.tm1 < self.save_interval:
                self.flush_caches()
                self.tm1 = time.time()
        # If exit is set, still empty the queue before quitting
        while True:
            try:
                command = self.command_queue.get(False)
                self.update_channel(command)
            except queue.Empty:
                break
        self.flush_caches()
        print("Monitor process is exiting")
        self.close()

    def update_channel(self, command):
        t, channel_name, channel_value = command
        self.channels[channel_name].append((t, channel_value))
        if len(self.channels[channel_name]) >= self.buffer_size:
            self.flush_cache(channel_name)

    def flush_caches(self):
        for channel_name in self.channels.keys():
            self.flush_cache(channel_name)

    def flush_cache(self, channel_name):
        #print("Flushing cache for channel {}".format(channel_name))
        if len(self.channels[channel_name]) > 0:
            if channel_name not in self.channel_files:
                channel_file_name = os.path.join(self.store_directory, channel_name + '.txt')
                if self.compress_log:
                    channel_file_name += '.gz'
                    channel_file = gzip.open(channel_file_name, 'w')
                else:
                    channel_file = open(channel_file_name, 'w')
                self.channel_files[channel_name] = channel_file
            else:
                channel_file = self.channel_files[channel_name]
            data = ''.join(['{} {}\n'.format(time, value) for time, value in self.channels[channel_name]])
            channel_file.write(data)
            channel_file.flush()
            self.channels[channel_name].clear()
        #print("Done flushing cache")

    def close(self):
        for channel_name, channel_file in self.channel_files.items():
            channel_file.close()


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

