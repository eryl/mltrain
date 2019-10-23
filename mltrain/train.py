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

from tqdm import trange, tqdm
import numpy as np


class JSONEncoder(json.JSONEncoder):
    "Custom JSONEncoder which tries to encode filed types (like pathlib Paths) as strings"
    def default(self, o):
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError:
            return str(o)


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
        self.values = list(sorted(values))
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
    def __init__(self, base_model, base_args, base_kwargs, search_method='random', search_iterations=None):
        self.base_model = base_model
        self.base_args = base_args
        self.base_kwargs = base_kwargs
        self.search_method = search_method
        self.search_iterations = search_iterations
        self.search_space = []
        self.history = []
        self.n_iter = 0
        if self.search_iterations is None and self.search_method == 'random':
            raise ValueError('If search method is random, you have to specify number of iterations')
        self.setup_search_space()

    def setup_search_space(self):
        for i, arg in enumerate(self.base_args):
            if isinstance(arg, HyperParameter):
                self.search_space.append((arg, 'args', i))
        for k, v in self.base_kwargs.items():
            if isinstance(v, HyperParameter):
                self.search_space.append((v, 'kwargs', k))

    def get_model(self):
        args = list(self.base_args)
        kwargs = dict(self.base_kwargs.items())
        hp_id = []

        if self.search_method == 'random':
            if self.n_iter >= self.search_iterations:
                raise StopIteration()
            for i, (hp, arg_type, arg_pos) in enumerate(self.search_space):
                # Here we might define other methods of sampling from the search space, for now we just do it randomly
                value = hp.random_sample()
                if arg_type == 'args':
                    args[arg_pos] = value
                else:
                    kwargs[arg_pos] = value
                hp_id.append((i, value))
        model = self.base_model(*args, **kwargs)
        return hp_id, model

    def report(self, hp_id, performance):
        # The idea is that the manager can do things with this history. Since we will probably not have a lot of
        # samples, just having a flat structure works for now. The argument is that if you need to do smart HP
        # optimization, the cost of producing a sample is high, and you will be in a data limited regime. Having to
        # iterate over a list will be a small cost compared to evaluating each sample.
        self.history.append((hp_id, performance))


def hyper_parameter_train(*, base_model, base_args, base_kwargs, search_method='random',
                          search_iterations=None, **train_kwargs):
    # Figure out what args are Hyper Parameter configurations
    hp_manager = HyperParameterManager(base_model, base_args, base_kwargs,
                                       search_method=search_method, search_iterations=search_iterations)
    best_model_path = None
    try:
        while True:
            with tqdm(desc='Hyper parameter') as pbar:
                hp_id, model = hp_manager.get_model()
                performance, best_model_path = train(model=model, **train_kwargs)
                hp_manager.report(hp_id, performance)
                pbar.update()
    except StopIteration:
        return best_model_path


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
          do_pre_eval=False,
          evaluation_metrics=('accuracy', 'loss'),
          **kwargs):
    evaluation_metrics, best_metrics, model_format_string, output_dir = setup_training(model=model,
                                                                           output_dir=output_dir,
                                                                           metadata=metadata,
                                                                           keep_snapshots=keep_snapshots,
                                                                           eval_time=eval_time,
                                                                           eval_iterations=eval_iterations,
                                                                           eval_epochs=eval_epochs,
                                                                           checkpoint_suffix=checkpoint_suffix,
                                                                           model_format_string=model_format_string,
                                                                           evaluation_metrics=evaluation_metrics)
    with Monitor(output_dir / 'logs') as monitor:
        best_metrics, best_model_path = training_loop(model,
                                        training_dataset,
                                        evaluation_dataset,
                                        max_epochs,
                                        monitor,
                                        evaluation_metrics,
                                        best_metrics,
                                        model_format_string,
                                        eval_time=eval_time,
                                        eval_iterations=eval_iterations,
                                        eval_epochs=eval_epochs,
                                        do_pre_eval=do_pre_eval,
                                        keep_snapshots=keep_snapshots)
        return best_metrics, best_model_path


def setup_training(*,
          model,
          output_dir: Path,
          metadata=None,
          keep_snapshots=False,
          eval_time=None,
          eval_iterations=None,
          eval_epochs=1,
          checkpoint_suffix='.pkl',
          model_format_string=None,
          evaluation_metrics=('accuracy', 'loss')):
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
        print("Model parameters are: ")
        print('\n'.join(list(sorted('{}: {}'.format(k, v) for k, v in model_metadata.items()))))
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

    evaluation_metrics, best_metrics = setup_metrics(evaluation_metrics)
    return evaluation_metrics, best_metrics, model_format_string, output_dir


def setup_metrics(evaluation_metrics):
    fixed_evaluation_metrics = []
    best_metrics = dict()
    for evaluation_metric in evaluation_metrics:
        try:
            evaluation_metric, comparator = evaluation_metric
        except ValueError:
            if 'loss' in evaluation_metric:
                comparator = operator.le
            elif 'accuracy' in evaluation_metric:
                comparator = operator.ge
            else:
                raise RuntimeError(
                    "We don't know how to compare metrics {}. Please supply comparator.".format(evaluation_metric))
        fixed_evaluation_metrics.append((evaluation_metric, comparator))
        # The metrics needs to be scalar for this to work
        pos_inf = np.inf
        neg_inf = -pos_inf
        if comparator(pos_inf, neg_inf):
            best_metrics[evaluation_metric] = neg_inf
        else:
            best_metrics[evaluation_metric] = pos_inf
    return fixed_evaluation_metrics, best_metrics


def training_loop(model,
                  training_dataset,
                  evaluation_dataset,
                  max_epochs,
                  monitor,
                  evaluation_metrics,
                  best_metrics,
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
                       evaluation_metrics=evaluation_metrics,
                       model_checkpoint_format=model_checkpoint_format,
                       monitor=monitor,
                       keep_snapshots=keep_snapshots)
    # These variables will be used to control when to do evaluation
    eval_timestamp = time.time()
    eval_epoch = 0
    eval_iteration = 0

    if do_pre_eval:
        best_metrics, best_model_path = evaluate_model(best_metrics=best_metrics, epoch=0, **eval_kwargs)

    for epoch in trange(max_epochs, desc='Epochs'):
        ## This is the main training loop
        for i, batch in enumerate(tqdm(training_dataset, desc='Training batch')):
            epoch_fraction = epoch + i / len(training_dataset)
            training_results = model.fit(batch)
            monitor.log_one_now('epoch', epoch_fraction)
            monitor.log_now(training_results)

            # eval_time and eval_iterations allow the user to control how often to run evaluations
            eval_time_dt = time.time() - eval_timestamp
            eval_iteration += 1

            if ((eval_time is not None and eval_time > 0 and eval_time_dt >= eval_time) or
                (eval_iterations is not None and eval_iterations > 0 and eval_iteration >= eval_iterations)):
                best_metrics, best_model_path = evaluate_model(best_metrics=best_metrics, epoch=epoch_fraction, **eval_kwargs)
                eval_timestamp = time.time()
                eval_iteration = 0

            monitor.tick()
            # End of training loop

        eval_epoch += 1
        if (eval_epochs is not None and eval_epochs > 0 and eval_epoch >= eval_epochs):
            best_metrics, best_model_path = evaluate_model(best_metrics=best_metrics, epoch=epoch + 1, **eval_kwargs)
            eval_epoch = 0
        # End of epoch

    # Done with the whole training loop
    best_metrics, best_model_path = evaluate_model(best_metrics=best_metrics, epoch=epoch, **eval_kwargs)
    return best_metrics, best_model_path


def evaluate_model(*,
                   model,
                   evaluation_dataset,
                   evaluation_metrics,
                   best_metrics,
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

    printout_metrics = {}
    # For the model to be the new best, it should be at least as good as the old model on all evaluation metrics
    # The problem is how to compare metrics, e.g. lower loss is better while higher accuracy is better
    comparisons = []
    for evaluation_metric, comparator in evaluation_metrics:
        # We prune the evaluation metrics to only include those which have matching values in the evaluation results
        for evaluation_result_key, evaluation_result_mean in evaluation_results.items():
            if evaluation_metric in evaluation_result_key:
                printout_metrics[evaluation_metric] = evaluation_result_mean
                previous_best = best_metrics[evaluation_metric]
                comparisons.append(comparator(evaluation_result_mean, previous_best))

    is_best = np.all(comparisons)
    if monitor is not None:
        monitor.log_now({k: v for k,v in evaluation_results.items()})

    best_model_path = checkpoint(model, model_checkpoint_format, epoch,
                                 printout_metrics, is_best, remove_models=not keep_snapshots)
    if is_best:
        best_metrics = printout_metrics
    return best_metrics, best_model_path


def setup_directory(output_dir: Path):
    # Create directory and set up symlinks if it doesn't already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = output_dir.parent
    symlink_name = parent_dir / 'latest_experiment'
    if symlink_name.is_symlink() or symlink_name.exists():
        symlink_name.unlink()
    symlink_name.symlink_to(output_dir)


def checkpoint(model,
               checkpoint_format: Path,
               epoch,
               metrics,
               is_best, latest_model_name='latest_model', best_model_name='best_model',
               remove_models=True):
    model_directory = checkpoint_format.parent
    metrics_string = '_'.join(['{}:{:.03f}'.format(k,v) for k,v in metrics.items()])
    model_name = checkpoint_format.name.format(epoch=epoch, metrics=metrics_string)
    checkpoint_path = checkpoint_format.with_name(model_name)
    model_directory.mkdir(exist_ok=True)
    model.save(checkpoint_path)
    model_suffix = checkpoint_path.suffix
    latest_model_symlink = (model_directory / latest_model_name).with_suffix(model_suffix)
    best_model_symlink = (model_directory / best_model_name).with_suffix(model_suffix)

    if remove_models and latest_model_symlink.exists():
        latest_model = latest_model_symlink.resolve(strict=True)
        if not best_model_symlink.exists() or latest_model != best_model_symlink.resolve(strict=True):
                latest_model.unlink()

    if latest_model_symlink.exists() or latest_model_symlink.is_symlink():
        latest_model_symlink.unlink()
    latest_model_symlink.symlink_to(checkpoint_path)

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
        best_model_symlink.symlink_to(checkpoint_path)

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

