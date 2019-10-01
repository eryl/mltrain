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
          eval_epochs=None,
          checkpoint_suffix='.pkl',
          model_format_string=None,
          do_pre_eval=False,
          evaluation_metrics=('accuracy', 'loss'),
          **kwargs):

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
        print('\n'.join(list(sorted('{}: {}'.format(k,v) for k,v in model_metadata.items()))))
    except AttributeError:
        print("Couldn't get model parameters, skipping model_params for the metadata")
        raise

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
    with Monitor(output_dir / 'logs') as monitor:
        training_loop(model,
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
    def sigint_handler(signal, frame):
        checkpoint(model, model_checkpoint_format, np.nan, [], is_best=False, remove_models=False)
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
        best_metrics = evaluate_model(best_metrics=best_metrics, epoch=0, **eval_kwargs)

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
                best_metrics = evaluate_model(best_metrics=best_metrics, epoch=epoch_fraction, **eval_kwargs)
                eval_timestamp = time.time()
                eval_iteration = 0

            monitor.tick()
            # End of training loop

        eval_epoch += 1
        if (eval_epochs is not None and eval_epochs > 0 and eval_epoch >= eval_epochs):
            best_metrics = evaluate_model(best_metrics=best_metrics, epoch=epoch + 1, **eval_kwargs)
            eval_epoch = 0
        # End of epoch

    # Done with the whole training loop
    best_metrics = evaluate_model(best_metrics=best_metrics, epoch=epoch, **eval_kwargs)


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

    printout_metrics = []
    # For the model to be the new best, it should be at least as good as the old model on all evaluation metrics
    # The problem is how to compare metrics, e.g. lower loss is better while higher accuracy is better
    comparisons = []
    for evaluation_metric, comparator in evaluation_metrics:
        printout_metrics.append((evaluation_metric, evaluation_results[evaluation_metric]))
        new_value = evaluation_results[evaluation_metric]
        previous_best = best_metrics[evaluation_metric]
        comparisons.append(comparator(new_value, previous_best))

    is_best = np.all(comparisons)
    if monitor is not None:
        monitor.log_now({'evaluation ' + k: v for k,v in evaluation_results.items()})

    checkpoint(model, model_checkpoint_format, epoch, printout_metrics, is_best, remove_models=not keep_snapshots)
    if is_best:
        best_metrics = printout_metrics
    return best_metrics


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
    metrics_string = '_'.join(['{}:{:.03f}'.format(k,v) for k,v in metrics])
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
        print("Flushing cache for channel {}".format(channel_name))
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
        print("Done flushing cache")

    def close(self):
        for channel_name, channel_file in self.channel_files.items():
            channel_file.close()


def add_training_arguments(parser):
    """ Add common command line arguments used by the training function.
    """
    parser.add_argument('--output-dir',
                        help=("Directory to write output to."))
    parser.add_argument('--max-epochs', help="Maximum number of epochs to train for.", type=int, default=100)
    parser.add_argument('--eval-time', help="How often to run the model on the validation set in seconds.", type=float)
    parser.add_argument('--eval-epochs', help="How often to run the model on the validation set in epochs. 1 means at the end of every epoch.", type=int)
    parser.add_argument('--eval-iterations', help="How often to run the model on the validation set in number of training iterations.", type=int)

    parser.add_argument('--do-pre-eval', help="If flag is set, the model will be evaluated once before training starts",
                        action='store_true')
    parser.add_argument('--keep-snapshots', help="If flag is set, all snapshots will be kept. otherwise only the best and the latest are kept.",
                        action='store_true')

