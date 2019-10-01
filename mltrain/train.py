import json
import time
import sys
import os
import os.path
import pickle
import itertools
import unittest
import signal
from collections import defaultdict

import numpy as np

from ylipy.monitor import Monitor
from ylipy.reporting import ProgressReport
from ylipy.time import timestamp

import deepsics.paramsearch.params
import multiprocessing

class NumericError(BaseException):
    pass


class JSONEncoder(json.JSONEncoder):
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


def train_on_iterators(model,
                       training_iterator,
                       test_iterator,
                       stopping_criterion,
                       training_iterations,
                       validation_iterations,
                       model_format_string=None,
                       plot_title=None,
                       plot_description=None,
                       output_dir=None,
                       iterator_mode=None,
                       metadata=None,
                       **kwargs):

    def make_iterator(base_iterator, num_iterations):
        def iterator():
            yield from itertools.islice(base_iterator, num_iterations)
        return iterator

    training_iterator_fun = make_iterator(training_iterator, training_iterations)
    test_iterator_fun = make_iterator(test_iterator, validation_iterations)

    train_with_meta_iterator(model, training_iterator_fun, test_iterator_fun, stopping_criterion=stopping_criterion,
                             model_format_string=model_format_string, plot_title=plot_title,
                             plot_description=plot_description,
                             output_dir=output_dir,
                             validation_iterations=validation_iterations,
                             training_iterations=training_iterations,
                             metadata=metadata, **kwargs)

def train_with_meta_iterator(model,
                             training_iterator_fun,
                             validation_iterator_fun,
                             max_epochs,
                             stopping_criterion=None,
                             training_function=None,
                             validation_function=None,
                             eval_time=None,
                             eval_iterations=None,
                             eval_epochs=None,
                             model_format_string=None,
                             plot_title=None,
                             plot_description=None,
                             plot_max_range=None,
                             output_dir=None,
                             training_iterations=0,
                             validation_iterations=0,
                             metadata=None,
                             keep_snapshots=False,
                             live_plot=True,
                             which_plots=None,
                             learning_rate=0.001,
                             lr_policy='decay',
                             lr_policy_params=None,
                             numeric_abort=True,
                             no_pre_eval=False,
                             monitor_save_interval=None,
                             monitor_storage_mode='directory',
                             **kwargs):
    """
    Performs training on the given model, generating batches using the given functions.
    :param model: The model to train. Needs to have a train and evaluate function.
    :param training_iterator_fun: A function which returns an iterator over training batches. The batches need to be
                                  accepted by the train method of the model.
    :param validation_iterator_fun: A function which returns an iterator over validation batches. The batches need to be
                              accepted by the evaluate method of the model.
    :param stopping_criterion: Maximum number of epochs (complete iterations over the iterator returned
                       from training_iterator_fun) to perform.
    :param training_function: A function or name of the model method to use for training.
                              The value returned from each call to the training
                              iterator will be given as the argument of this function.
                              If None, the function 'train' of the model object will be used.
    :param validation_function: A function to the model method to use for evaluation.
                                The value returned from each call to the validation
                                iterator will be passed to this function. If None,
                                the function 'evaluate' of the model
                                object will be used.
    :param eval_every: How often in seconds to evaluate the model on the validation data.
    :param model_format_string: A format string to use for saving the model. The string should have two format
                                specifiers, one for the variable 'epoch' and one for 'test_cost'.
    :param plot_title: Title for the interactive training plot.
    :param plot_description: Extra text to display on the training plot.
    :param output_dir: Directory to save models to.
    :param training_iterations: How many batches the training_iterator_fun iterator has. This is only used for
                                giving an estimate of training progress and is not used to determine how many
                                iterations to do.
    :param validation_iterations: How many batches the training_iterator_fun iterator has. This is only used for
                            giving an estimate of training progress and is not used to determine how many
                            iterations to do.
    :param metadata: If given as a dictionary, this will be saved to a JSON file in the output directory. This can be
                     used to give metadata about the trained model (such as parameters for the training).
    :param keep_snapshots: If True, all snapshots are kept. If False, only the snapshot for the best and latest model
                           are kept.
    :param live_plot: If True, a graphical plot will show the training progress. If False, a plot will still be generated
                      but won't be created as a graphical window.
    :param kwargs:
    :return:
    """
    if output_dir is None:
        output_dir = ''
    if model_format_string is None:
        model_format_string = model.__class__.__name__ + '_epoch-{epoch:.04f}_model_test-error-{test_cost:.04f}-test_accuracy-{test_accuracy:.04f}'

    output_string = ("{progress} (epoch {epoch:.03f})\n"
                     "{results}\n"
                     "time/batch = {time}. Epoch progress: ")

    output_dir = os.path.abspath(os.path.join(output_dir, timestamp()))
    while os.path.exists(output_dir):
        time.sleep(1)
        output_dir = os.path.abspath(os.path.join(output_dir, timestamp()))
    model_format_string = os.path.join(output_dir, model_format_string)
    total_iterations = max_epochs*training_iterations
    best_test_cost = float('inf')

    setup_directory(output_dir)

    if metadata is None:
        metadata = dict()

    try:
        model_params = model.get_init_params()
        metadata['model_params'] = model_params
        print("Model parameters are: ")
        print('\n'.join(list(sorted('{}: {}'.format(k,v) for k,v in model_params.items()))))
    except AttributeError:
        print("Couldn't get model parameters, skipping model_params for the metadata")
        raise

    training_params = dict(eval_time=eval_time,
                           eval_iterations=eval_iterations,
                           eval_epochs=eval_epochs,
                           model_format_string=model_format_string,
                           plot_title=plot_title,
                           plot_description=plot_description,
                           output_dir=output_dir,
                           training_iterations=training_iterations,
                           test_iterations=validation_iterations,
                           keep_snapshots=keep_snapshots,
                           live_plot=live_plot,
                           learning_rate=learning_rate,
                           lr_policy=lr_policy,
                           lr_policy_params=lr_policy_params,
                           )
    metadata['training_params'] = training_params

    json_encoder = JSONEncoder(sort_keys=True, indent=4, separators=(',', ': '))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as metadata_fp:
        metadata_fp.write(json_encoder.encode(metadata))

    if training_function is None:
        training_function = model.train
    elif isinstance(training_function, str):
        training_function = getattr(model, training_function)

    if validation_function is None:
        validation_function = model.evaluate
    elif isinstance(validation_function, str):
        validation_function = getattr(model, validation_function)

    monitor_store = os.path.join(output_dir, 'monitor.hdf5')


    def sigint_handler(signal, frame):
        model_path = model_format_string.format(epoch=np.nan,
                                                test_cost=np.nan,
                                                test_accuracy=np.nan)
        save_snapshot(model, model_filename=model_path, is_best=False, keep_snapshots=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    with Monitor(monitor_dir=monitor_store,
                 save_interval=monitor_save_interval) as monitor:
        # These variables will be used to control when to do evaluation
        eval_timestamp = None
        eval_epoch = 0
        eval_iteration = 0

        if not no_pre_eval:
            best_test_cost = evaluate_model(model,
                                            validation_iterator_fun,
                                            validation_function,
                                            best_test_cost,
                                            model_format_string,
                                            epoch=0,
                                            monitor=monitor,
                                            validation_iterations=validation_iterations,
                                            keep_snapshots=keep_snapshots)

        for epoch in range(max_epochs):
            reporter = ProgressReport(training_iterations, report_msg=output_string)
            gradient_norms = -1
            gradient_norm = -1
            update_ratios=-1
            i = 0
            if eval_timestamp is None:
                eval_timestamp = time.time()

            ## This is the main training loop
            for training_args in training_iterator_fun():
                monitor.register_batch(training_args)
                training_results = training_function(training_args)
                monitor.update_many_now(training_results)


                # eval_time and eval_iterations allow the user to control how often to run evaluations
                eval_time_dt = time.time() - eval_timestamp
                eval_iteration += 1

                if ((eval_time is not None and eval_time > 0 and eval_time_dt >= eval_time) or
                    (eval_iterations is not None and eval_iterations > 0 and eval_iteration >= eval_iterations)):
                    best_test_cost = evaluate_model(model,
                                                    validation_iterator_fun,
                                                    validation_function,
                                                    best_test_cost,
                                                    model_format_string,
                                                    epoch_fraction,
                                                    monitor=monitor,
                                                    validation_iterations=validation_iterations,
                                                    keep_snapshots=keep_snapshots)
                    eval_timestamp = time.time()
                    eval_iteration = 0
                # end of training iteration

            ## After epoch
            eval_epoch += 1
            if (eval_epochs is not None and eval_epochs > 0 and eval_epoch >= eval_epochs):
                best_test_cost = evaluate_model(model,
                                                validation_iterator_fun,
                                                validation_function,
                                                best_test_cost,
                                                model_format_string,
                                                epoch + 1,
                                                monitor=monitor,
                                                validation_iterations=validation_iterations,
                                                keep_snapshots=keep_snapshots)
                eval_epoch = 0
    # Done with the whole training loop

    evaluate_model(model,
                   validation_iterator_fun,
                   validation_function,
                   best_test_cost,
                   model_format_string,
                   max_epochs,
                   monitor=monitor,
                   validation_iterations=validation_iterations,
                   keep_snapshots=keep_snapshots)


def evaluate_model(model,
                   validation_iterator_fun,
                   validation_function,
                   best_test_cost,
                   model_format_string,
                   epoch,
                   validation_iterations=None,
                   monitor=None,
                   keep_snapshots=False):
    is_best = False
    validation_results = []
    channels = defaultdict(list)
    for validation_batch in validation_iterator_fun():
        validation_result = validation_function(**validation_batch)
        validation_results.append(validation_result)

    if len(validation_results) > 0:
        if isinstance(validation_results[0], dict):
            gathered_results = defaultdict(list)
            for name, validation_result in validation_results.items():
                gathered_results[name].append(validation_result)
            mean_results = dict()
            for name, gathered_result in gathered_results.items():
                try:
                    mean_results[name] = np.mean(gathered_results, axis=0)
                except TypeError:
                    


    validation_cost = np.mean(channels['evaluation loss'], dtype=np.float)
    validation_accuracy = np.mean(channels['evaluation accuracy'], dtype=np.float)
    if monitor is not None:
        monitor.update_one_now('evaluation loss', validation_cost)
        monitor.update_one_now('evaluation accuracy', validation_accuracy)

    if validation_cost < best_test_cost:
        best_test_cost = validation_cost
        is_best = True

    model_path = model_format_string.format(epoch=epoch, test_cost=validation_cost, test_accuracy=validation_accuracy)
    save_snapshot(model, model_path, is_best, monitor=monitor, keep_snapshots=keep_snapshots)
    return best_test_cost


def setup_directory(output_dir):
    # Create directory and set up symlinks if it doesn't already exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        parent_dir = os.path.dirname(output_dir)
        symlink_name = os.path.join(parent_dir, 'latest_training')
        if os.path.exists(symlink_name) and not os.readlink(symlink_name) == output_dir:
            os.remove(symlink_name)
        try:
            os.symlink(output_dir, symlink_name, target_is_directory=True)
        except:
            print("Could not create symlink to latest training directory.")


def save_snapshot(model, model_filename, is_best=False, monitor=None, keep_snapshots=False):
    """
    Saves a snapshot of the model to the given model filename.
    :param model: The model to save.
    :param model_filename: The path to save the model to.
    :param is_best: If True, the model will be updated as the best model.
    :param monitor: A Monitor object for this training. If supplied, it will be saved together with the model.
    :param keep_snapshots: If this is False, snapshots which aren't either the latest or the best model are
                           removed. This constrains the total memory use to at most 2 model snapshots per
                           training. Otherwise the number of snapshots increases unbounded during training.
    """
    model_filename = os.path.abspath(model_filename)
    output_dir = os.path.dirname(model_filename)
    os.makedirs(output_dir, exist_ok=True)

    latest_file_link = os.path.join(output_dir, 'latest_model')
    best_file_link = os.path.join(output_dir, 'best_model')

    if hasattr(model, 'state_dict'):
        latest_file_link += '.torch'
        best_file_link += '.torch'
        model_filename += '.torch'
        import torch
        torch.save(model.state_dict(), model_filename)
    else:
        latest_file_link += '.pkl'
        best_file_link += '.pkl'
        model_filename += '.pkl'
        with open(model_filename, 'wb') as model_fp:
            print("Saving model to {}".format(model_filename))
            pickle.dump(model, model_fp, protocol=4)

    # We keep track of files we might have replaced as the latest or best model. If keep_snapshots is False, these files
    # will automatically be removed.
    to_remove = set()

    # Setting up the latest model symbolic link
    if os.path.exists(latest_file_link):
        latest_file = os.path.realpath(latest_file_link)
        if latest_file != latest_file_link:
            to_remove.add(latest_file)
        os.remove(latest_file_link)
    try:
        # The symlinks should be relative so that we can move
        # the training directories without mucking things up
        target = os.path.basename(model_filename)
        os.symlink(target, latest_file_link)
    except:
        print("Could not create symlink to latest file.")

    # If there is a link to a best file, we always have to check it to make sure it's not also in the to_remove set
    if os.path.exists(best_file_link):
        best_file = os.path.realpath(best_file_link)
        if best_file != best_file_link:
            # If the best file is still the best (is_best == False) we should override the deletion of the file
            # if it was the latest file.
            if not is_best:
                # We have to remove the best file if it's in the set
                if best_file in to_remove:
                    to_remove.remove(best_file)
            else:
                # The new model is the best. We should remove the old model as well as the link to it
                to_remove.add(best_file)
                os.remove(best_file_link)
    else:
        # If there is no symlink to a best file, we set the current model to be the best one, regardless of if the
        # flag was set.
        is_best = True
    if is_best:
        try:
            # The symlinks should be relative so that we can move
            # the training directories without mucking things up
            target = os.path.basename(model_filename)
            os.symlink(target, best_file_link)
        except:
            print("Could not create symlink to latest file.")

    if not keep_snapshots:
        for file in to_remove:
            print("Removing snapshot {}".format(file))
            try:
                os.remove(file)
            except FileNotFoundError:
                pass



class TestSnapshotting(unittest.TestCase):
    def testSnapshotting(self):
        model = dict(foo='bar')
        model_filename1 = '/tmp/snapshots/foo1.pkl'
        model_filename2 = '/tmp/snapshots/foo2.pkl'
        model_filename3 = '/tmp/snapshots/foo3.pkl'
        model_filename4 = '/tmp/snapshots/foo4.pkl'
        model_filename5 = '/tmp/snapshots/foo5.pkl'

        save_snapshot(model, model_filename1, is_best=True, monitor=None, keep_snapshots=False)
        save_snapshot(model, model_filename2, is_best=False, monitor=None, keep_snapshots=False)
        save_snapshot(model, model_filename3, is_best=True, monitor=None, keep_snapshots=False)
        save_snapshot(model, model_filename4, is_best=False, monitor=None, keep_snapshots=False)
        save_snapshot(model, model_filename5, is_best=False, monitor=None, keep_snapshots=False)


def add_training_arguments(parser):
    """ Add common command line arguments used by the training function.
    """
    parser.add_argument('--output-dir',
                        help=("Directory to write output to."))
    parser.add_argument('--max-epochs', help="Maximum number of epochs to train for. A tota number of training_iterations*max_epochs batches are evaluated.", type=int, default=100)
    parser.add_argument('--training-iterations', help="How many iterations to train each model per epoch.", type=int, default=100)
    parser.add_argument('--validation-iterations', help="How many iterations to evaluate a model per epoch.", type=int, default=10)
    parser.add_argument('--iterator-mode', help=("Determines how batches are drawn from the data. If 'sequential', the "
                                                "batches are drawn from the whole data in sequential order. If 'random' "
                                                 "each sequence is randomly drawn from all the data."),
                        choices=('sequential', 'random'), default='random')
    parser.add_argument('--training-iterator-mode', help=("Determines how batches are drawn from the data. If 'sequential', the "
                                                          "batches are drawn from the whole data in sequential order. If 'random' "
                                                          "each sequence is randomly drawn from all the data."),
                        choices=('sequential', 'random'))
    parser.add_argument('--validation-iterator-mode', help=("Determines how batches are drawn from the data. If 'sequential', the "
                                                            "batches are drawn from the whole data in sequential order. If 'random' "
                                                            "each sequence is randomly drawn from all the data."),
                        choices=('sequential', 'random'))
    parser.add_argument('--eval-time', '--eval-every', help="How often to run the model on the validation set in seconds.", type=float)
    parser.add_argument('--eval-epochs', help="How often to run the model on the validation set in epochs. 1 means at the end of every epoch.", type=float)
    parser.add_argument('--eval-iterations', help="How often to run the model on the validation set number of training iterations.", type=int)

    parser.add_argument('--live-plot', help="Plot training progress.", action='store_true')
    parser.add_argument('--plot-max-range', help="If given, the plot will be windowed over at most this many examples.", type=int)
    parser.add_argument('--no-pre-eval', help="If flag is set, no evaluation will be run before training starts",
                        action='store_true')
    parser.add_argument('--keep-snapshots', help="If flag is set, all snapshots will be kept. otherwise only the best and the latest are kept.",
                        action='store_true')
    parser.add_argument('--monitor-save-interval',
                        help="If given, the monitor will flush its values to disk at the given interval in seconds",
                        type=float)
    parser.add_argument('--monitor-storage-mode',
                        help="The storage method to use for monitor data. 'hdf5' uses the HDF5 format which is good "
                             "for large amounts of data. 'directory' uses a directory of compressed text files which "
                             "takes more space but is less prone to data corruption.",
                        choices=('hdf5', 'directory'),
                        default='directory')
    parser.add_argument('--daemon', help="Start the training in a separate process, controllable through an external"
                                         "app", action="store_true")


def pick_arguments(kwargs):
    arg_names = ('output_dir', 'max_epochs', 'training_iterations', 'test_iterations', 'iterator_mode', 'eval_time')
    return {arg:value for arg, value in kwargs.items() if arg in arg_names}
