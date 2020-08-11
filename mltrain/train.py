import copy
import datetime
import json
import time
import sys
import os
import os.path
import multiprocessing
import signal
import pickle
from collections import defaultdict
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional, Any, Iterable, Dict, List, Union, Collection
try:  # Literal might not be supported in python versions earlier than 3.7
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from mltrain.performance import setup_metrics, EvaluationMetric
from mltrain.monitor import Monitor
from mltrain.util import load_config

from tqdm import trange, tqdm
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def get_metadata(self) -> Dict:
        pass

    @abstractmethod
    def evaluation_metrics(self) -> List[EvaluationMetric]:
        pass

    @abstractmethod
    def save(self, save_path) -> Union[str, Path]:
        pass


class MinibatchModel(BaseModel):
    @abstractmethod
    def fit_batch(self, batch) -> Dict:
        pass

    @abstractmethod
    def evaluate_batch(self, batch) -> Dict:
        pass


class FullbatchModel(BaseModel):
    @abstractmethod
    def fit_dataset(self, batch) -> Dict:
        pass

    @abstractmethod
    def evaluate_dataset(self, batch) -> Dict:
        pass




class JSONEncoder(json.JSONEncoder):
    "Custom JSONEncoder which tries to encode filed types (like pathlib Paths) as strings"
    def default(self, o):
        if is_dataclass(o):
            attributes = copy.copy(o.__dict__)
            attributes['dataclass_name'] = o.__class__.__name__
            attributes['dataclass_module'] = o.__module__
            return attributes
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


@dataclass
class TrainingConfig(object):
    """ Configuration dataclass for the mltrain.train.train function.

    Args:
        max_epochs (int): Train for at most this number of epochs
        keep_snapshots (bool): If True, keep all checkpoints from training. If False, only the best and the lates
                               checkpoints are saved.
        eval_time (int): Run the evaluation loop and checkpointing after this number of seconds has passed
        eval_iterations (int): Run the evaluation loop after these many training iterations (batches)
        eval_epochs (int): Run the evaluation loop after these many epochs have passed.
        model_format_string (str): Use this format string for saving model checkpoint files.
        do_pre_eval (bool): Run the evaluation loop before training starts.
    """
    max_epochs: int = 1
    keep_snapshots: Union[Literal['all', 'none', 'best'], bool] = 'none'
    eval_time: Optional[int] = None
    eval_iterations: Optional[int] = None
    eval_epochs: int = 1
    model_format_string: Optional[str] = None
    do_pre_eval: bool = False


@dataclass
class TrainingArguments(object):
    model: Union[BaseModel, FullbatchModel, MinibatchModel]
    output_dir: Path
    training_dataset: Iterable
    evaluation_dataset: Iterable
    training_config: TrainingConfig
    metadata: Optional[Dict] = None
    artifacts: Optional[Dict] = None


def train(
        *,
        training_args: TrainingArguments):
    model = training_args.model
    metadata = training_args.metadata
    output_dir = training_args.output_dir
    training_dataset = training_args.training_dataset
    evaluation_dataset = training_args.evaluation_dataset

    best_performance, model_format_string, output_dir = setup_training(model=model,
                                                                       training_config=training_args.training_config,
                                                                       metadata=metadata,
                                                                       artifacts=training_args.artifacts,
                                                                       output_dir=output_dir,
                                                                       )
    try:
            best_performance, best_model_path = training_loop(model=model,
                                                              training_dataset=training_dataset,
                                                              evaluation_dataset=evaluation_dataset,
                                                              training_config=training_args.training_config,
                                                              best_performance=best_performance,
                                                              model_checkpoint_format=model_format_string,
                                                              output_dir=output_dir
                                                              )
            return best_performance, best_model_path
    except Exception as e:
        raise TrainingError(metadata, "Error during training") from e


def setup_training(
        *,
        model,
        training_config,
        output_dir,
        artifacts=None,
        metadata=None):

    model_format_string = training_config.model_format_string
    if model_format_string is None:
        model_format_string = model.__class__.__name__ + '_epoch-{epoch:.04f}_{metrics}'

    output_dir = output_dir / make_timestamp()
    while output_dir.exists():
        time.sleep(1)
        output_dir = output_dir / make_timestamp()
    model_format_string = output_dir / model_format_string
    setup_directory(output_dir)

    if artifacts is not None:
        artifacts_dir = output_dir / 'artifacts'
        artifacts_dir.mkdir()
        for k, v in artifacts.items():
            with open(artifacts_dir / (k + '.pkl'), 'wb') as fp:
                pickle.dump(v, fp)
    if metadata is None:
        metadata = dict()
    try:
        model_metadata = model.get_metadata()
        metadata['model_metadata'] = model_metadata
        #print("Model parameters are: ")
        #print('\n'.join(list(sorted('{}: {}'.format(k, v) for k, v in model_metadata.items()))))
    except AttributeError:
        print("Couldn't get model parameters, skipping model_params for the metadata")

    metadata['training_params'] = training_config

    json_encoder = JSONEncoder(sort_keys=True, indent=4, separators=(',', ': '))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as metadata_fp:
        json_encoding = json_encoder.encode(metadata)
        metadata_fp.write(json_encoding)

    best_performance = setup_metrics(model.evaluation_metrics())
    return best_performance, model_format_string, output_dir


def training_loop(
        *,
        model,
        training_dataset,
        evaluation_dataset,
        training_config,
        best_performance,
        model_checkpoint_format,
        output_dir
):

    epoch = 0
    best_model_path = None

    def sigint_handler(signal, frame):
        checkpoint(model, model_checkpoint_format, np.nan, {}, is_best=False, keep_snapshots='all')
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    # Since we call evaluate_models from som many places below, we summarize the common arguments in a dict

    with Monitor(output_dir / 'logs') as monitor:
        eval_kwargs = dict(model=model,
                           evaluation_dataset=evaluation_dataset,
                           model_checkpoint_format=model_checkpoint_format,
                           monitor=monitor,
                           keep_snapshots=training_config.keep_snapshots)
        if training_config.do_pre_eval:
            best_performance, best_model_path = evaluate_model(best_performance=best_performance, epoch=0, **eval_kwargs)

        # These variables will be used to control when to do evaluation
        eval_timestamp = time.time()
        eval_epoch = 0
        eval_iteration = 0
        needs_final_eval = True

        for epoch in trange(training_config.max_epochs, desc='Epochs'):
            if hasattr(model, 'fit_dataset'):
                model.fit_dataset(training_dataset)
            elif hasattr(model, 'fit_batch') or hasattr(model, 'fit'):
                ## This is the main training loop
                for i, batch in enumerate(tqdm(training_dataset, desc='Training batch')):
                    needs_final_eval = True
                    epoch_fraction = epoch + i / len(training_dataset)
                    if hasattr(model, 'fit_batch'):
                        training_results = model.fit_batch(batch)
                    elif hasattr(model, 'fit'):
                        print("Model has 'fit' attribute, we treat it as fit_batch")
                        training_results = model.fit(batch)
                    monitor.log_one_now('epoch', epoch_fraction)
                    if training_results is not None:
                        monitor.log_now(training_results)

                    # eval_time and eval_iterations allow the user to control how often to run evaluations
                    eval_time_dt = time.time() - eval_timestamp
                    eval_iteration += 1

                    if (
                            (training_config.eval_time is not None
                             and training_config.eval_time > 0
                             and eval_time_dt >= training_config.eval_time)
                            or
                            (training_config.eval_iterations is not None
                             and training_config.eval_iterations > 0
                             and eval_iteration >= training_config.eval_iterations)
                    ):
                        best_performance, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch_fraction, **eval_kwargs)
                        eval_timestamp = time.time()
                        eval_iteration = 0
                        needs_final_eval = False

                    monitor.tick()
                    # End of training loop

            eval_epoch += 1
            if (
                    training_config.eval_epochs is not None
                    and training_config.eval_epochs > 0
                    and eval_epoch >= training_config.eval_epochs
            ):
                best_performance, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch, **eval_kwargs)
                eval_epoch = 0
                needs_final_eval = False
            # End of epoch

    # Done with the whole training loop. If we ran the evaluate_model at the end of the last epoch, we shouldn't do
    # it again
    if needs_final_eval:
        best_performance, best_model_path = evaluate_model(best_performance=best_performance, epoch=epoch, **eval_kwargs)
    return best_performance, best_model_path


def evaluate_model(*,
                   model,
                   evaluation_dataset,
                   best_performance,
                   model_checkpoint_format,
                   epoch,
                   monitor=None,
                   keep_snapshots=False):
    evaluation_results = {}

    if hasattr(model, 'evaluate_dataset'):
        evaluation_results.update(model.evaluate_dataset(evaluation_dataset))
    elif hasattr(model, 'evaluate_batch') or hasattr(model, 'evaluate'):
        gathered_evaluation_results = defaultdict(list)
        for batch in tqdm(evaluation_dataset, desc='Validation batch'):
            if hasattr(model, 'evaluate_batch'):
                batch_eval_results = model.evaluate_batch(batch)
            elif hasattr(model, 'evaluate'):
                print("Model has 'evaluate' method, we treat it as 'evaluate_batch'")
                batch_eval_results = model.evaluate(batch)
            for k, v in batch_eval_results.items():
                gathered_evaluation_results[k].append(v)
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
                                 new_performance, is_best, keep_snapshots=keep_snapshots)
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
               is_best,
               latest_model_name='latest_model',
               best_model_name='best_model',
               keep_snapshots: Union[Literal['all', 'none', 'best'], bool] = False):
    if isinstance(keep_snapshots, bool):
        if keep_snapshots:
            keep_snapshots = 'all'
        else:
            keep_snapshots = 'none'
    model_directory = checkpoint_format.parent.resolve(strict=False)
    model_name = checkpoint_format.name.format(epoch=epoch, metrics=performances)
    checkpoint_path = checkpoint_format.with_name(model_name).resolve()
    model_directory.mkdir(exist_ok=True)
    checkpoint_path = model.save(checkpoint_path)
    latest_model_symlink = model_directory / latest_model_name
    best_model_symlink = model_directory / best_model_name

    if keep_snapshots != 'all' and latest_model_symlink.exists():
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
            if keep_snapshots == 'none':
                # The previous best model can't also be the latest model since we take care of that above, so it's safe
                # to remove
                previous_best_model = best_model_symlink.resolve(strict=True)
                previous_best_model.unlink()
        if best_model_symlink.is_symlink():
            best_model_symlink.unlink()
        relative_checkpoint = checkpoint_path.relative_to(best_model_symlink.absolute().parent)
        best_model_symlink.symlink_to(relative_checkpoint)
    return best_model_symlink.resolve()


def setup_training_config(config_path):
    return load_config(config_path, TrainingConfig)


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


if __name__ == '__main__':
    conf = TrainingConfig()
    foo = dict(a=1, b=2, c=[1,2,3])
    json_encoder = JSONEncoder(sort_keys=True, indent=4, separators=(',', ': '))
    json_encoder.encode(conf)