import unittest
from pathlib import Path

import numpy as np
import time

from mltrain.train import Monitor, train, HyperParameterManager, HyperParameterTrainer, DiscreteHyperParameter


class DummyModel(object):
    def __init__(self):
        self.i = 1

    def fit(self, batch):
        self.i += 1
        return {'iteration': self.i, 'batch': batch, 'accuracy': self.i, 'loss': 1/self.i}

    def evaluate(self, batch):
        return {'iteration': self.i, 'batch': batch, 'accuracy': self.i, 'loss': 1/self.i}

    def get_metadata(self):
        return dict()

    def save(self, path):
        with open(path, 'wb') as fp:
            fp.write(bytes('{}'.format(self.i), encoding='ascii'))


class TestTraining(unittest.TestCase):
    def test_dummy_model(self):
        model = DummyModel()
        training_dataset = ['a', 'b', 'c']
        evaluation_dataset = ['d', 'e', 'f']
        train(model=model, training_dataset=training_dataset, evaluation_dataset=evaluation_dataset, max_epochs=3,
              output_dir=Path('/tmp/experiments'))


class TestMonitor(unittest.TestCase):
    def testMonitor(self):
        # Some parameters for the test:
        test_every = 10
        num_batches = 200
        sleep_time = 0.05 # seconds
        num_lines = 20
        channels = ['channel_{}'.format(i) for i in range(num_lines)]
        line_values = {channel: np.random.randn(i+1) for i, channel in enumerate(channels)}
        validation_error = np.random.randn()

        with Monitor('/tmp/monitor') as monitor:
            t0 = time.time()
            for batch_num in range(num_batches):
                # Updates the channel 'channel1' and increments time.
                print(batch_num)
                line_values = {channel: line_value + np.random.randn(*line_value.shape) for channel, line_value in line_values.items()}
                monitor.log_now(line_values)
                if (batch_num+1) % test_every == 0:
                    # Add values to the test channel every *test_every* iteration. This will not increment the time for
                    # the plot.
                    validation_error += np.random.randn()
                    monitor.log_one_now('validation error', validation_error)
                monitor.tick()
            print("Total run time = {}".format(time.time() - t0))


class TestHyperParameterManager(unittest.TestCase):
    def test_hpmanager(self):
        base_args = ('foo', 2, ('a', 'b', 'c'), DiscreteHyperParameter([3, 5]))
        base_kwargs = {'baz': 'bar', 'asdf': DiscreteHyperParameter([True, False])}
        hp_manager = HyperParameterManager(base_args=base_args, base_kwargs=base_kwargs)
        args, kwargs = hp_manager.get_any_hyper_params()
        print(hp_manager.get_any_hyper_params())
        print(hp_manager.get_any_hyper_params())
