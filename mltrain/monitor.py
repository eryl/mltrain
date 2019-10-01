import gzip
import queue
import signal
import unittest
import multiprocessing
import os.path

import numpy as np
import time
from collections import defaultdict


class Monitor(object):
    def __init__(self, monitor_dir, save_interval=60, buffer_size=100):
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
        print("Exiting")

    def tick(self):
        """
        Progress the time one step.
        """
        self.time += 1

    def update_one_at(self, time, channel_name, value):
        """
        Updates the channel at the given time. If the time is greater than the clock, the clock is updated to the new
        time.
        :param time: The time to update the value at. If this is greater than the current internal time, the internal time is updated to this value.
        :param channel_name: The channel to update.
        :param value: The value at the time for the given channel.
        :return: None
        """
        self.time = max(self.time, time)  # If the time is in the future, we update the clock
        update_command = (time, channel_name, value)
        self.channel_values.put(update_command)

    def update_one_now(self, channel_name, value):
        """
        Updates a single channel_name at the current timestep. This is useful for adding a new datapoint at the same
        time as a different channel.
        :param channel_name: The channel_name name to update.
        :param value: The value to add to the current timestep.
        :return: None
        """
        self.update_one_at(self.time, channel_name, value)

    def update_one_next(self, channel_name, value):
        """
        Updates a single channel and increment time.
        If you wan't to add multiple points to the same
        timestep, use update_many(), or update_one_now()
        :param channel_name: The channel_name name to update.
        :param value: The value for the next timestep.
        :return None
        """
        self.tick()
        self.update_one_now(channel_name, value)

    def update_many_next(self, channel_values):
        """
        Increment time and update all the given channel with the corresponding values.
        :param channel_values: A dictionary of channel: value mappings.
        :return: None
        """
        self.time += 1
        self.update_many_now(channel_values)

    def update_many_now(self, channel_values):
        """
        Update all the given channel with the corresponding values without incrementing time.
        :param channel_values: A dictionary of channel: value mappings.
        :return: None
        """
        for channel_name, value in channel_values.items():
            self.update_one_now(channel_name, value)


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

        self.flush_caches()
        print("Exiting")
        self.close()
        return

    def update_channel(self, command):
        t, channel_name, channel_value = command
        self.channels[channel_name].append((t, channel_value))
        if self.save_interval is not None and time.time() - self.tm1 < self.save_interval:
            self.flush_caches()
            self.tm1 = time.time()
        elif len(self.channels[channel_name]) >= self.buffer_size:
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
                monitor.update_many_next(line_values)
                if (batch_num+1) % test_every == 0:
                    # Add values to the test channel every *test_every* iteration. This will not increment the time for
                    # the plot.
                    validation_error += np.random.randn()
                    monitor.update_one_now('validation error', validation_error)
            print("Total run time = {}".format(time.time() - t0))


def load_channel_data(directory, loss_channel):
    channel_base_name = os.path.join(directory, loss_channel)
    for ext in ('.txt.gz', '.txt'):
        if os.path.exists(channel_base_name + ext):
            channel_name, times, values = load_data(channel_base_name + ext)
            return times, values
    raise FileNotFoundError('Could not find channel {} in directory {}'.format(loss_channel, directory))

def load_data(monitor_file):
    print("Loading data from file {}".format(monitor_file))
    basename = os.path.basename(monitor_file)
    channel_name, ext = os.path.splitext(basename)

    if ext == '.gz':
        times, values = load_gzipped_data(monitor_file)
    elif ext == '.txt':
        times, values = load_txt_data(monitor_file)
    else:
        raise ValueError("Unkown file extension {}".format(ext))

    times = [float(t) for t in times]
    try:
        values = [float(v) for v in values]
        values = np.array(values).squeeze()
    except ValueError:
        pass
    print("Len of values", len(values))
    return channel_name, np.array(times), values


def load_gzipped_data(gzipped_file):
    with gzip.open(gzipped_file, 'r') as fp:
        lines = []
        try:
            for line in fp:
                lines.append(line)
        except EOFError:
            pass
        times, values = zip(*(line.split() for line in lines))
        return times, values

def load_txt_data(text_file):
    with open(text_file, 'r') as fp:
        lines = []
        try:
            for line in fp:
                lines.append(line)
        except EOFError:
            pass
        times, values = zip(*(line.split() for line in lines))
        return times, values