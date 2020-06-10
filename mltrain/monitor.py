import time
import os
import os.path
import multiprocessing
#import multiprocessing.dummy as multiprocessing
import queue
import gzip
import signal
from collections import defaultdict


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
        #print("Waiting for monitor process to exit cleanly")
        self.monitor_process.join()
        #print("Monitor exiting")

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
        #print("Monitor process is exiting")
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
