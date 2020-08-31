import argparse
from numbers import Number

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot logs as curves")
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--xscale', default='linear')
    parser.add_argument('--yscale', default='linear')
    parser.add_argument('--xlim', type=float, nargs=2)
    parser.add_argument('--ylim', type=float, nargs=2)
    args = parser.parse_args()
    plot_online(args.files, args.xscale, args.yscale, args.xlim, args.ylim)

def plot_offline(files, xscale, yscale, xlim, ylim):
    for f in files:
        channel_name = f.with_suffix('').name
        channel_df = pd.read_csv(f, sep=' ', names=['iteration', channel_name], header=None)
        plt.plot(channel_df['iteration'], channel_df[channel_name], label=channel_name, alpha=.5)
    plt.legend()
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlim is not None:
        start, end = xlim
        plt.xlim(start, end)
    if ylim is not None:
        start, end = ylim
        plt.ylim(start, end)
    plt.show()


class Limits(object):
    def __init__(self,*, x_max=None, x_min=None, y_max=None, y_min=None, alpha=0.9):
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min


    def update(self, x, y):
        update_x = self.update_x(x)
        update_y = self.update_y(y)
        return update_x or update_y

    def update_x(self, x):
        updated = False
        if self.x_max is None or x > self.x_max:
            self.x_max = x
            updated = True
        if self.x_min is None or x < self.x_min:
            self.x_min = x
            updated = True
        return updated

    def update_y(self, y):
        updated = False
        if self.y_max is None or y > self.y_max:
            self.y_max = y
            updated = True
        if self.y_min is None or y < self.y_min:
            self.y_min = y
            updated = True
        return updated

    def update_limits(self, limits: 'Limits'):
        # This will only look at the same sides of the limits, since newly initialized limits are set to the lowest vs.
        # highest values
        updated = False
        if self.x_min is None or limits.x_min < self.x_min:
            self.x_min = limits.x_min
            updated = True
        if self.x_max is None or limits.x_max > self.x_max:
            self.x_max = limits.x_max
            updated = True
        if self.y_min is None or limits.y_min < self.y_min:
            self.y_min = limits.y_min
            updated = True
        if self.y_max is None or limits.y_max > self.y_max:
            self.y_max = limits.y_max
            updated = True
        return updated

    def __str__(self):
        return f'<Limits: x_lim: ({self.x_min}, {self.x_max}), y_lim: ({self.y_min}, {self.y_max})'

    def __mul__(self, other):
        """Returns new Limits where distaince between the minimum and maximum have been scaled by this
        number on both sides"""
        if isinstance(other, Number):
            dx = self.x_max - self.x_min
            x_pad = dx*(other - 1)/2
            dy = self.y_max - self.y_min
            y_pad = dy*(other - 1)/2
            return Limits(x_min = self.x_min-x_pad, x_max=self.x_max+x_pad, y_min=self.y_min-y_pad, y_max = self.y_max+y_pad)


class FileMonitor(object):
    def __init__(self, file, ax, smoothing=0.9, **plot_kwargs):
        self.fd = open(file)
        self.channel_name = file.with_suffix('').name
        self.ax = ax
        self.smoothing = smoothing
        self.x_data = []
        self.y_data = []
        self.y_data_smoothed = []
        self.remainder = ''
        self.limits = Limits()

        self.update_data()
        self.line, = ax.plot(self.x_data, self.y_data, label=self.channel_name, alpha=.3)
        self.smoothed_line, = ax.plot(self.x_data,
                                      self.y_data_smoothed,
                                      label=self.channel_name + f' smoothed ({self.smoothing})',
                                      c=self.line.get_color(),
                                      alpha=1)


    def update_data(self):
        text = self.remainder + self.fd.read()
        if text:
            lines = text.split('\n')
            for line in lines[:-1]:
                if line != '':
                    iteration, value = line.split(' ')
                    x = float(iteration)
                    y = float(value)
                    self.x_data.append(x)
                    self.y_data.append(y)
                    if self.y_data_smoothed:
                        y_smooth = self.y_data_smoothed[-1]
                    else:
                        y_smooth = y

                    self.y_data_smoothed.append(y_smooth*self.smoothing + (1-self.smoothing)*y)
                    self.limits.update_x(x)
                    self.limits.update_y(y)
            last_line = lines[-1]
            if text[-1] != '\n':  # If the last token is not a newline, the line is just a partial line
                self.remainder = last_line
            else:
                if last_line != '':
                    iteration, value = last_line.split(' ')
                    self.x_data.append(float(iteration))
                    self.y_data.append(float(value))

    def update_line(self):
        self.update_data()
        self.line.set_data(self.x_data, self.y_data)
        self.smoothed_line.set_data(self.x_data, self.y_data_smoothed)

    def get_artists(self):
        return self.line, self.smoothed_line


def plot_online(files, xscale, yscale, xlim, ylim):
    fig, ax = plt.subplots()
    fmonitors = [FileMonitor(f, ax, alpha=.5) for f in files]
    artists = []
    limits = Limits()
    limits.update(1, 1)
    limits.update(-1, -1)

    for fm in fmonitors:
        limits.update_limits(fm.limits)
        artists.extend(fm.get_artists())
    plt.legend()

    def init():  # only required for blitting to give a clean slate.
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        return [fm.line for fm in fmonitors]

    def animate(i):
        update_xlim = False
        update_ylim = False
        for fm in fmonitors:
            fm.update_line()

            xd_pad = (limits.x_max - limits.x_min)*0.05
            if fm.limits.x_min - xd_pad < limits.x_min:
                limits.update_x(limits.x_min-xd_pad*2)
                update_xlim = True
            if fm.limits.x_max + xd_pad > limits.x_max:
                limits.update_x(limits.x_max+xd_pad*2)
                update_xlim = True

            yd_pad = (limits.y_max - limits.y_min)*0.05
            if fm.limits.y_min - yd_pad < limits.y_min:
                limits.update_y(limits.y_min - yd_pad * 2)
                update_ylim = True
            if fm.limits.y_max + yd_pad > limits.y_max:
                limits.update_y(limits.y_max + yd_pad * 2)
                update_ylim = True

        if update_xlim:
            ax.set_xlim(limits.x_min, limits.x_max)

        if update_ylim:
            print("Updating ylim")
            ax.set_ylim(limits.y_min, limits.y_max)

        if update_xlim or update_ylim:
            fig.canvas.resize_event()

        return [fm.line for fm in fmonitors]

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=10, blit=False, save_count=50)

    plt.show()







if __name__ == '__main__':
    main()