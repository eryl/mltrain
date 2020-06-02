import argparse
from collections import defaultdict
from pathlib import Path
import json
from mltrain import plotting
import matplotlib.pyplot as plt
from csv import DictReader

def main():
    parser = argparse.ArgumentParser(description="Plot performence vs hyper parameters")
    parser.add_argument('performance_data', help="CSV with performance data produced by `gather_performance_data.py`", type=Path)
    parser.add_argument('--performance-metrics', nargs='+', default=["root_mean_squared_error"])
    parser.add_argument('--hyper-parameters', nargs='+', default=['max_depth', 'n_estimators'])
    args = parser.parse_args()

    with open(args.performance_data) as fp:
        csv_reader =DictReader(fp)
        data = list(csv_reader)

    transpose_data = defaultdict(list)
    for d in data:
        for k, v in d.items():
            transpose_data[k].append(float(v))

    dim_labels, values = zip(*sorted(transpose_data.items()))
    values = list(zip(*values))
    plotting.parallel_coordinates(values,
                                  dim_labels=dim_labels,
                                  c=transpose_data[args.performance_metrics[0]],
                                  alpha=.5)
    plt.show()


if __name__ == '__main__':
    main()