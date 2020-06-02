import argparse
from collections import defaultdict
from pathlib import Path
import json
from mltrain import plotting
import matplotlib.pyplot as plt
from csv import DictReader
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot performence vs hyper parameters")
    parser.add_argument('performance_data', help="CSV with performance data produced by `gather_performance_data.py`", type=Path)
    parser.add_argument('--performance-metrics', nargs='+', default=["median_absolute_deviance"])
    parser.add_argument('--hyper-parameters', nargs='+', default=['max_depth', 'n_estimators'])
    args = parser.parse_args()

    with open(args.performance_data) as fp:
        csv_reader =DictReader(fp)
        data = list(csv_reader)

    n_hyper_params = len(args.hyper_parameters)
    for performance_metric in args.performance_metrics:
        for i in range(n_hyper_params):
            for j in range(i+1, n_hyper_params):
                hp_i = args.hyper_parameters[i]
                hp_j = args.hyper_parameters[j]
                # This assumes the hyper parameters have relatively few, discrete values
                summary = defaultdict(lambda: defaultdict(list))
                hp_j_vals = set()
                for d in data:
                    try:
                        hp_i_val = float(d[hp_i])
                    except ValueError:
                        hp_i_val = -1
                    try:
                        hp_j_val = float(d[hp_j])
                    except ValueError:
                        hp_j_val = -1
                    hp_j_vals.add(hp_j_val)
                    perf = d[performance_metric]
                    if perf:
                        try:
                            perf = float(perf)
                        except ValueError:
                            pass
                        summary[hp_i_val][hp_j_val].append(perf)
                hp_j_vals = list(sorted(hp_j_vals))
                n_rows = len(summary)
                n_cols = len(hp_j_vals)
                fig, axes = plt.subplots(n_rows, n_cols, sharex='all', sharey='all')
                for k, (hp_i_val, hp_i_hp_j_vals) in enumerate(sorted(summary.items())):
                    for l, hp_j_val in enumerate(hp_j_vals):
                        perfs = hp_i_hp_j_vals.get(hp_j_val, None)
                        ax = axes[k, l]
                        if perfs is not None:
                            sns.kdeplot(perfs, ax=ax)
                        if l == 0:
                            ax.set_ylabel(hp_i_val)
                        if k == 0:
                            ax.set_title(hp_j_val)
                plt.figtext(0.005, 0.45, hp_i, rotation=90)
                plt.figtext(0.45, 0.97, hp_j, rotation=0)
    plt.show()


if __name__ == '__main__':
    main()