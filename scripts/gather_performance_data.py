import argparse
from collections import defaultdict
from pathlib import Path
import json
from mltrain import plotting
import matplotlib.pyplot as plt
from csv import DictWriter

def main():
    parser = argparse.ArgumentParser(description="Summarize performance data to a csv ")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('output', type=Path)
    parser.add_argument('--performance-metrics', nargs='+', default=["mean_absolute_error"])
    parser.add_argument('--hyper-parameters', nargs='+', default=['max_depth', 'n_estimators'])
    args = parser.parse_args()

    experiments = []
    for d in args.experiment_directories:
        for metadata_file in d.glob('**/metadata.json'):
            log_dir = metadata_file.with_name('logs')
            if log_dir.exists():
                experiments.append(metadata_file)

    data = []

    for metadata_file in experiments:
        try:
            with open(metadata_file) as fp:
                metadata = json.load(fp)
            experiment_data = dict()
            model_kwargs = metadata['model_metadata']['kwargs']
            for hp_param in args.hyper_parameters:
                param_value = model_kwargs[hp_param]
                if param_value is None:
                    param_value = -1
                else:
                    param_value = float(param_value)
                experiment_data[hp_param] = param_value
            log_dir = metadata_file.with_name('logs')

            for performance_metric in args.performance_metrics:
                performance_metric_file = log_dir / 'best_{}.txt'.format(performance_metric)
                metric_value = None
                with open(performance_metric_file) as fp:
                    for line in fp:
                        it, value = line.strip().split()
                        metric_value = float(value)
                if metric_value is not None:
                    experiment_data[performance_metric] = metric_value
            data.append(experiment_data)
        except FileNotFoundError:
            continue

    fieldnames = list(args.performance_metrics) + list(args.hyper_parameters)
    with open(args.output, 'w') as fp:
        csv_writer = DictWriter(fp, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(data)


if __name__ == '__main__':
    main()