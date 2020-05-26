import argparse
from collections import defaultdict
from pathlib import Path
import json
import re
from mltrain import plotting
import matplotlib.pyplot as plt
from csv import DictWriter

def main():
    parser = argparse.ArgumentParser(description="Summarize performance data to a csv ")
    parser.add_argument('experiment_directories', help="Scan these directories for experiments", type=Path, nargs='+')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    experiments = []
    for d in args.experiment_directories:
        for metadata_file in d.glob('**/metadata.json'):
            log_dir = metadata_file.with_name('logs')
            if log_dir.exists():
                experiments.append(metadata_file)

    data = []
    model_kwarg_names = set()
    metric_names = set()
    for metadata_file in experiments:
        try:
            with open(metadata_file) as fp:
                metadata = json.load(fp)
            experiment_data = dict()
            model_kwargs = metadata['model_metadata']['kwargs']
            for kwarg, value in model_kwargs.items():
                experiment_data[kwarg] = value
                model_kwarg_names.add(kwarg)
            log_dir = metadata_file.with_name('logs')
            for f in log_dir.iterdir():
                m = re.match('best_(\w+).txt', f.name)
                if m is not None:
                    performance_metric = m.group(1)
                    with open(f) as fp:
                        for line in fp:
                            it, value = line.strip().split()
                            metric_value = float(value)
                    if metric_value is not None:
                        metric_names.add(performance_metric)
                        experiment_data[performance_metric] = metric_value
            data.append(experiment_data)
        except FileNotFoundError:
            continue

    fieldnames = list(sorted(metric_names)) + list(sorted(model_kwarg_names))
    with open(args.output, 'w') as fp:
        csv_writer = DictWriter(fp, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(data)


if __name__ == '__main__':
    main()