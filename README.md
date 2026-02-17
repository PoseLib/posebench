# posebench

This repository was setup as a form of regression testing for [PoseLib](https://github.com/vlarsson/PoseLib). Each estimator is compared to the corresponding one in [pycolmap](https://github.com/colmap/pycolmap) if available.

## Installation

Install the package in editable mode:

```
pip install -e .
```

## Running the benchmark

The full benchmark suite can be run as

```
posebench
```

or equivalently

```
python -m posebench
```

which will show the average metrics across all datasets. Results are automatically saved to `results/<timestamp>.json` after each run.

You can also specify RANSAC/bundle options and which specific methods/datasets to run as command line options.

```
usage: posebench [-h] [--min_iterations MIN_ITERATIONS] [--max_iterations MAX_ITERATIONS]
                 [--success_prob SUCCESS_PROB] [--initial_lambda INITIAL_LAMBDA]
                 [--method METHOD] [--dataset DATASET] [--subsample SUBSAMPLE] [--subset]
                 [--output OUTPUT] [--no-save]
                 [--compare FILE [FILE ...]] [--per-dataset]

optional arguments:
  -h, --help            show this help message and exit
  --min_iterations MIN_ITERATIONS
  --max_iterations MAX_ITERATIONS
  --success_prob SUCCESS_PROB
  --initial_lambda INITIAL_LAMBDA
  --method METHOD
  --dataset DATASET
  --subsample SUBSAMPLE
  --subset              Use a smaller subsampled dataset for quick testing
  --output OUTPUT       Output path for results JSON (default: results/<timestamp>.json)
  --no-save             Do not save results to file
  --compare FILE [FILE ...]
                        Compare result files. With two args: REF NEW. With one arg:
                        compares results/baseline.json vs FILE
  --per-dataset         Show per-dataset breakdowns in comparison (use with --compare)
```

### Comparing results

To compare two saved result files (e.g. to check regressions after a PoseLib change):

```
posebench --compare results/baseline.json results/new.json
```

With one argument, `results/baseline.json` is used as the reference:

```
posebench --compare results/new.json
```

Add `--per-dataset` to also show the breakdown per individual dataset.

## Python API

```python
from posebench import run_benchmark

all_metrics, dataset_names = run_benchmark(
    min_iterations=100,
    max_iterations=1000,
    success_prob=0.9999,
    initial_lambda=1e-3,   # bundle adjustment initial lambda
    method_filter=["poselib"],
    dataset_filter=["megadepth"],
    subsample=10,
    subset=False,
    output="results/my_run.json",
    no_save=False,
)
```

`run_benchmark` returns a tuple of:
- `all_metrics`: `Dict[str, Dict[str, Dict[str, Dict[str, float]]]]` — `{problem: {dataset: {method: {metric: value}}}}`
- `dataset_names`: list of all dataset names that were evaluated

## Datasets

The benchmarking is done on a collection of datasets from various papers. Note that we use the same metrics for all datasets in each problem category (and not necessarily the ones used in the original dataset).

The required datasets are downloaded automatically on first run. To download them manually:

```
sh download_data.sh
```

To use a smaller subsampled dataset (10x fewer pairs) for quick testing, pass `--subset`.

## TODO

* Add missing estimators (multi-camera, hybrid, etc.)
* Add refinement benchmark
