# posebench

Regression testing benchmark for [PoseLib](https://github.com/vlarsson/PoseLib). Each estimator is compared to the corresponding one in [pycolmap](https://github.com/colmap/pycolmap) where available, across four problem categories: relative pose, absolute pose, homography, and depth-aided relative pose.

## Installation

```bash
pip install -e .
```

## Quick start

Run the fast subset benchmark (downloads ~10 instances per dataset automatically):

```bash
posebench --subset
```

Since `results/baseline-subsampled-10.json` is committed to the repo, this will automatically compare your results against the baseline and show a regression diff:

```
Running problem absolute pose
Running problem relative pose
Running problem homography
Running problem monodepth pose
Finished running evaluation in 67.1 seconds (57 datasets)
Results saved to results/20260217_230848.json

REF: results/baseline-subsampled-10.json
  timestamp: 2026-02-17T23:07:13.446200
  poselib:   2.1.0
NEW: results/20260217_230848.json
  timestamp: 2026-02-17T23:08:48.016309
  poselib:   2.1.0

Values shown are from NEW, with (delta = NEW - REF) in parentheses.

=== absolute pose (average over 9 shared datasets) ===
                            AUC0             AUC1             AUC5              med_rot      med_pos             avg_rt             med_rt
PnP (poselib)      56.52 (+0.00)    93.27 (+0.00)    98.65 (+0.00)   3.5E-01 (+0.0E+00)   0.0 (+0.0)     9.3ms (+0.0ms)     8.2ms (+0.0ms)
PnP (COLMAP)       54.45 (+0.00)    91.47 (+0.00)    98.47 (+0.00)   4.3E-01 (+0.0E+00)   0.0 (+0.0)    15.0ms (-0.1ms)    13.8ms (-0.0ms)
PnPf (poselib)     31.24 (+0.00)    74.93 (+0.00)    90.43 (+0.00)   1.4E+01 (+0.0E+00)   0.7 (+0.0)    49.7ms (+0.2ms)    43.8ms (+0.2ms)
PnPf (COLMAP)      26.52 (+0.00)    67.40 (+0.00)    90.49 (+0.00)   5.8E-01 (+0.0E+00)   0.4 (+0.0)    13.5ms (-0.1ms)    12.5ms (-0.1ms)
PnPfr (poselib)    31.19 (+0.00)    75.37 (+0.00)    91.46 (+0.00)   1.7E+01 (+0.0E+00)   0.4 (+0.0)    45.6ms (-0.0ms)    36.8ms (+0.1ms)
PnPfr (COLMAP)     26.68 (+0.00)    68.60 (+0.00)    89.03 (+0.00)   5.6E-01 (+0.0E+00)   0.3 (+0.0)    13.2ms (-0.2ms)    12.0ms (-0.2ms)

=== relative pose (average over 22 shared datasets) ===
                           AUC5            AUC10            AUC20             avg_rt             med_rt
E (poselib)       51.89 (+0.00)    65.63 (+0.00)    76.33 (+0.00)    23.2ms (-0.0ms)    21.9ms (-0.1ms)
E (poselib,TS)    51.46 (+0.00)    65.26 (+0.00)    76.14 (+0.00)    28.2ms (-0.1ms)    24.9ms (-0.1ms)
E (COLMAP)        49.00 (+0.00)    64.70 (+0.00)    76.17 (+0.00)    14.8ms (-0.0ms)    12.9ms (+0.0ms)
F (poselib)       33.83 (+0.00)    47.40 (+0.00)    60.47 (+0.00)    22.7ms (+0.1ms)    20.6ms (+0.2ms)
F (COLMAP)        30.38 (+0.00)    43.40 (+0.00)    55.77 (+0.00)     5.5ms (-0.0ms)     4.0ms (-0.0ms)

=== homography (average over 2 shared datasets) ===
                        AUC5            AUC10            AUC20              avg_rt             med_rt
H (poselib)     3.15 (+0.00)     4.08 (+0.00)    10.03 (+0.00)   515.9us (+11.7us)   234.8us (+0.8us)
H (COLMAP)      3.15 (+0.00)     4.08 (+0.00)     4.54 (+0.00)      1.1ms (+0.0ms)     1.1ms (+0.0ms)

=== monodepth pose (average over 24 shared datasets) ===
                                            AUC5            AUC10            AUC20             avg_rt             med_rt
E                                  63.36 (+0.00)    74.67 (+0.00)    82.48 (+0.00)    10.5ms (-0.0ms)     9.9ms (+0.0ms)
RePoseD (calibrated)               62.91 (+0.00)    76.49 (+0.00)    85.24 (+0.00)    25.3ms (+0.3ms)    24.1ms (+0.3ms)
RePoseD (calibrated) with shift    66.45 (+0.00)    77.67 (+0.00)    85.79 (+0.00)    32.7ms (+0.8ms)    31.6ms (+0.8ms)
E + shared f                       53.28 (+0.00)    68.28 (+0.00)    79.42 (+0.00)    14.1ms (+0.2ms)    12.1ms (+0.1ms)
RePoseD (shared focal)             58.58 (+0.00)    74.19 (+0.00)    83.61 (+0.00)    19.4ms (+0.7ms)    18.2ms (+0.6ms)
F                                  44.42 (+0.00)    59.04 (+0.00)    72.03 (+0.00)     9.8ms (+0.1ms)     8.9ms (+0.1ms)
RePoseD (varying focal)            43.01 (+0.00)    58.68 (+0.00)    73.57 (+0.00)    23.0ms (+1.0ms)    22.1ms (+0.9ms)

=== Summary (across all problems) ===
  unchanged: 72/72
```

This is particularly useful in CI to catch regressions quickly.

## Full benchmark

```bash
posebench
```

Downloads the full dataset on first run. Results are saved to `results/<timestamp>.json`. If `results/baseline.json` exists, the run automatically shows a comparison against it.

## Baselines and comparison

### Creating and updating baselines

```bash
# Save the current results as the subset baseline:
posebench --subset --create_baseline   # → results/baseline-subsampled-10.json

# Save the current results as the full-data baseline:
posebench --create_baseline            # → results/baseline.json
```

### Auto-compare

Whenever a matching baseline file exists (`results/baseline.json` for full runs, `results/baseline-subsampled-10.json` for `--subset`), the benchmark automatically shows a comparison diff instead of the plain metrics table. Use `--no-save` to suppress this.

### Manual comparison

Compare any two saved result files explicitly:

```bash
# Compare two specific files:
posebench --compare results/baseline.json results/new.json

# With one argument, results/baseline.json is used as the reference:
posebench --compare results/new.json

# Add --per-dataset for a per-dataset breakdown:
posebench --compare results/baseline.json results/new.json --per-dataset
```

## CLI reference

```
usage: posebench [-h] [--min_iterations N] [--max_iterations N]
                 [--success_prob P] [--initial_lambda L]
                 [--method METHOD] [--dataset DATASET]
                 [--subsample N] [--subset]
                 [--create_subset [N]]
                 [--output OUTPUT] [--no-save]
                 [--create_baseline]
                 [--compare FILE [FILE ...]] [--per-dataset]

options:
  --min_iterations N    RANSAC minimum iterations
  --max_iterations N    RANSAC maximum iterations
  --success_prob P      RANSAC success probability
  --initial_lambda L    Bundle adjustment initial lambda
  --method METHOD       Only run methods whose name contains METHOD (comma-separated)
  --dataset DATASET     Only run datasets whose name contains DATASET (comma-separated)
  --subsample N         Use every Nth instance at runtime (without modifying data files)
  --subset              Use the pre-built subset dataset (~10 instances per dataset)
  --create_subset [N]   Build a subsampled dataset from the full data and zip it.
                        Uniformly samples up to N instances per dataset (default N=10).
                        Output: data-subsampled-N/ directory and data-subsampled-N.zip
  --output OUTPUT       Output path for results JSON (default: results/<timestamp>.json)
  --no-save             Do not save results to file
  --create_baseline     Run benchmark and save to the canonical baseline file
                        (results/baseline.json or results/baseline-subsampled-10.json)
  --compare FILE [FILE ...]
                        Compare result files. With two args: REF NEW. With one arg:
                        compares results/baseline.json vs FILE
  --per-dataset         Show per-dataset breakdown in comparison (use with --compare)
```

## Python API

```python
from posebench import run_benchmark

all_metrics, dataset_names, saved_path = run_benchmark(
    min_iterations=100,
    max_iterations=1000,
    success_prob=0.9999,
    initial_lambda=1e-3,
    method_filter=["poselib"],
    dataset_filter=["megadepth"],
    subsample=10,
    subset=False,
    output="results/my_run.json",
    no_save=False,
    print_results=True,
)
```

`run_benchmark` returns a 3-tuple:
- `all_metrics`: `Dict[str, Dict[str, Dict[str, Dict[str, float]]]]` — `{problem: {dataset: {method: {metric: value}}}}`
- `dataset_names`: list of all dataset names that were evaluated
- `saved_path`: path to the saved JSON file, or `None` if `no_save=True`

## Datasets

The benchmark covers four problem categories across 57 datasets from various papers. Datasets are downloaded automatically on first run.

| Category | Datasets |
|---|---|
| Relative pose | MegaDepth1500 (×6 matchers), ScanNet1500 (×5 matchers), IMC (9 scenes), Fisheye (2) |
| Absolute pose | 7-Scenes (2), Cambridge Landmarks (5), ETH3D, MegaScenes32k |
| Homography | Barath et al. (2 scenes) |
| Monodepth relative pose | 6 scenes × 2 depth estimators × 2 matchers |

To download datasets manually instead of automatically:

```bash
sh download_data.sh
```

## Maintenance

### Regenerating the subset after adding new datasets

After adding a new dataset to the benchmark:

```bash
# Rebuild the subset from the full dataset (requires full data in data/):
posebench --create_subset 10

# Upload the resulting data-subsampled-10.zip to storage and update the download URL
# in download_data() in src/posebench/__init__.py.

# Then update the committed baseline:
posebench --subset --create_baseline
git add results/baseline-subsampled-10.json
git commit -m "update subset baseline"
```

## TODO

* Add missing estimators (multi-camera, hybrid, etc.)
* Add refinement benchmark
