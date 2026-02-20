import json
from pathlib import Path

import numpy as np
import pycolmap
import requests
from tqdm import tqdm
import poselib


def trapezoid(y, x=None, dx=1.0, axis=-1):
    if np.__version__ < "2.0.0":
        return np.trapz(y, x=x, dx=dx, axis=axis)
    else:
        return np.trapezoid(y, x=x, dx=dx, axis=axis)


def deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def substr_in_list(s, lst):
    return np.any([s.find(t) >= 0 for t in lst])


def poselib_opt_to_pycolmap_opt(opt):
    pyc_opt = pycolmap.RANSACOptions()

    if "max_error" in opt:
        pyc_opt.max_error = opt["max_error"]

    if "ransac" in opt:
        if "max_iterations" in opt["ransac"]:
            pyc_opt.max_num_trials = opt["ransac"]["max_iterations"]
        if "min_iterations" in opt["ransac"]:
            pyc_opt.min_num_trials = opt["ransac"]["min_iterations"]
        if "success_prob" in opt["ransac"]:
            pyc_opt.confidence = opt["ransac"]["success_prob"]

    return pyc_opt


def h5_to_camera_dict(data):
    camera_dict = {}
    camera_dict["model"] = data["model"].asstr()[0]
    camera_dict["width"] = int(data["width"][0])
    camera_dict["height"] = int(data["height"][0])
    camera_dict["params"] = data["params"][:]
    return camera_dict


def calib_matrix_to_camera_dict(K):
    camera_dict = {}
    camera_dict["model"] = "PINHOLE"
    camera_dict["width"] = int(np.ceil(K[0, 2] * 2))
    camera_dict["height"] = int(np.ceil(K[1, 2] * 2))
    camera_dict["params"] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    return camera_dict


def camera_dict_to_calib_matrix(cam):
    cam = poselib.Camera(cam)
    return cam.calib_matrix()


# From Paul
def compute_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(trapezoid(r, x=e) / t)
    return aucs


def format_metric(name, value):
    name = name.upper()
    if "AUC" in name:
        return f"{100.0 * value:>6.2f}"
    elif "ROT" in name:
        return f"{value:-3.1E}"
    elif "INL" in name:
        return f"{value:-3.5f}"
    elif "RMS" in name:
        return f"{value:-3.4f}px"
    elif "MSAC" in name:
        return f"{value:-3.4f}px"
    elif "POS" in name:
        return f"{value:-3.1f}"
    elif "TIME" in name or "RT" in name:
        if value < 1e-6:
            return f"{1e9 * value:-5.1f}ns"
        elif value < 1e-3:
            return f"{1e6 * value:-5.1f}us"
        elif value < 1.0:
            return f"{1e3 * value:-5.1f}ms"
        else:
            return f"{value:.2}s"
    else:
        return f"{value}"


def print_metrics_per_method(metrics):
    for name, res in metrics.items():
        s = f"{name:18s}: "
        for metric_name, value in res.items():
            s = s + f"{metric_name}={format_metric(metric_name, value)}" + ", "
        s = s[0:-2]
        print(s)


def print_metrics_per_method_table(metrics, sort_by_metric=None, reverse_sort=False):
    method_names = list(metrics.keys())
    if len(method_names) == 0:
        return
    metric_names = list(metrics[method_names[0]].keys())

    if sort_by_metric is not None:
        vals = [metrics[m][sort_by_metric] for m in method_names]
        if reverse_sort:
            ind = np.argsort(-np.array(vals))
        else:
            ind = np.argsort(np.array(vals))
        method_names = np.array(method_names)[ind]

    field_lengths = {x: len(x) + 2 for x in metric_names}
    name_length = np.max([len(x) for x in metrics.keys()])

    # print header
    print(f"{'':{name_length}s}", end=" ")
    for metric_name in metric_names:
        print(f"{metric_name:>{field_lengths[metric_name]}s}", end=" ")
    print("")

    for name in method_names:
        res = metrics[name]
        print(f"{name:{name_length}s}", end=" ")
        for metric_name, value in res.items():
            print(
                f"{format_metric(metric_name, value):>{field_lengths[metric_name]}s}",
                end=" ",
            )
        print("")


def print_metrics_per_dataset(
    metrics, as_table=True, sort_by_metric=None, reverse_sort=False
):
    for dataset in metrics.keys():
        print(f"Dataset: {dataset}")
        if as_table:
            print_metrics_per_method_table(
                metrics[dataset],
                sort_by_metric=sort_by_metric,
                reverse_sort=reverse_sort,
            )
        else:
            print_metrics_per_method(metrics[dataset])


def compute_average_metrics(metrics):
    avg_metrics = {}
    for dataset, dataset_metrics in metrics.items():
        for method, res in dataset_metrics.items():
            if method not in avg_metrics:
                avg_metrics[method] = {}

            for m_name, m_val in res.items():
                if m_name not in avg_metrics[method]:
                    avg_metrics[method][m_name] = []
                avg_metrics[method][m_name].append(m_val)

    for method in avg_metrics.keys():
        for m_name, m_vals in avg_metrics[method].items():
            avg_metrics[method][m_name] = np.mean(m_vals)

    return avg_metrics


def save_results(path, all_metrics, metadata=None):
    """Save benchmark results to a JSON file.

    Args:
        path: Output file path.
        all_metrics: Dict of {problem_name: {dataset: {method: {metric: value}}}}.
        metadata: Optional dict with run metadata (timestamp, options, etc.).
    """
    results = {
        "metadata": metadata or {},
        "problems": {},
    }
    for problem_name, metrics in all_metrics.items():
        # Convert numpy values to Python floats for JSON serialization
        datasets = {}
        for dataset, dataset_metrics in metrics.items():
            datasets[dataset] = {}
            for method, res in dataset_metrics.items():
                datasets[dataset][method] = {
                    k: float(v) for k, v in res.items()
                }

        avg = compute_average_metrics(metrics)
        average = {}
        for method, res in avg.items():
            average[method] = {k: float(v) for k, v in res.items()}

        results["problems"][problem_name] = {
            "datasets": datasets,
            "average": average,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path):
    """Load benchmark results from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def print_comparison(ref_path, new_path, per_dataset=False):
    """Load two result files and print a comparison table.

    ref_path is the reference/baseline, new_path is the new run.
    Shows new values with delta from reference. Methods/datasets only in one
    file are shown with N/A. Averaged metrics only use shared datasets.

    If per_dataset is True, also prints per-dataset breakdowns.
    """
    ref = load_results(ref_path)
    new = load_results(new_path)

    # Print metadata for both runs
    ref_meta = ref.get("metadata", {})
    new_meta = new.get("metadata", {})
    print(f"REF: {ref_path}")
    print(f"  timestamp: {ref_meta.get('timestamp', 'N/A')}")
    print(f"  poselib:   {ref_meta.get('poselib_version', 'N/A')}")
    if ref_meta.get("force_opt"):
        print(f"  options:   {json.dumps(ref_meta['force_opt'])}")
    print(f"NEW: {new_path}")
    print(f"  timestamp: {new_meta.get('timestamp', 'N/A')}")
    print(f"  poselib:   {new_meta.get('poselib_version', 'N/A')}")
    if new_meta.get("force_opt"):
        print(f"  options:   {json.dumps(new_meta['force_opt'])}")
    print("")
    print("Values shown are from NEW, with (delta = NEW - REF) in parentheses.")
    print("")

    all_problems = list(dict.fromkeys(
        list(ref["problems"].keys()) + list(new["problems"].keys())
    ))

    # Accumulate change summary across all problems
    summary_better = {"slight": 0, "moderate": 0, "large": 0}
    summary_worse = {"slight": 0, "moderate": 0, "large": 0}
    summary_same = [0]

    for problem in all_problems:
        ref_prob = ref["problems"].get(problem, {"datasets": {}, "average": {}})
        new_prob = new["problems"].get(problem, {"datasets": {}, "average": {}})

        all_datasets = list(dict.fromkeys(
            list(ref_prob["datasets"].keys()) + list(new_prob["datasets"].keys())
        ))

        # Per-dataset comparison (only if requested)
        if per_dataset:
            for dataset in all_datasets:
                ref_ds = ref_prob["datasets"].get(dataset, {})
                new_ds = new_prob["datasets"].get(dataset, {})
                if not ref_ds and not new_ds:
                    continue
                print(f"=== {problem} / {dataset} ===")
                _print_comparison_block(ref_ds, new_ds)
                print("")

        # Average comparison (only shared datasets)
        shared_datasets = [
            d for d in all_datasets
            if d in ref_prob["datasets"] and d in new_prob["datasets"]
        ]
        if shared_datasets:
            ref_shared = {d: ref_prob["datasets"][d] for d in shared_datasets}
            new_shared = {d: new_prob["datasets"][d] for d in shared_datasets}
            ref_avg = _compute_average_from_datasets(ref_shared)
            new_avg = _compute_average_from_datasets(new_shared)
            if ref_avg or new_avg:
                print(f"=== {problem} (average over {len(shared_datasets)} shared datasets) ===")
                _print_comparison_block(ref_avg, new_avg)
                _accumulate_changes(ref_avg, new_avg, summary_better, summary_worse, summary_same)
                print("")

    # Print global change summary
    print("=== Summary (across all problems) ===")
    _print_change_summary(summary_better, summary_worse, summary_same[0])


def _compute_average_from_datasets(datasets_metrics):
    """Compute average metrics from a dict of {dataset: {method: {metric: val}}}."""
    acc = {}
    for dataset_metrics in datasets_metrics.values():
        for method, res in dataset_metrics.items():
            if method not in acc:
                acc[method] = {}
            for m_name, m_val in res.items():
                if m_name not in acc[method]:
                    acc[method][m_name] = []
                acc[method][m_name].append(m_val)

    avg = {}
    for method in acc:
        avg[method] = {}
        for m_name, m_vals in acc[method].items():
            avg[method][m_name] = np.mean(m_vals)
    return avg


def _format_metric_with_delta(name, new_val, ref_val):
    """Format a metric value with delta from reference."""
    if ref_val is None:
        return format_metric(name, new_val)
    if new_val is None:
        return "N/A"
    delta = new_val - ref_val
    formatted = format_metric(name, new_val)
    # Format delta in the same style
    name_upper = name.upper()
    if "AUC" in name_upper:
        delta_str = f"{100.0 * delta:+.2f}"
    elif "ROT" in name_upper:
        delta_str = f"{delta:+.1E}"
    elif "POS" in name_upper:
        delta_str = f"{delta:+.1f}"
    elif "TIME" in name_upper or "RT" in name_upper:
        # Show delta in same unit as the value
        if abs(new_val) < 1e-6:
            delta_str = f"{1e9 * delta:+.1f}ns"
        elif abs(new_val) < 1e-3:
            delta_str = f"{1e6 * delta:+.1f}us"
        elif abs(new_val) < 1.0:
            delta_str = f"{1e3 * delta:+.1f}ms"
        else:
            delta_str = f"{delta:+.2f}s"
    else:
        delta_str = f"{delta:+.4f}"
    return f"{formatted} ({delta_str})"


def _classify_change(metric_name, ref_val, new_val):
    """Classify a metric change with direction and magnitude.

    Returns (direction, magnitude) where:
      direction: 'better', 'worse', 'same', or None (skip)
      magnitude: 'slight', 'moderate', 'large', or None

    AUC thresholds (percentage points): <0.5 same, 0.5-2 slight, 2-5 moderate, >5 large
    Error thresholds (relative %): <1% same, 1-5% slight, 5-20% moderate, >20% large
    """
    if ref_val is None or new_val is None:
        return None, None

    name = metric_name.upper()

    # Skip runtime metrics (too hardware-dependent)
    if "TIME" in name or "RT" in name:
        return None, None

    delta = new_val - ref_val

    if "AUC" in name:
        ppt = abs(delta) * 100.0  # percentage points
        if ppt < 0.25:
            return "same", None
        direction = "better" if delta > 0 else "worse"
        if ppt < 1.0:
            return direction, "slight"
        elif ppt < 2.5:
            return direction, "moderate"
        else:
            return direction, "large"

    # Error metrics (rot, pos): lower is better
    if ref_val != 0:
        rel = abs(delta / ref_val) * 100.0  # relative %
    else:
        rel = 0.0 if abs(delta) < 1e-10 else 100.0

    if rel < 1.0:
        return "same", None
    direction = "better" if delta < 0 else "worse"
    if rel < 5.0:
        return direction, "slight"
    elif rel < 20.0:
        return direction, "moderate"
    else:
        return direction, "large"


def _accumulate_changes(ref_metrics, new_metrics, better, worse, same_count):
    """Accumulate change classifications from a pair of metric dicts into the provided counters.

    better/worse are dicts with keys 'slight', 'moderate', 'large'.
    same_count is a list with a single int element (mutable counter).
    """
    all_methods = list(dict.fromkeys(
        list(ref_metrics.keys()) + list(new_metrics.keys())
    ))
    for method in all_methods:
        ref_m = ref_metrics.get(method, {})
        new_m = new_metrics.get(method, {})
        all_metric_names = list(dict.fromkeys(
            list(ref_m.keys()) + list(new_m.keys())
        ))
        for mn in all_metric_names:
            direction, magnitude = _classify_change(mn, ref_m.get(mn), new_m.get(mn))
            if direction == "better":
                better[magnitude] += 1
            elif direction == "worse":
                worse[magnitude] += 1
            elif direction == "same":
                same_count[0] += 1


def _print_change_summary(better, worse, same):
    """Print a summary of how many metrics improved/degraded/unchanged, with magnitude bucketing."""
    total_better = sum(better.values())
    total_worse = sum(worse.values())
    total = total_better + total_worse + same
    if total == 0:
        return

    def _fmt_bucket(counts, total):
        breakdown = []
        for label in ("large", "moderate", "slight"):
            if counts[label] > 0:
                breakdown.append(f"{counts[label]} {label}")
        return f"{sum(counts.values())}/{total} ({', '.join(breakdown)})"

    parts = []
    if total_better > 0:
        parts.append(f"better: {_fmt_bucket(better, total)}")
    if total_worse > 0:
        parts.append(f"worse: {_fmt_bucket(worse, total)}")
    if same > 0:
        parts.append(f"unchanged: {same}/{total}")
    print(f"  {' | '.join(parts)}")


_COLOR_GREEN = "\033[32m"
_COLOR_RED = "\033[31m"
_COLOR_DIM = "\033[2m"
_COLOR_RESET = "\033[0m"


def _colorize(text, direction):
    """Wrap text with ANSI color based on change direction."""
    if direction == "better":
        return f"{_COLOR_GREEN}{text}{_COLOR_RESET}"
    elif direction == "worse":
        return f"{_COLOR_RED}{text}{_COLOR_RESET}"
    elif direction == "same":
        return f"{_COLOR_DIM}{text}{_COLOR_RESET}"
    return text


def _print_comparison_block(ref_metrics, new_metrics):
    """Print a comparison table for two method->metric dicts."""
    all_methods = list(dict.fromkeys(
        list(ref_metrics.keys()) + list(new_metrics.keys())
    ))
    if not all_methods:
        print("  (no data)")
        return

    # Collect all metric names (preserving order from ref first, then new)
    all_metric_names = []
    for m in all_methods:
        for mn in list(ref_metrics.get(m, {}).keys()) + list(new_metrics.get(m, {}).keys()):
            if mn not in all_metric_names:
                all_metric_names.append(mn)

    if not all_metric_names:
        print("  (no metrics)")
        return

    # Compute formatted cells and classify changes
    cells = {}  # (method, metric) -> formatted string (no color)
    colors = {}  # (method, metric) -> direction for coloring
    for method in all_methods:
        ref_m = ref_metrics.get(method, {})
        new_m = new_metrics.get(method, {})
        for metric_name in all_metric_names:
            ref_val = ref_m.get(metric_name)
            new_val = new_m.get(metric_name)
            if new_val is not None:
                cells[(method, metric_name)] = _format_metric_with_delta(
                    metric_name, new_val, ref_val
                )
                direction, _ = _classify_change(metric_name, ref_val, new_val)
                colors[(method, metric_name)] = direction
            elif ref_val is not None:
                cells[(method, metric_name)] = "N/A"
                colors[(method, metric_name)] = None
            else:
                cells[(method, metric_name)] = ""
                colors[(method, metric_name)] = None

    name_length = max(len(m) for m in all_methods)
    col_widths = {}
    for metric_name in all_metric_names:
        w = len(metric_name) + 2
        for method in all_methods:
            w = max(w, len(cells.get((method, metric_name), "")) + 2)
        col_widths[metric_name] = w

    # Print header
    print(f"{'':{name_length}s}", end=" ")
    for metric_name in all_metric_names:
        print(f"{metric_name:>{col_widths[metric_name]}s}", end=" ")
    print("")

    # Print rows with color
    for method in all_methods:
        print(f"{method:{name_length}s}", end=" ")
        for metric_name in all_metric_names:
            cell = cells.get((method, metric_name), "")
            padded = f"{cell:>{col_widths[metric_name]}s}"
            direction = colors.get((method, metric_name))
            print(_colorize(padded, direction), end=" ")
        print("")


def download_file_with_progress(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm(
        desc=f"Downloading {filename}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))
