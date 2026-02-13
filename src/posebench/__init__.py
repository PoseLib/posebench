from pathlib import Path
import datetime
import os
import zipfile
from typing import Optional, List, Dict, Any, Tuple

import posebench.absolute_pose
import posebench.relative_pose
import posebench.monodepth_relative_pose
import posebench.homography
import argparse
from posebench.utils.misc import (
    download_file_with_progress,
    print_metrics_per_method_table,
    compute_average_metrics,
    save_results,
    print_comparison,
)

__all__ = ["run_benchmark", "main"]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_iterations", required=False, type=int)
    parser.add_argument("--max_iterations", required=False, type=int)
    parser.add_argument("--success_prob", required=False, type=float)
    parser.add_argument("--initial_lambda", required=False, type=float)
    parser.add_argument("--method", required=False, type=str)
    parser.add_argument("--dataset", required=False, type=str)
    parser.add_argument("--subsample", required=False, type=int)
    parser.add_argument("--subset", required=False, action="store_true")
    parser.add_argument("--output", required=False, type=str, default=None,
                        help="Output path for results JSON (default: results/<timestamp>.json)")
    parser.add_argument("--no-save", required=False, action="store_true",
                        help="Do not save results to file")
    parser.add_argument("--compare", required=False, nargs="+", metavar="FILE",
                        help="Compare result files. With two args: REF NEW. With one arg: compares results/baseline.json vs FILE")
    parser.add_argument("--per-dataset", required=False, action="store_true",
                        help="Show per-dataset breakdowns in comparison (use with --compare)")
    args = parser.parse_args()

    force_opt = {}
    ransac_opt = {}
    if args.min_iterations is not None:
        ransac_opt["min_iterations"] = args.min_iterations
    if args.max_iterations is not None:
        ransac_opt["max_iterations"] = args.max_iterations
    if args.success_prob is not None:
        ransac_opt["success_prob"] = args.success_prob
    if ransac_opt:
        force_opt["ransac"] = ransac_opt

    bundle_opt = {}
    if args.initial_lambda is not None:
        bundle_opt["initial_lambda"] = args.initial_lambda
    if bundle_opt:
        force_opt["bundle"] = bundle_opt

    method_filter = []
    if args.method is not None:
        method_filter = args.method.split(",")
    dataset_filter = []
    if args.dataset is not None:
        dataset_filter = args.dataset.split(",")
    return force_opt, method_filter, dataset_filter, args.subsample, args.subset, args.output, args.no_save, args.compare, args.per_dataset


def download_data(subset: bool = False):
    # Download and extract data if needed
    if not Path("data").is_dir():
        if subset:
            data_zipfile_name = "data-subsampled-10.zip"
        else:
            data_zipfile_name = "data.zip"
        if not Path(data_zipfile_name).is_file():
            print(f"Downloading {data_zipfile_name}...")
            download_file_with_progress(
                f"https://github.com/Parskatt/storage/releases/download/posebench-v0.0.1/{data_zipfile_name}",
                data_zipfile_name,
            )
        print("Extracting data...")
        with zipfile.ZipFile(data_zipfile_name, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(data_zipfile_name)

def run_benchmark(
    min_iterations: Optional[int] = None,
    max_iterations: Optional[int] = None,
    success_prob: Optional[float] = None,
    initial_lambda: Optional[float] = None,
    method_filter: Optional[List[str]] = None,
    dataset_filter: Optional[List[str]] = None,
    subsample: Optional[int] = None,
    subset: bool = False,
    output: Optional[str] = None,
    no_save: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    # Build force_opt dictionary
    force_opt = {}
    ransac_opt = {}
    if min_iterations is not None:
        ransac_opt["min_iterations"] = min_iterations
    if max_iterations is not None:
        ransac_opt["max_iterations"] = max_iterations
    if success_prob is not None:
        ransac_opt["success_prob"] = success_prob
    if ransac_opt:
        force_opt["ransac"] = ransac_opt

    bundle_opt = {}
    if initial_lambda is not None:
        bundle_opt["initial_lambda"] = initial_lambda
    if bundle_opt:
        force_opt["bundle"] = bundle_opt

    # Set defaults for filters
    if method_filter is None:
        method_filter = []
    if dataset_filter is None:
        dataset_filter = []

    download_data(subset)

    # Define problems
    problems = {
        "absolute pose": posebench.absolute_pose.main,
        "relative pose": posebench.relative_pose.main,
        "homography": posebench.homography.main,
        "monodepth pose": posebench.monodepth_relative_pose.main,
    }

    start_time = datetime.datetime.now()
    all_metrics = {}  # {problem_name: {dataset: {method: {metric: value}}}}
    compiled_metrics = []
    dataset_names = []

    for name, problem in problems.items():
        print(f"Running problem {name}")
        metrics, _ = problem(
            force_opt=force_opt,
            method_filter=method_filter,
            dataset_filter=dataset_filter,
            subsample=subsample,
        )

        all_metrics[name] = metrics
        avg_metrics = compute_average_metrics(metrics)
        compiled_metrics.append(avg_metrics)
        dataset_names += metrics.keys()

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(
        f"Finished running evaluation in {total_time:.1f} seconds ({len(dataset_names)} datasets)"
    )
    print("Datasets: " + (",".join(dataset_names)) + "\n")

    # Output all the average metrics
    for avg_metrics in compiled_metrics:
        print_metrics_per_method_table(avg_metrics)
        print("")

    # Save results
    if not no_save:
        if output is None:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            output = f"results/{timestamp}.json"

        metadata = {
            "timestamp": start_time.isoformat(),
            "poselib_version": str(getattr(__import__("poselib"), "__version__", "unknown")),
            "force_opt": force_opt,
            "method_filter": method_filter,
            "dataset_filter": dataset_filter,
            "subsample": subsample,
        }
        save_results(output, all_metrics, metadata)

    return all_metrics, dataset_names


def main():
    """Console script entry point."""
    force_opt, method_filter, dataset_filter, subsample, subset, output, no_save, compare, per_dataset = _parse_args()

    # Compare mode: compare two result files and exit
    if compare is not None:
        if len(compare) == 1:
            ref_path = "results/baseline.json"
            new_path = compare[0]
        elif len(compare) == 2:
            ref_path = compare[0]
            new_path = compare[1]
        else:
            print("Error: --compare takes 1 or 2 arguments")
            return
        print_comparison(ref_path, new_path, per_dataset=per_dataset)
        return

    # Extract individual parameters from force_opt for the function call
    run_benchmark(
        min_iterations=force_opt.get("ransac", {}).get("min_iterations"),
        max_iterations=force_opt.get("ransac", {}).get("max_iterations"),
        success_prob=force_opt.get("ransac", {}).get("success_prob"),
        initial_lambda=force_opt.get("bundle", {}).get("initial_lambda"),
        method_filter=method_filter,
        dataset_filter=dataset_filter,
        subsample=subsample,
        subset=subset,
        output=output,
        no_save=no_save,
    )


if __name__ == "__main__":
    main()
