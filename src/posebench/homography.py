import datetime

import cv2
import h5py
import numpy as np
import poselib
import pycolmap
from tqdm import tqdm

import posebench
from posebench.utils.geometry import angle, rotation_angle
from posebench.utils.misc import (
    print_metrics_per_dataset,
    camera_dict_to_calib_matrix,
    compute_auc,
    h5_to_camera_dict,
    poselib_opt_to_pycolmap_opt,
    substr_in_list,
)
from posebench.estimators import homography_poselib, homography_pycolmap


# Compute metrics for homography estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds=[5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [np.max((a, b)) for (a, b) in zip(results[m]["rot"], results[m]["t"])]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f"AUC{int(t)}"] = auc
        metrics[m]["avg_rt"] = np.mean(results[m]["rt"])
        metrics[m]["med_rt"] = np.median(results[m]["rt"])

    return metrics


def main(
    dataset_path="data/homography",
    force_opt={},
    dataset_filter=[],
    method_filter=[],
    subsample=None,
):
    datasets = [
        ("barath_Alamo", 1.0),
        ("barath_NYC_Library", 1.0),
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        "H (poselib)": lambda i: homography_poselib(i),
        "H (COLMAP)": lambda i: homography_pycolmap(i),
    }
    if len(method_filter) > 0:
        evaluators = {
            k: v for (k, v) in evaluators.items() if substr_in_list(k, method_filter)
        }

    metrics = {}
    full_results = {}
    for dataset, threshold in datasets:
        f = h5py.File(f"{dataset_path}/{dataset}.h5", "r")

        results = {}
        for k in evaluators.keys():
            results[k] = {}

        # RANSAC options
        opt = {
            "max_error": threshold,
            "ransac":{
                "max_iterations": 1000,
                "min_iterations": 100,
                "success_prob": 0.9999,
            }
        }

        # Add in global overrides
        for k, v in force_opt.items():
            opt[k] = v

        # Since the datasets are so large we only take the first 1k pairs
        pairs = list(f.keys())
        if subsample is not None:
            print(
                f"Subsampling {len(pairs)} instances to {len(pairs) // subsample} instances"
            )
            pairs = pairs[::subsample]

        for k in tqdm(pairs, desc=dataset, total=len(pairs)):
            v = f[k]
            instance = {
                "x1": v["x1"][:],
                "x2": v["x2"][:],
                "cam1": h5_to_camera_dict(v["camera1"]),
                "cam2": h5_to_camera_dict(v["camera2"]),
                "R": v["R"][:],
                "t": v["t"][:],
                "threshold": threshold,
                "opt": opt,
            }

            for name, fcn in evaluators.items():
                errs = fcn(instance)
                for key in errs.keys():
                    if key not in results[name]:
                        results[name][key] = []
                    results[name][key].append(errs[key])
        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results
    return metrics, full_results


if __name__ == "__main__":
    force_opt, method_filter, dataset_filter, subsample, subset = posebench._parse_args()
    posebench.download_data(subset)
    metrics, _ = main(
        force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter, subsample=subsample
    )
    print_metrics_per_dataset(metrics)