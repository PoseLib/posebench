
import h5py
import numpy as np
from tqdm import tqdm

import posebench
from posebench.utils.misc import (
    deep_merge,
    print_metrics_per_dataset,
    compute_auc,
    h5_to_camera_dict,
    substr_in_list,
)
from posebench.estimators import absolute_pose_poselib, absolute_pose_pycolmap

DATASET_SUBPATH = "absolute"
DATASETS = [
    ("eth3d_130_dusmanu", 12.0),
    ("7scenes_heads", 5.0),
    ("7scenes_stairs", 5.0),
    ("cambridge_landmarks_GreatCourt", 6.0),
    ("cambridge_landmarks_ShopFacade", 6.0),
    ("cambridge_landmarks_KingsCollege", 6.0),
    ("cambridge_landmarks_StMarysChurch", 6.0),
    ("cambridge_landmarks_OldHospital", 6.0),
    ("MegaScenes32k", 6.0),
]

# Compute metrics for absolute pose estimation
# AUC for camera center and avg/med for runtime
def compute_metrics(results, thresholds=[0.1, 1.0, 5.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        rot_err = results[m]["rot"]
        cc_err = results[m]["pos"]
        metrics[m] = {}
        aucs = compute_auc(cc_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f"AUC{int(t)}"] = auc
        metrics[m]["med_rot"] = np.median(rot_err)
        metrics[m]["med_pos"] = np.median(cc_err)
        metrics[m]["avg_rt"] = np.mean(results[m]["rt"])
        metrics[m]["med_rt"] = np.median(results[m]["rt"])

    return metrics


def main(
    data_root="data",
    force_opt={},
    dataset_filter=[],
    method_filter=[],
    subsample=None,
):
    dataset_path = f"{data_root}/{DATASET_SUBPATH}"
    datasets = list(DATASETS)

    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        "PnP (poselib)": lambda i: absolute_pose_poselib(i),
        "PnP (COLMAP)": lambda i: absolute_pose_pycolmap(i),

        "PnPf (poselib)": lambda i: absolute_pose_poselib(i, estimate_focal_length=True),
        "PnPf (COLMAP)": lambda i: absolute_pose_pycolmap(i, estimate_focal_length=True),

        "PnPfr (poselib)": lambda i: absolute_pose_poselib(i, estimate_focal_length=True, estimate_extra_params=True),
        "PnPfr (COLMAP)": lambda i: absolute_pose_pycolmap(i, estimate_focal_length=True, estimate_extra_params=True),
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
            "ransac": {
                "max_iterations": 1000,
                "min_iterations": 100,
                "success_prob": 0.9999,
            }
        }

        # Add in global overrides
        deep_merge(opt, force_opt)

        data = list(f.items())
        if subsample is not None:
            print(
                f"Subsampling {len(data)} instances to {len(data) // subsample} instances"
            )
            data = data[::subsample]

        for k, v in tqdm(data, desc=dataset, total=len(data)):
            instance = {
                "p2d": v["p2d"][:],
                "p3d": v["p3d"][:],
                "cam": h5_to_camera_dict(v["camera"]),
                "R": v["R"][:],
                "t": v["t"][:],
                "threshold": threshold,
                "opt": opt,
            }

            # Check if we have 2D-3D line correspondences
            if "l2d" in v:
                instance["l2d"] = v["l2d"][:]
                instance["l3d"] = v["l3d"][:]
            else:
                instance["l2d"] = np.zeros((0, 4))
                instance["l3d"] = np.zeros((0, 6))

            # Run each of the evaluators
            for name, fcn in evaluators.items():
                errs = fcn(instance)
                for k,v in errs.items():
                    if k not in results[name]:
                        results[name][k] = []
                    results[name][k].append(v)

        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results

    return metrics, full_results


if __name__ == "__main__":
    force_opt, method_filter, dataset_filter, subsample, subset, *_ = posebench._parse_args()
    data_root = posebench.download_data(subset)
    metrics, _ = main(
        data_root=data_root, force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter, subsample=subsample
    )
    print_metrics_per_dataset(metrics)