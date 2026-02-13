import datetime

import cv2
import h5py
import numpy as np
import poselib
import pycolmap
from tqdm import tqdm

import posebench
from posebench.utils.geometry import (
    angle,
    calibrate_pts,
    eigen_quat_to_wxyz,
    essential_from_pose,
    qvec2rotmat,
    rotation_angle,
    sampson_error,
)
from posebench.utils.misc import (
    print_metrics_per_dataset,
    camera_dict_to_calib_matrix,
    compute_auc,
    h5_to_camera_dict,
    poselib_opt_to_pycolmap_opt,
    substr_in_list,
)
from posebench.estimators import essential_poselib, essential_pycolmap, fundamental_poselib, fundamental_pycolmap

# Compute metrics for relative pose estimation
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
    dataset_path="data/relative",
    force_opt={},
    dataset_filter=[],
    method_filter=[],
    subsample=None,
):
    datasets = [
        #('fisheye_grossmunster_4342', 1.0),
        #('fisheye_kirchenge_2731', 1.0),
        ("megadepth1500_sift", 1.0),
        ("megadepth1500_spsg", 1.0),
        ("megadepth1500_splg", 1.0),
        ("megadepth1500_roma", 1.0),
        ("megadepth1500_dkm", 1.0),
        ("megadepth1500_aspanformer", 1.0),
        ("scannet1500_sift", 1.5),
        ("scannet1500_spsg", 1.5),
        ("scannet1500_roma", 2.5),
        ("scannet1500_dkm", 2.5),
        ("scannet1500_aspanformer", 2.5),
        ("imc_british_museum", 0.75),
        ("imc_london_bridge", 0.75),
        ("imc_piazza_san_marco", 0.75),
        ("imc_florence_cathedral_side", 0.75),
        ("imc_milan_cathedral", 0.75),
        ("imc_sagrada_familia", 0.75),
        ("imc_lincoln_memorial_statue", 0.75),
        ("imc_mount_rushmore", 0.75),
        ("imc_st_pauls_cathedral", 0.75),
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        "E (poselib)": lambda i: essential_poselib(i),
        "E (poselib,TS)": lambda i: essential_poselib(i, tangent_sampson=True),
        "E (COLMAP)": lambda i: essential_pycolmap(i),
        "F (poselib)": lambda i: fundamental_poselib(i),
        "F (COLMAP)": lambda i: fundamental_pycolmap(i),
    }
    if len(method_filter) > 0:
        evaluators = {
            k: v for (k, v) in evaluators.items() if substr_in_list(k, method_filter)
        }

    metrics = {}
    full_results = {}
    for dataset, threshold in datasets:
        f = h5py.File(f"{dataset_path}/{dataset}.h5", "r")

        opt = {
            "max_error": threshold,
            "ransac": {
                "max_iterations": 10000,
                "min_iterations": 100,
                "success_prob": 0.9999,
            },
            "bundle": {
                "loss_type": "CAUCHY",
                "loss_scale": 0.5 * threshold,
                "gradient_tol": 1e-10
            }
        }

        for k, v in force_opt.items():
            opt[k] = v

        results = {}
        for k in evaluators.keys():
            results[k] = {}
        data = list(f.items())
        if subsample is not None:
            print(
                f"Subsampling {len(data)} instances to {len(data) // subsample} instances"
            )
            data = data[::subsample]

        for k, v in tqdm(data, desc=dataset):
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
