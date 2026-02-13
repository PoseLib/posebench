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
from posebench.estimators import (
    essential_poselib,
    fundamental_poselib,
    monodepth_calibrated_poselib,
    shared_focal_poselib,
    monodepth_shared_focal_poselib,
    monodepth_varying_focal_poselib,
)


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
    dataset_path="data/relative/monodepth",
    force_opt={},
    dataset_filter=[],
    method_filter=[],
    subsample=None,
):
    datasets = [
        ("florence_cathedral_roma_moge", 2.0),
        ("florence_cathedral_roma_unidepth", 2.0),
        ("florence_cathedral_splg_moge", 2.0),
        ("florence_cathedral_splg_unidepth", 2.0),
        ("lincoln_memorial_roma_moge", 2.0),
        ("lincoln_memorial_roma_unidepth", 2.0),
        ("lincoln_memorial_splg_moge", 2.0),
        ("lincoln_memorial_splg_unidepth", 2.0),
        ("london_bridge_roma_moge", 2.0),
        ("london_bridge_roma_unidepth", 2.0),
        ("london_bridge_splg_moge", 2.0),
        ("london_bridge_splg_unidepth", 2.0),
        ("milan_cathedral_roma_moge", 2.0),
        ("milan_cathedral_roma_unidepth", 2.0),
        ("milan_cathedral_splg_moge", 2.0),
        ("milan_cathedral_splg_unidepth", 2.0),
        ("sagrada_familia_roma_moge", 2.0),
        ("sagrada_familia_roma_unidepth", 2.0),
        ("sagrada_familia_splg_moge", 2.0),
        ("sagrada_familia_splg_unidepth", 2.0),
        ("scannet_roma_moge", 2.0),
        ("scannet_roma_unidepth", 2.0),
        ("scannet_splg_moge", 2.0),
        ("scannet_splg_unidepth", 2.0),
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        "E": lambda i: essential_poselib(i),
        "RePoseD (calibrated)": lambda i: monodepth_calibrated_poselib(i, estimate_shift=False),
        "RePoseD (calibrated) with shift": lambda i: monodepth_calibrated_poselib(i, estimate_shift=True),
        "E + shared f": lambda i: shared_focal_poselib(i),
        "RePoseD (shared focal)": lambda i: monodepth_shared_focal_poselib(i),
        "F": lambda i: fundamental_poselib(i),
        "RePoseD (varying focal)": lambda i: monodepth_varying_focal_poselib(i),
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
            "max_errors": [8 * threshold, threshold],
            "ransac": {
                "max_iterations": 1000,
                "min_iterations": 100,
                "success_prob": 0.9999,
            },
            "bundle": {},
        }

        deep_merge(opt, force_opt)

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
                "K1": v["K1"][:],
                "K2": v["K2"][:],
                "depth1": v["depth1"][:],
                "depth2": v["depth2"][:],
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
    force_opt, method_filter, dataset_filter, subsample, _ = posebench._parse_args()
    metrics, _ = main(
        force_opt=force_opt,
        method_filter=method_filter,
        dataset_filter=dataset_filter,
        subsample=subsample,
    )
    print_metrics_per_dataset(metrics)
