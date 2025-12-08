import copy
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
    camera_dict_to_calib_matrix,
    compute_auc,
    h5_to_camera_dict,
    poselib_opt_to_pycolmap_opt,
    substr_in_list,
)


# Compute metrics for relative pose estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds=[5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [np.max((a, b)) for (a, b) in results[m]["errs"]]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f"AUC{int(t)}"] = auc
        metrics[m]["avg_rt"] = np.mean(results[m]["runtime"])
        metrics[m]["med_rt"] = np.median(results[m]["runtime"])

    return metrics


def eval_essential_estimator(instance, estimator="poselib"):
    opt = instance["opt"]
    if estimator == "poselib":
        tt1 = datetime.datetime.now()
        pose, info = poselib.estimate_relative_pose(
            instance["x1"], instance["x2"], instance["cam1"], instance["cam2"], opt, {}
        )
        tt2 = datetime.datetime.now()
        (R, t) = (pose.R, pose.t)
    elif estimator == "pycolmap":
        opt = poselib_opt_to_pycolmap_opt(opt)
        tt1 = datetime.datetime.now()
        result = pycolmap.estimate_essential_matrix(
            instance["x1"], instance["x2"], instance["cam1"], instance["cam2"], opt
        )
        tt2 = datetime.datetime.now()

        if result is not None:
            R = qvec2rotmat(eigen_quat_to_wxyz(result["cam2_from_cam1"].rotation.quat))
            t = result["cam2_from_cam1"].translation
        else:
            R = np.eye(3)
            t = np.zeros(3)
    else:
        raise Exception("nyi")

    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()

def eval_calib_monodepth_estimator(instance, estimate_shift=False):
    opt = instance["opt"]
    opt['monodepth_estimate_shift'] = estimate_shift

    tt1 = datetime.datetime.now()
    monodepth_geometry, info = poselib.estimate_monodepth_relative_pose(
        instance["x1"], instance["x2"], instance['depth1'], instance['depth2'], instance["cam1"], instance["cam2"],
        opt, {}
    )
    tt2 = datetime.datetime.now()
    (R, t) = (monodepth_geometry.pose.R, monodepth_geometry.pose.t)

    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()

def eval_fundamental_estimator(instance):
    opt = instance["opt"]

    tt1 = datetime.datetime.now()
    F, info = poselib.estimate_fundamental(instance["x1"], instance["x2"], opt, {})
    tt2 = datetime.datetime.now()
    inl = info["inliers"]

    K1 = camera_dict_to_calib_matrix(instance["cam1"])
    K2 = camera_dict_to_calib_matrix(instance["cam2"])
    if np.sum(inl) < 5:
        return [180.0, 180.0], (tt2 - tt1).total_seconds()

    E = K2.T @ F @ K1
    x1i = calibrate_pts(instance["x1"][inl], K1)
    x2i = calibrate_pts(instance["x2"][inl], K2)

    _, R, t, good = cv2.recoverPose(E, x1i, x2i)
    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()


def force_instance_to_share_focals(instance):
    new_instance = copy.deepcopy(instance)

    if np.any(new_instance['K1'] != new_instance['K2']):
        kp2_h = np.column_stack([new_instance['x2'], np.ones(len(new_instance['x2']))])
        kp2_new = (new_instance['K1'] @ (np.linalg.inv(new_instance['K2']) @ kp2_h.T)).T
        kp2_new = kp2_new[:, :2] / kp2_new[:, 2:]

        new_instance['K2'] = new_instance['K1']
        new_instance['x2'] = kp2_new
        new_instance['cam2'] = new_instance['cam1']

    return new_instance

def eval_shared_focal_estimator(instance):
    opt = instance["opt"]

    instance = force_instance_to_share_focals(instance)

    pp1 = instance["K1"][:2, 2]
    pp2 = instance["K2"][:2, 2]

    tt1 = datetime.datetime.now()
    image_pair, info = poselib.estimate_shared_focal_relative_pose(
        instance["x1"] - pp1, instance["x2"] - pp2, np.array([0.0, 0.0]), opt, {}
    )
    tt2 = datetime.datetime.now()
    pose = image_pair.pose
    (R, t) = (pose.R, pose.t)

    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()

def eval_shared_focal_monodepth_estimator(instance):
    opt = instance["opt"]

    instance = force_instance_to_share_focals(instance)

    pp1 = instance["K1"][:2, 2]
    pp2 = instance["K2"][:2, 2]

    tt1 = datetime.datetime.now()
    image_pair, info = poselib.estimate_monodepth_shared_focal_relative_pose(
        instance["x1"] - pp1, instance["x2"] - pp2, instance['depth1'], instance['depth2'], opt, {}
    )
    tt2 = datetime.datetime.now()
    pose = image_pair.geometry.pose
    (R, t) = (pose.R, pose.t)

    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()

def eval_varying_focal_monodepth_estimator(instance):
    opt = instance["opt"]

    pp1 = instance["K1"][:2, 2]
    pp2 = instance["K2"][:2, 2]

    tt1 = datetime.datetime.now()
    image_pair, info = poselib.estimate_monodepth_varying_focal_relative_pose(
        instance["x1"] - pp1, instance["x2"] - pp2, instance['depth1'], instance['depth2'], opt, {}
    )
    tt2 = datetime.datetime.now()
    pose = image_pair.geometry.pose
    (R, t) = (pose.R, pose.t)

    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return [err_R, err_t], (tt2 - tt1).total_seconds()


def main(
    dataset_path="data/monodepth",
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
        ("scannet_splg_unidepth", 2.0)
    ]
    if len(dataset_filter) > 0:
        datasets = [(n, t) for (n, t) in datasets if substr_in_list(n, dataset_filter)]

    evaluators = {
        "E": lambda i: eval_essential_estimator(i),
        "RePoseD (calibrated)": lambda i: eval_calib_monodepth_estimator(i, estimate_shift=False),
        "RePoseD (calibrated) with shift": lambda i: eval_calib_monodepth_estimator(i, estimate_shift=True),
        "E + shared f": lambda i: eval_shared_focal_estimator(i),
        "RePoseD (shared focal)": lambda i: eval_shared_focal_monodepth_estimator(i),
        "F": lambda i: eval_fundamental_estimator(i),
        "RePoseD (varying focal)": lambda i: eval_varying_focal_monodepth_estimator(i),
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
            "max_reproj_error": 8 * threshold,
            "max_epipolar_error": threshold,
            "max_iterations": 1000,
            "min_iterations": 100,
            "success_prob": 0.9999,
        }

        for k, v in force_opt.items():
            opt[k] = v

        results = {}
        for k in evaluators.keys():
            results[k] = {"errs": [], "runtime": []}
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
                errs, runtime = fcn(instance)
                results[name]["errs"].append(np.array(errs))
                results[name]["runtime"].append(runtime)
        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results
    return metrics, full_results


if __name__ == "__main__":
    force_opt, method_filter, dataset_filter = posebench.parse_args()
    metrics, _ = main(
        force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter
    )
    posebench.print_metrics_per_dataset(metrics)
