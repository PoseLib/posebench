import poselib
import pycolmap
import datetime
import numpy as np
from posebench.utils.geometry import rotation_angle, eigen_quat_to_wxyz, qvec2rotmat, angle, calibrate_pts
from posebench.utils.misc import poselib_opt_to_pycolmap_opt, camera_dict_to_calib_matrix
import cv2


def clean_camera(cam, estimate_focal_length=False, estimate_extra_params=False):
    cam = poselib.Camera(cam.copy())
    if estimate_focal_length:
        cam.set_focal(1.0)
        for k in cam.extra_idx():
            cam.params[k] = 0.0

    return cam.todict()

def absolute_pose_poselib(instance, estimate_focal_length=False, estimate_extra_params=False):
    opt = instance["opt"].copy()
    cam = clean_camera(instance["cam"], estimate_focal_length, estimate_extra_params)
    if estimate_focal_length:
        opt["estimate_focal_length"] = True
    if estimate_extra_params:
        opt["estimate_extra_params"] = True

    tt1 = datetime.datetime.now()
    image, info = poselib.estimate_absolute_pose(
        instance["p2d"], instance["p3d"], cam, opt
    )
    tt2 = datetime.datetime.now()
    #import ipdb
    #ipdb.set_trace()

    (R, t) = (image.pose.R, image.pose.t)
    runtime = (tt2 - tt1).total_seconds()
    err_R = rotation_angle(instance["R"] @ R.T)
    err_c = np.linalg.norm(instance["R"].T @ instance["t"] - R.T @ t)

    if not estimate_focal_length:
        return {'rot': err_R, 'pos': err_c, 'rt': runtime}
    else:
        cam_gt = poselib.Camera(instance["cam"])
        focal_err = (cam_gt.focal() - image.camera.focal()) / cam_gt.focal()
        return {'rot': err_R, 'pos': err_c, 'focal': focal_err, 'rt': runtime}
    

def absolute_pose_pycolmap(instance, estimate_focal_length=False, estimate_extra_params=False):
    opt = poselib_opt_to_pycolmap_opt(instance["opt"])
    cam = clean_camera(instance["cam"], estimate_focal_length, estimate_extra_params)
    cam = pycolmap.Camera(cam)
    tt1 = datetime.datetime.now()
    result = pycolmap.estimate_and_refine_absolute_pose(
        instance["p2d"],
        instance["p3d"],
        cam,
        {"estimate_focal_length": estimate_focal_length, "ransac": opt},
    )
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()
    if result is None:
        err_R = 180
        err_c = np.inf
    else:
        R = qvec2rotmat(eigen_quat_to_wxyz(result["cam_from_world"].rotation.quat))
        t = result["cam_from_world"].translation
        err_R = rotation_angle(instance["R"] @ R.T)
        err_c = np.linalg.norm(instance["R"].T @ instance["t"] - R.T @ t)
    
    if not estimate_focal_length:
        return {'rot': err_R, 'pos': err_c, 'rt': runtime}
    else:
#        cam_dict = cam.todict()
#        cam_dict['model'] = instance['cam']['model']
        cam_gt = poselib.Camera(instance['cam'])
#        cam_est = poselib.Camera(cam_dict)

        focal_err = (cam_gt.focal() - cam.mean_focal_length()) / cam_gt.focal()
        return {'rot': err_R, 'pos': err_c, 'focal': focal_err, 'rt': runtime}
    



def homography_error(H, instance):
    K1 = camera_dict_to_calib_matrix(instance["cam1"])
    K2 = camera_dict_to_calib_matrix(instance["cam2"])
    Hnorm = np.linalg.inv(K2) @ H @ K1
    _, rotations, translations, _ = cv2.decomposeHomographyMat(Hnorm, np.identity(3))

    best_err_R = 180.0
    best_err_t = 180.0
    for k in range(len(rotations)):
        R = rotations[k]
        t = translations[k][:, 0]

        err_R = rotation_angle(instance["R"] @ R.T)
        err_t = angle(instance["t"], t)

        if err_R + err_t < best_err_R + best_err_t:
            best_err_R = err_R
            best_err_t = err_t

    return best_err_R, best_err_t




def essential_poselib(instance, tangent_sampson=False):
    opt = instance["opt"].copy()
    if tangent_sampson:
        opt['tangent_sampson'] = True

    tt1 = datetime.datetime.now()
    pose, info = poselib.estimate_relative_pose(
        instance["x1"], instance["x2"], instance["cam1"], instance["cam2"], opt
    )
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()
    (R, t) = (pose.R, pose.t)
    
    err_R = rotation_angle(instance["R"] @ R.T)
    err_t = angle(instance["t"], t)

    return {'rot': err_R, 't': err_t, 'rt': runtime}

def essential_pycolmap(instance):
    opt = poselib_opt_to_pycolmap_opt(instance["opt"])
    tt1 = datetime.datetime.now()
    result = pycolmap.estimate_essential_matrix(
        instance["x1"], instance["x2"], instance["cam1"], instance["cam2"], opt
    )
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()

    if result is not None:
        R = qvec2rotmat(eigen_quat_to_wxyz(result["cam2_from_cam1"].rotation.quat))
        t = result["cam2_from_cam1"].translation
        err_R = rotation_angle(instance["R"] @ R.T)
        err_t = angle(instance["t"], t)
    else:
        err_R = 180.0
        err_t = 180.0

    return {'rot': err_R, 't': err_t, 'rt': runtime}



def fundamental_error(F, instance):
    K1 = camera_dict_to_calib_matrix(instance["cam1"])
    K2 = camera_dict_to_calib_matrix(instance["cam2"])
    E = K2.T @ F @ K1

    poses = poselib.motion_from_essential(E, np.zeros((0,3)), np.zeros((0,3)))

    best_err_R = 180.0
    best_err_t = 180.0
    for k in range(len(poses)):
        R = poses[k].R
        t = poses[k].t

        err_R = rotation_angle(instance["R"] @ R.T)
        err_t = angle(instance["t"], t)

        if err_R + err_t < best_err_R + best_err_t:
            best_err_R = err_R
            best_err_t = err_t

    return best_err_R, best_err_t



def fundamental_poselib(instance):
    opt = instance["opt"].copy()
    tt1 = datetime.datetime.now()
    F, info = poselib.estimate_fundamental(instance["x1"], instance["x2"], opt)
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()

    err_R, err_t = fundamental_error(F, instance)
    return {'rot': err_R, 't': err_t, 'rt': runtime}

def fundamental_pycolmap(instance):
    opt = poselib_opt_to_pycolmap_opt(instance['opt'])
    tt1 = datetime.datetime.now()
    result = pycolmap.estimate_fundamental_matrix(
        instance["x1"], instance["x2"], opt
    )
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()
    if result is None or "F" not in result:
        return  {'rot': 180.0, 't': 180.0, 'rt': runtime}
    F = result["F"]

    err_R, err_t = fundamental_error(F, instance)
    return {'rot': err_R, 't': err_t, 'rt': runtime}



def homography_poselib(instance):
    opt = instance["opt"].copy()
    tt1 = datetime.datetime.now()
    H, info = poselib.estimate_homography(instance["x1"], instance["x2"], opt)
    tt2 = datetime.datetime.now()
    runtime = (tt2 - tt1).total_seconds()
    [err_r, err_t] = homography_error(H, instance)
    return {'rot': err_r, 't': err_t, 'rt': runtime}
    


def homography_pycolmap(instance):
    opt = poselib_opt_to_pycolmap_opt(instance['opt'])
    tt1 = datetime.datetime.now()
    result = pycolmap.estimate_homography_matrix(
        instance["x1"], instance["x2"], opt
    )
    tt2 = datetime.datetime.now()
    H = result["H"]
    runtime = (tt2 - tt1).total_seconds()
    [err_r, err_t] = homography_error(H, instance)
    return {'rot': err_r, 't': err_t, 'rt': runtime}
