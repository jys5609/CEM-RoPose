import numpy as np
import cv2 as cv
import json
import glob
from pathlib import Path


def parse_matrices(data_dir, json_paths, intrinsics, dist_coeffs):

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    hor = 7 - 1
    ver = 10 - 1
    square_size = 0.025  # 25mm

    objp = np.zeros((hor * ver, 3), np.float32)
    objp[:, :2] = np.mgrid[0:hor, 0:ver].T.reshape(-1, 2)
    objp *= square_size

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    success_count = 0
    print(f"[INFO] Processing {len(json_paths)} samples...")

    for json_path in json_paths:
        json_path = Path(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        T_ee_base = np.array(meta['O_T_EE_matrix'], dtype=np.float64)

        img_filename = meta['color_file']
        img_path = data_dir / img_filename

        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path.name}")
            continue

        img = cv.imread(str(img_path))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (hor, ver), None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret_pnp, rvec, tvec = cv.solvePnP(objp, corners2, intrinsics, dist_coeffs)

            if ret_pnp:
                # Eye-to-Hand EE->Base inverse
                T_base_ee = np.linalg.inv(T_ee_base)

                R_gripper2base.append(T_base_ee[:3, :3])
                t_gripper2base.append(T_base_ee[:3, 3])

                R_board, _ = cv.Rodrigues(rvec)
                R_target2cam.append(R_board)
                t_target2cam.append(tvec)

                success_count += 1
            else:
                print(f"[FAIL] PnP failed for {img_path.name}")
        else:
            print(f"[FAIL] Chessboard not found in {img_path.name}")

    if success_count < 3:
        raise RuntimeError("Need at least 3 valid matching pairs for Hand-Eye calibration.")

    print(f"[SUCCESS] Prepared {success_count} matching pairs.")
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def compute_extrinsic(data_dir, json_paths, instr_path):
    # 1. intrinsic parameter
    with open(instr_path, 'r', encoding='utf-8') as f:
        calib_data = json.load(f)

    K = np.array(calib_data['camera_matrix'], dtype=np.float64)
    D = np.array(calib_data['dist_coeffs'], dtype=np.float64)
    print(f"[INFO] Loaded intrinsics from {instr_path.name}")

    R_grip2base, t_grip2base, R_target2cam, t_target2cam = parse_matrices(
        data_dir, json_paths, K, D
    )

    print("[INFO] Computing Hand-Eye Calibration (Eye-to-Hand)...")

    R_cam2base, t_cam2base = cv.calibrateHandEye(
        R_grip2base, t_grip2base,
        R_target2cam, t_target2cam,
        method=cv.CALIB_HAND_EYE_TSAI
    )

    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = R_cam2base
    T_base_cam[:3, 3] = t_cam2base.squeeze()

    print("\n" + "=" * 70)
    print("[RESULT] Transformation Matrix (Camera Frame in Robot Base Frame):")
    np.set_printoptions(suppress=True, precision=4)
    print(T_base_cam)
    print("=" * 70 + "\n")

    return T_base_cam


def print_mujoco_camera_tag(T_base_cam, name="real_view_cam", fovy="64.31"):
    R_base_cam_oc = T_base_cam[:3, :3]
    t_base_cam_oc = T_base_cam[:3, 3]

    X_mj = R_base_cam_oc[:, 0]
    Y_mj = -R_base_cam_oc[:, 1]

    mj_xyaxes = np.concatenate([X_mj, Y_mj])

    mj_pos_world = t_base_cam_oc + np.array([0, 0, 0.7], dtype=np.float64)

    print("!!! Copy and paste the following line into your scene.xml file !!!\n")
    tag_str = f'<camera name="{name}" pos="{mj_pos_world[0]:.4f} {mj_pos_world[1]:.4f} {mj_pos_world[2]:.4f}" ' \
              f'xyaxes="{mj_xyaxes[0]:.4f} {mj_xyaxes[1]:.4f} {mj_xyaxes[2]:.4f} {mj_xyaxes[3]:.4f} {mj_xyaxes[4]:.4f} {mj_xyaxes[5]:.4f}" ' \
              f'fovy="{fovy}"/>'
    print(tag_str)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data/checker_board"
    instr_path = data_dir / "camera_calibration_result.json"
    json_paths = sorted(glob.glob(str(data_dir / "sample_*_joints.json")))

    if not data_dir.exists() or not instr_path.exists() or len(json_paths) == 0:
        print(f"[ERROR] Data directories or required files are missing in {data_dir.resolve()}")
    else:
        T_base_cam = compute_extrinsic(data_dir, json_paths, instr_path)
        print_mujoco_camera_tag(T_base_cam)