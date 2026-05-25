import numpy as np
import cv2 as cv
import glob
import json
from pathlib import Path


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected_imgpoints, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv.norm(imgpoints[i], projected_imgpoints, cv.NORM_L2)
        n = len(projected_imgpoints)
        total_error += error * error
        total_points += n

    mean_error = np.sqrt(total_error / total_points) if total_points > 0 else 0.0
    return mean_error


def save_results(save_dir: Path, ret, mtx, dist, rvecs, tvecs, image_size, mean_error):
    npz_path = save_dir / "camera_calibration_result.npz"
    json_path = save_dir / "camera_calibration_result.json"

    np.savez(
        npz_path,
        ret=ret,
        mtx=mtx,
        dist=dist,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=mean_error,
    )

    result_dict = {
        "ret": float(ret),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "reprojection_error": float(mean_error),
        "num_views": len(rvecs),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {npz_path}")
    print(f"[SAVE] {json_path}")


def main():
    # checkerboard blocks count
    horizon = 7
    vertical = 10

    # inner corners count
    horizon_params = horizon - 1
    vertical_params = vertical - 1

    square_size = 0.025

    data_dir = Path("../data/checker_board")
    image_paths = sorted(glob.glob(str(data_dir / "*_color.png")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {data_dir.resolve()}")

    print(f"[INFO] Found {len(image_paths)} images")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((horizon_params * vertical_params, 3), np.float32)
    objp[:, :2] = np.mgrid[0:horizon_params, 0:vertical_params].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    image_size = None
    success_count = 0

    for fname in image_paths:
        img = cv.imread(fname)
        if img is None:
            print(f"[WARN] Could not read {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv.findChessboardCorners(
            gray,
            (horizon_params, vertical_params),
            None
        )

        if ret:
            objpoints.append(objp.copy())

            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners2)

            vis = img.copy()
            cv.drawChessboardCorners(
                vis, (horizon_params, vertical_params), corners2, ret
            )
            cv.imshow("corners", vis)
            cv.waitKey(200)

            success_count += 1
            print(f"[OK] {Path(fname).name}")
        else:
            print(f"[FAIL] {Path(fname).name}")

    cv.destroyAllWindows()

    if success_count < 3:
        raise RuntimeError(
            f"Too few valid checkerboard images: {success_count}. "
            f"Need at least 3, preferably 10~20+."
        )

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    mean_error = compute_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, mtx, dist
    )

    print("\n" + "=" * 60)
    print("[RESULT] Camera calibration finished")
    print(f"ret = {ret}")
    print(f"camera matrix (mtx) =\n{mtx}")
    print(f"dist coeffs =\n{dist}")
    print(f"num valid views = {len(rvecs)}")
    print(f"image size = {image_size}")
    print(f"mean reprojection error = {mean_error:.6f} px")
    print("=" * 60 + "\n")

    save_results(
        save_dir=data_dir,
        ret=ret,
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=image_size,
        mean_error=mean_error,
    )


if __name__ == "__main__":
    main()