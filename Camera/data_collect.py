import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import pylibfranka as pf


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial_number=None, enable_depth=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_number is not None:
            self.config.enable_device(str(serial_number))

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # auto exposure / auto white balance 안정화
        for _ in range(20):
            self.pipeline.wait_for_frames()

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()

        self.depth_intrinsics = None
        if enable_depth:
            depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_stream.get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()

        if self.enable_depth:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())

        depth_image = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def read_latest(self, repeat=3):

        color, depth = None, None
        for _ in range(max(1, repeat)):
            color, depth = self.read()
        return color, depth

    def get_intrinsics_dict(self):
        c = self.color_intrinsics
        data = {
            "color": {
                "width": c.width,
                "height": c.height,
                "fx": c.fx,
                "fy": c.fy,
                "cx": c.ppx,
                "cy": c.ppy,
                "model": str(c.model),
                "coeffs": list(c.coeffs),
            }
        }

        if self.depth_intrinsics is not None:
            d = self.depth_intrinsics
            data["depth"] = {
                "width": d.width,
                "height": d.height,
                "fx": d.fx,
                "fy": d.fy,
                "cx": d.ppx,
                "cy": d.ppy,
                "model": str(d.model),
                "coeffs": list(d.coeffs),
            }

        return data

    def release(self):
        self.pipeline.stop()


class FrankaRobotInterface:
    def __init__(self, robot_ip="172.16.0.2"):
        self.robot = pf.Robot(robot_ip)

    @staticmethod
    def parse_O_T_EE(O_T_EE):
        T = np.array(O_T_EE, dtype=np.float64).reshape(4, 4).T
        return T

    def read_state_once(self):
        return self.robot.read_once()

    def state_to_dict(self, state):
        q = [float(x) for x in state.q]
        dq = [float(x) for x in state.dq]

        T_ee = self.parse_O_T_EE(state.O_T_EE)
        position = T_ee[:3, 3].tolist()
        rotation = T_ee[:3, :3].tolist()

        robot_time_sec = None
        if hasattr(state, "time"):
            try:
                robot_time_sec = float(state.time.to_sec())
            except Exception:
                robot_time_sec = None

        return {
            "joint_positions": q,
            "joint_velocities": dq,
            "O_T_EE_raw": [float(x) for x in state.O_T_EE],
            "O_T_EE_matrix": T_ee.tolist(),
            "end_effector_position": position,
            "end_effector_rotation_matrix": rotation,
            "robot_time_sec": robot_time_sec,
        }

    def get_latest_state_dict(self, repeat=3):
        state = None
        for _ in range(max(1, repeat)):
            state = self.read_state_once()
        return self.state_to_dict(state)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_next_index(save_dir: Path):
    existing = sorted(save_dir.glob("sample_*_color.png"))
    if not existing:
        return 0

    max_idx = -1
    for f in existing:
        try:
            idx = int(f.stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except Exception:
            pass
    return max_idx + 1


def save_intrinsics_once(save_dir: Path, intrinsics: dict):
    intr_path = save_dir / "realsense_intrinsics.json"
    if not intr_path.exists():
        with open(intr_path, "w", encoding="utf-8") as f:
            json.dump(intrinsics, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] intrinsics: {intr_path.name}")


def save_sample(save_dir: Path, idx: int, color, depth, robot_state_dict, save_depth=True):
    timestamp = datetime.now().isoformat()

    color_name = f"sample_{idx:04d}_color.png"
    depth_name = f"sample_{idx:04d}_depth.npy"
    joint_name = f"sample_{idx:04d}_joints.json"

    color_path = save_dir / color_name
    depth_path = save_dir / depth_name
    joint_path = save_dir / joint_name
    log_path = save_dir / "capture_log.jsonl"

    cv.imwrite(str(color_path), color)

    if save_depth and depth is not None:
        np.save(depth_path, depth)

    meta = {
        "index": idx,
        "timestamp": timestamp,
        "color_file": color_name,
        "depth_file": depth_name if (save_depth and depth is not None) else None,
        "joint_file": joint_name,
        **robot_state_dict,
    }

    with open(joint_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return color_path, joint_path, meta


def delete_session_files(save_dir: Path):
    patterns = [
        "sample_*_color.png",
        "sample_*_depth.npy",
        "sample_*_joints.json",
        "capture_log.jsonl",
    ]

    deleted = []
    for pattern in patterns:
        for p in save_dir.glob(pattern):
            try:
                p.unlink()
                deleted.append(p.name)
            except Exception as e:
                print(f"[WARN] failed to delete {p.name}: {e}")

    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="../data/checker_board")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--robot-ip", type=str, default="172.16.0.2")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--camera-fresh-read", type=int, default=3)
    parser.add_argument("--robot-fresh-read", type=int, default=3)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    cam = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial_number=args.serial,
        enable_depth=not args.no_depth,
    )
    robot = FrankaRobotInterface(robot_ip=args.robot_ip)

    save_intrinsics_once(save_dir, cam.get_intrinsics_dict())
    next_idx = get_next_index(save_dir)

    print("=" * 70)
    print("[INFO] RealSense + Franka data collection started")
    print("[INFO] ENTER : save latest frame + latest robot state")
    print("[INFO] r     : reset saved samples in save-dir")
    print("[INFO] q     : quit")
    print(f"[INFO] save dir : {save_dir.resolve()}")
    print(f"[INFO] robot ip : {args.robot_ip}")
    print(f"[INFO] camera fresh read : {args.camera_fresh_read}")
    print(f"[INFO] robot fresh read  : {args.robot_fresh_read}")
    print("=" * 70)

    try:
        while True:
            color, depth = cam.read()
            if color is None:
                print("[WARN] Failed to read RealSense frame")
                continue

            cv.imshow("RealSense Color", color)

            if depth is not None:
                depth_vis = cv.convertScaleAbs(depth, alpha=0.03)
                cv.imshow("RealSense Depth", depth_vis)

            key = cv.waitKey(1) & 0xFF

            if key in (10, 13): # data collect
                fresh_color, fresh_depth = cam.read_latest(repeat=args.camera_fresh_read)
                robot_state_dict = robot.get_latest_state_dict(repeat=args.robot_fresh_read)

                if fresh_color is None:
                    fresh_color = color
                    fresh_depth = depth

                color_path, joint_path, meta = save_sample(
                    save_dir=save_dir,
                    idx=next_idx,
                    color=fresh_color,
                    depth=fresh_depth,
                    robot_state_dict=robot_state_dict,
                    save_depth=not args.no_depth,
                )

                print(f"[SAVE] {color_path.name}")
                print(f"[SAVE] {joint_path.name}")
                print(f"[SAVE] q = {meta['joint_positions']}")
                print(f"[SAVE] ee_position = {meta['end_effector_position']}")
                print("-" * 70)

                next_idx += 1

            elif key == ord("r"): # reset
                deleted = delete_session_files(save_dir)
                next_idx = 0
                print("[RESET] saved samples deleted.")
                print(f"[RESET] deleted {len(deleted)} files.")
                print("-" * 70)

            elif key == ord("q"): # quite
                print("[INFO] Quit")
                break

    finally:
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()