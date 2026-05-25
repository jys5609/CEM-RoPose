import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MUJOCO_SIM_DIR = PROJECT_ROOT / "Simulation/Mujoco"

for import_path in (WORKSPACE_ROOT, MUJOCO_SIM_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from Simulation.Mujoco.mujoco_sim_rgb import SCENE_XML_PATH
from Simulation.Mujoco.mujoco_sim_robot_depth import MuJoCoRobotDepthSim

POLICY_DIR = PROJECT_ROOT / "Simulation/Utils/SytheticPolicy"
FRAMES_ROOT_DIR = PROJECT_ROOT / "Data/SimulationSyntheticFrames"

SET1_JSON_PATH = POLICY_DIR / "set1_closecamera.json"
SET2_JSON_PATH = POLICY_DIR / "set2_donwandup.json"

SET1_OUTPUT_ROOT = FRAMES_ROOT_DIR / "Set1_CloseCamera"
SET2_OUTPUT_ROOT = FRAMES_ROOT_DIR / "Set2_DownAndUp"

DEFAULT_FRAME_COUNT = 80
DEFAULT_CAMERA_NAME = "real_view_cam"
DEFAULT_SETTLE_STEPS = 1
DEFAULT_PROFILE_DYNAMIC_GAMMA = 0.78
DEFAULT_CLOSE_SEARCH_SAMPLES = 2400
DEFAULT_DOWN_SEARCH_SAMPLES = 4000
DEFAULT_SEARCH_SEED = 0
DEFAULT_TABLE_CLEARANCE_M = 0.025


def load_action_sequence(action_json_path):
    with open(action_json_path, "r") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"action sequence가 비어 있습니다: {action_json_path}")
    return payload, frames


def save_action_sequence(payload, output_json_path):
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(payload, f, indent=2)


def extract_reference_poses(action_json_path):
    payload, frames = load_action_sequence(action_json_path)
    start_qpos = np.array(frames[0]["joint_positions"], dtype=np.float64)
    peak_idx = max(
        range(len(frames)),
        key=lambda idx: sum(abs(x - y) for x, y in zip(frames[idx]["joint_positions"], frames[0]["joint_positions"])),
    )
    target_qpos = np.array(frames[peak_idx]["joint_positions"], dtype=np.float64)
    return payload, start_qpos, target_qpos


def map_reference_target_to_sim_home(sim_home_qpos, reference_start_qpos, reference_target_qpos, joint_bounds):
    sim_home_qpos = np.array(sim_home_qpos, dtype=np.float64)
    reference_start_qpos = np.array(reference_start_qpos, dtype=np.float64)
    reference_target_qpos = np.array(reference_target_qpos, dtype=np.float64)

    reference_delta = reference_target_qpos - reference_start_qpos
    mapped_target_qpos = sim_home_qpos + reference_delta
    return np.clip(mapped_target_qpos, joint_bounds[:, 0], joint_bounds[:, 1])


def get_camera_world_position(sim, camera_name):
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"카메라를 찾을 수 없습니다: {camera_name}")

    mujoco.mj_forward(sim.model, sim.data)
    if hasattr(sim.data, "cam_xpos"):
        return np.array(sim.data.cam_xpos[cam_id], dtype=np.float64)

    cam_body_id = int(sim.model.cam_bodyid[cam_id])
    cam_local_pos = np.array(sim.model.cam_pos[cam_id], dtype=np.float64)
    if cam_body_id >= 0:
        body_pos = np.array(sim.data.xpos[cam_body_id], dtype=np.float64)
        body_rot = np.array(sim.data.xmat[cam_body_id], dtype=np.float64).reshape(3, 3)
        return body_pos + body_rot @ cam_local_pos
    return cam_local_pos


def evaluate_pose(sim, joint_positions, camera_name):
    clipped = sim.set_joint_positions(joint_positions, settle_steps=0)
    ee_pose = sim.get_end_effector_pose()
    if ee_pose["position"] is None:
        raise RuntimeError("end effector site를 찾을 수 없습니다.")

    ee_position = np.array(ee_pose["position"], dtype=np.float64)
    camera_position = get_camera_world_position(sim, camera_name)
    camera_distance = float(np.linalg.norm(ee_position - camera_position))
    return {
        "joint_positions": clipped,
        "end_effector_position": ee_position,
        "camera_position": camera_position,
        "camera_distance": camera_distance,
    }


def get_table_top_surface_z(sim):
    geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if geom_id == -1:
        raise ValueError("table_top geom을 찾을 수 없습니다.")

    mujoco.mj_forward(sim.model, sim.data)
    geom_center_z = float(sim.data.geom_xpos[geom_id][2])
    geom_half_height = float(sim.model.geom_size[geom_id][2])
    return geom_center_z + geom_half_height


def min_jerk_profile(t):
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def dynamic_min_jerk_profile(t, gamma=DEFAULT_PROFILE_DYNAMIC_GAMMA):
    return np.power(min_jerk_profile(t), gamma)


def build_round_trip_profile(frame_count):
    if frame_count < 3:
        raise ValueError("frame_count는 최소 3 이상이어야 합니다.")

    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    profile = np.zeros_like(phase)
    first_half = phase <= 0.5
    profile[first_half] = dynamic_min_jerk_profile(phase[first_half] / 0.5)
    profile[~first_half] = dynamic_min_jerk_profile((1.0 - phase[~first_half]) / 0.5)
    return profile


def build_close_camera_offsets(phase, motion_span):
    motion_span = np.array(motion_span, dtype=np.float64)
    abs_span = np.abs(motion_span)

    style_scale = float(np.clip(np.linalg.norm(motion_span[:6]) / 0.35, 0.85, 1.35))
    shoulder_sweep = np.sin(2.0 * np.pi * phase) * np.power(np.sin(np.pi * phase), 0.9)
    wrist_twist = np.sin(4.0 * np.pi * phase) * np.power(np.sin(np.pi * phase), 1.75)

    sweep_delta = style_scale * np.array(
        [
            0.028 + 0.18 * abs_span[0],
            -0.018 - 0.08 * abs_span[1],
            0.032 + 0.14 * abs_span[2],
            -0.026 - 0.10 * abs_span[3],
            0.024 + 0.45 * max(abs_span[4], 0.01),
            0.030 + 0.12 * abs_span[5],
            0.0,
        ],
        dtype=np.float64,
    )
    twist_delta = style_scale * np.array(
        [
            -0.012,
            0.010,
            0.015,
            0.018,
            -0.035,
            0.024,
            0.0,
        ],
        dtype=np.float64,
    )

    return shoulder_sweep * sweep_delta + wrist_twist * twist_delta


def build_down_bow_offsets(phase, motion_span):
    motion_span = np.array(motion_span, dtype=np.float64)
    abs_span = np.abs(motion_span)

    bow_curve = np.power(np.sin(np.pi * phase), 1.2)
    bow_sway = np.sin(2.0 * np.pi * phase) * np.power(np.sin(np.pi * phase), 1.5)
    wrist_fold = np.sin(3.0 * np.pi * phase) * np.power(np.sin(np.pi * phase), 2.0)

    bow_delta = np.array(
        [
            0.016 + 0.12 * abs_span[0],
            0.030 + 0.10 * abs_span[1],
            -0.020 - 0.08 * abs_span[2],
            -0.028 - 0.08 * abs_span[3],
            0.040 + 0.14 * abs_span[4],
            0.028 + 0.08 * abs_span[5],
            0.0,
        ],
        dtype=np.float64,
    )
    sway_delta = np.array(
        [
            0.024,
            -0.018,
            0.020,
            -0.026,
            0.030,
            0.018,
            0.0,
        ],
        dtype=np.float64,
    )
    fold_delta = np.array(
        [
            -0.010,
            0.012,
            -0.014,
            0.024,
            0.026,
            -0.016,
            0.0,
        ],
        dtype=np.float64,
    )

    return bow_curve * bow_delta + bow_sway * sway_delta + wrist_fold * fold_delta


def build_joint_offsets(phase, motion_span, motion_style):
    if motion_style == "table_bow_down_and_up":
        return build_down_bow_offsets(phase, motion_span)
    return build_close_camera_offsets(phase, motion_span)


def build_action_payload(start_qpos, target_qpos, frame_count, joint_bounds, trajectory_type, motion_style):
    start_qpos = np.array(start_qpos, dtype=np.float64)
    target_qpos = np.array(target_qpos, dtype=np.float64)
    motion_span = target_qpos - start_qpos
    profile = build_round_trip_profile(frame_count)
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)

    frames = []
    for frame_index, (alpha, phase_value) in enumerate(zip(profile, phase)):
        base_joint_positions = (1.0 - alpha) * start_qpos + alpha * target_qpos
        expressive_offset = build_joint_offsets(phase_value, motion_span, motion_style)
        joint_positions = np.clip(base_joint_positions + expressive_offset, joint_bounds[:, 0], joint_bounds[:, 1])
        frames.append(
            {
                "frame_index": frame_index,
                "joint_positions": [float(x) for x in joint_positions],
            }
        )

    return {
        "trajectory_type": trajectory_type,
        "frame_count": int(frame_count),
        "joint_count": int(start_qpos.size),
        "frames": frames,
    }


def search_close_camera_target(sim, start_qpos, reference_target, camera_name, search_samples, search_seed):
    base_info = evaluate_pose(sim, start_qpos, camera_name)
    reference_info = evaluate_pose(sim, reference_target, camera_name)
    bounds = sim.get_joint_bounds()

    search_mean = reference_target + np.array([0.01, 0.05, 0.02, -0.02, 0.03, 0.03, 0.0], dtype=np.float64)
    search_scale = np.array([0.06, 0.08, 0.08, 0.08, 0.06, 0.08, 0.0], dtype=np.float64)
    scale_norm = np.maximum(search_scale, 0.01)
    rng = np.random.default_rng(search_seed)

    best = {
        "score": -np.inf,
        "info": reference_info,
    }

    candidate_list = [reference_target, np.clip(search_mean, bounds[:, 0], bounds[:, 1])]
    candidate_list.extend(
        np.clip(rng.normal(search_mean, search_scale), bounds[:, 0], bounds[:, 1]) for _ in range(search_samples)
    )

    for candidate in candidate_list:
        candidate_info = evaluate_pose(sim, candidate, camera_name)
        ee_position = candidate_info["end_effector_position"]
        score = (
            (base_info["camera_distance"] - candidate_info["camera_distance"])
            - 0.08 * np.linalg.norm((candidate_info["joint_positions"] - reference_target) / scale_norm)
            - 0.35 * abs(float(ee_position[1]) - float(reference_info["end_effector_position"][1]))
        )

        if score > best["score"]:
            best = {
                "score": score,
                "info": candidate_info,
            }

    return base_info, best["info"]


def search_table_bow_target(sim, start_qpos, reference_target, camera_name, table_clearance, search_samples, search_seed):
    base_info = evaluate_pose(sim, start_qpos, camera_name)
    reference_info = evaluate_pose(sim, reference_target, camera_name)
    bounds = sim.get_joint_bounds()
    table_top_z = get_table_top_surface_z(sim)

    target_hover_z = table_top_z + table_clearance + 0.015
    search_mean = reference_target + np.array([0.0, -0.12, 0.05, 0.08, -0.10, -0.14, 0.0], dtype=np.float64)
    search_scale = np.array([0.05, 0.10, 0.05, 0.06, 0.08, 0.10, 0.0], dtype=np.float64)
    scale_norm = np.maximum(search_scale, 0.01)
    rng = np.random.default_rng(search_seed)

    best = {
        "score": -np.inf,
        "info": reference_info,
    }

    candidate_list = [reference_target, np.clip(search_mean, bounds[:, 0], bounds[:, 1])]
    candidate_list.extend(
        np.clip(rng.normal(search_mean, search_scale), bounds[:, 0], bounds[:, 1]) for _ in range(search_samples)
    )

    for candidate in candidate_list:
        candidate_info = evaluate_pose(sim, candidate, camera_name)
        ee_position = candidate_info["end_effector_position"]
        ee_x = float(ee_position[0])
        ee_y = float(ee_position[1])
        ee_z = float(ee_position[2])
        clearance_delta = ee_z - (table_top_z + table_clearance)

        score = 0.0
        score -= abs(ee_z - target_hover_z) * 7.0
        score -= abs(ee_y) * 1.5
        score -= max(0.0, 0.45 - ee_x) * 3.0
        score -= max(0.0, ee_x - 0.82) * 2.5
        score -= max(0.0, -clearance_delta) * 18.0
        score += min(float(candidate_info["joint_positions"][1]), 1.0) * 0.15
        score += min(float(candidate_info["joint_positions"][4]), 1.0) * 0.15
        score += min(float(candidate_info["joint_positions"][5] - start_qpos[5]), 1.2) * 0.10
        score -= 0.15 * np.linalg.norm((candidate_info["joint_positions"] - reference_target) / scale_norm)

        if score > best["score"]:
            best = {
                "score": score,
                "info": candidate_info,
            }

    return base_info, best["info"], table_top_z


def enforce_table_clearance(sim, payload, start_qpos, camera_name, table_top_z, table_clearance):
    min_safe_height = table_top_z + table_clearance
    safe_frames = []

    for frame in payload["frames"]:
        original_qpos = np.array(frame["joint_positions"], dtype=np.float64)
        original_info = evaluate_pose(sim, original_qpos, camera_name)
        safe_qpos = original_qpos

        if float(original_info["end_effector_position"][2]) < min_safe_height:
            low = 0.0
            high = 1.0
            safe_qpos = np.array(start_qpos, dtype=np.float64)

            for _ in range(16):
                mid = 0.5 * (low + high)
                candidate_qpos = start_qpos + mid * (original_qpos - start_qpos)
                candidate_info = evaluate_pose(sim, candidate_qpos, camera_name)
                if float(candidate_info["end_effector_position"][2]) >= min_safe_height:
                    safe_qpos = candidate_info["joint_positions"]
                    low = mid
                else:
                    high = mid

        safe_frames.append(
            {
                "frame_index": frame["frame_index"],
                "joint_positions": [float(x) for x in safe_qpos],
            }
        )

    payload["frames"] = safe_frames
    return payload


def build_set1_payload(sim, output_json_path, frame_count, camera_name, search_samples, search_seed):
    _, reference_start_qpos, reference_target = extract_reference_poses(SET1_JSON_PATH)
    start_qpos = np.array(sim.home_qpos, dtype=np.float64)
    reference_target = map_reference_target_to_sim_home(
        sim_home_qpos=start_qpos,
        reference_start_qpos=reference_start_qpos,
        reference_target_qpos=reference_target,
        joint_bounds=sim.get_joint_bounds(),
    )
    sim.reset(start_qpos)

    base_info, target_info = search_close_camera_target(
        sim=sim,
        start_qpos=start_qpos,
        reference_target=reference_target,
        camera_name=camera_name,
        search_samples=search_samples,
        search_seed=search_seed,
    )
    payload = build_action_payload(
        start_qpos=start_qpos,
        target_qpos=target_info["joint_positions"],
        frame_count=frame_count,
        joint_bounds=sim.get_joint_bounds(),
        trajectory_type="smooth_camera_approach_retreat",
        motion_style="camera_approach",
    )
    save_action_sequence(payload, output_json_path)

    metadata = {
        "start_qpos": start_qpos.tolist(),
        "target_qpos": target_info["joint_positions"].tolist(),
        "start_distance": base_info["camera_distance"],
        "target_distance": target_info["camera_distance"],
        "distance_reduction": base_info["camera_distance"] - target_info["camera_distance"],
        "start_ee_position": base_info["end_effector_position"].tolist(),
        "target_ee_position": target_info["end_effector_position"].tolist(),
    }
    return payload, metadata


def build_set2_payload(sim, output_json_path, frame_count, camera_name, table_clearance, search_samples, search_seed):
    _, reference_start_qpos, reference_target = extract_reference_poses(SET2_JSON_PATH)
    start_qpos = np.array(sim.home_qpos, dtype=np.float64)
    reference_target = map_reference_target_to_sim_home(
        sim_home_qpos=start_qpos,
        reference_start_qpos=reference_start_qpos,
        reference_target_qpos=reference_target,
        joint_bounds=sim.get_joint_bounds(),
    )
    sim.reset(start_qpos)

    base_info, target_info, table_top_z = search_table_bow_target(
        sim=sim,
        start_qpos=start_qpos,
        reference_target=reference_target,
        camera_name=camera_name,
        table_clearance=table_clearance,
        search_samples=search_samples,
        search_seed=search_seed,
    )
    payload = build_action_payload(
        start_qpos=start_qpos,
        target_qpos=target_info["joint_positions"],
        frame_count=frame_count,
        joint_bounds=sim.get_joint_bounds(),
        trajectory_type="table_bow_down_and_up",
        motion_style="table_bow_down_and_up",
    )
    payload = enforce_table_clearance(
        sim=sim,
        payload=payload,
        start_qpos=start_qpos,
        camera_name=camera_name,
        table_top_z=table_top_z,
        table_clearance=table_clearance,
    )
    save_action_sequence(payload, output_json_path)

    frame_heights = []
    for frame in payload["frames"]:
        frame_info = evaluate_pose(sim, frame["joint_positions"], camera_name)
        frame_heights.append(float(frame_info["end_effector_position"][2]))

    metadata = {
        "start_qpos": start_qpos.tolist(),
        "target_qpos": target_info["joint_positions"].tolist(),
        "start_distance": base_info["camera_distance"],
        "target_distance": target_info["camera_distance"],
        "distance_reduction": base_info["camera_distance"] - target_info["camera_distance"],
        "start_ee_position": base_info["end_effector_position"].tolist(),
        "target_ee_position": target_info["end_effector_position"].tolist(),
        "table_top_z": table_top_z,
        "table_clearance": table_clearance,
        "minimum_frame_ee_height": min(frame_heights),
        "height_drop": float(base_info["end_effector_position"][2] - min(frame_heights)),
    }
    return payload, metadata


def ensure_output_dirs(root_dir):
    rgb_dir = root_dir / "RGB"
    binary_dir = root_dir / "Binary"
    depth_dir = root_dir / "Depth"
    depth_1_dir = root_dir / "Depth_1"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_1_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, binary_dir, depth_dir, depth_1_dir


def normalize_depth_to_single_channel(depth_map, robot_mask, near_percentile=5.0, far_percentile=95.0):
    depth_gray = np.zeros(depth_map.shape, dtype=np.uint8)
    valid_mask = np.asarray(robot_mask, dtype=bool) & np.isfinite(depth_map) & (depth_map > 0.0)
    if not np.any(valid_mask):
        return depth_gray

    valid_depth = depth_map[valid_mask]
    near_value = float(np.percentile(valid_depth, near_percentile))
    far_value = float(np.percentile(valid_depth, far_percentile))

    if far_value - near_value < 1e-8:
        depth_gray[valid_mask] = 255
        return depth_gray

    normalized = (depth_map - near_value) / (far_value - near_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    depth_gray[valid_mask] = ((1.0 - normalized[valid_mask]) * 255.0).astype(np.uint8)
    return depth_gray


def render_frames_from_payload(sim, payload, output_root, camera_name, settle_steps):
    rgb_dir, binary_dir, depth_dir, depth_1_dir = ensure_output_dirs(output_root)

    for frame_idx, frame in enumerate(payload["frames"]):
        joint_positions = np.array(frame["joint_positions"], dtype=np.float64)
        sim.set_joint_positions(joint_positions, settle_steps=settle_steps)

        rgb = sim.render_rgb(camera_name=camera_name)
        depth_map, robot_mask = sim.compute_robot_camera_distance(camera_name=camera_name)
        robot_mask = np.asarray(robot_mask, dtype=bool)
        binary = np.zeros(robot_mask.shape, dtype=np.uint8)
        binary[robot_mask] = 255
        depth_rgb = sim.depth_to_color(depth_map, robot_mask)

        print(np.max(depth_map), np.min(depth_map))

        stem = f"{frame.get('frame_index', frame_idx):06d}"
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(binary_dir / f"{stem}.png"), binary)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
        np.save(str(depth_1_dir / f"{stem}.npy"), depth_map.astype(np.float32))        

        if frame_idx % 10 == 0 or frame_idx == len(payload["frames"]) - 1:
            print(f"[RENDER] {output_root.name} {frame_idx + 1}/{len(payload['frames'])} -> {stem}.png")


def print_metadata(job_name, json_path, output_root, metadata):
    print(f"[DONE] {job_name} action json: {json_path}")
    print(f"[INFO] {job_name} start distance : {metadata['start_distance']:.4f}")
    print(f"[INFO] {job_name} target distance: {metadata['target_distance']:.4f}")
    print(f"[INFO] {job_name} reduction      : {metadata['distance_reduction']:.4f}")
    if "table_top_z" in metadata:
        print(f"[INFO] {job_name} table top z   : {metadata['table_top_z']:.4f}")
        print(f"[INFO] {job_name} min ee height : {metadata['minimum_frame_ee_height']:.4f}")
        print(f"[INFO] {job_name} height drop   : {metadata['height_drop']:.4f}")
        print(f"[INFO] {job_name} clearance     : {metadata['table_clearance']:.4f}")
    print(f"[DONE] {job_name} frames root: {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MuJoCoRobotDepthSim 기반으로 Set1/Set2 action json과 RGB/Binary/Depth 프레임을 다시 생성합니다."
    )
    parser.add_argument("--set", choices=("all", "set1", "set2"), default="all")
    parser.add_argument("--scene-xml", type=Path, default=SCENE_XML_PATH)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--close-search-samples", type=int, default=DEFAULT_CLOSE_SEARCH_SAMPLES)
    parser.add_argument("--down-search-samples", type=int, default=DEFAULT_DOWN_SEARCH_SAMPLES)
    parser.add_argument("--search-seed", type=int, default=DEFAULT_SEARCH_SEED)
    parser.add_argument("--table-clearance", type=float, default=DEFAULT_TABLE_CLEARANCE_M)
    return parser.parse_args()


def main():
    args = parse_args()

    sim = MuJoCoRobotDepthSim(
        model_path=args.scene_xml,
        calibration_path=args.calibration_json,
        width=args.width,
        height=args.height,
    )

    try:
        if args.set in ("all", "set1"):
            set1_payload, set1_metadata = build_set1_payload(
                sim=sim,
                output_json_path=SET1_JSON_PATH,
                frame_count=args.frame_count,
                camera_name=args.camera_name,
                search_samples=args.close_search_samples,
                search_seed=args.search_seed,
            )
            render_frames_from_payload(
                sim=sim,
                payload=set1_payload,
                output_root=SET1_OUTPUT_ROOT,
                camera_name=args.camera_name,
                settle_steps=args.settle_steps,
            )
            print_metadata("Set1", SET1_JSON_PATH, SET1_OUTPUT_ROOT, set1_metadata)

        if args.set in ("all", "set2"):
            set2_payload, set2_metadata = build_set2_payload(
                sim=sim,
                output_json_path=SET2_JSON_PATH,
                frame_count=args.frame_count,
                camera_name=args.camera_name,
                table_clearance=args.table_clearance,
                search_samples=args.down_search_samples,
                search_seed=args.search_seed + 17,
            )
            render_frames_from_payload(
                sim=sim,
                payload=set2_payload,
                output_root=SET2_OUTPUT_ROOT,
                camera_name=args.camera_name,
                settle_steps=args.settle_steps,
            )
            print_metadata("Set2", SET2_JSON_PATH, SET2_OUTPUT_ROOT, set2_metadata)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
