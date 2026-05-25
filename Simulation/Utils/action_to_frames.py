import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np

from Simulation.Mujoco.mujoco_sim_rgb import SCENE_XML_PATH, MuJoCoRGBSim


BASE_DIR = Path(__file__).resolve()
BASE_DIR = BASE_DIR.parent.parent.parent
print(BASE_DIR)

SAVE_DIR = BASE_DIR / "Simulation/Utils/PredResult"
DEFAULT_ACTION_JSON = BASE_DIR / "Data/_Output/set2/result.json"
DEFAULT_OUTPUT_VIDEO = BASE_DIR / "Simulation/Utils/PredResult/result.mp4"
DEFAULT_TARGET_DURATION_SEC = 4.0

ORIGINAL_MATERIALS = {
    "black": [0.2, 0.2, 0.2, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "red": [1.0, 0.072272, 0.039546, 1.0],
    "gray": [0.863156, 0.863156, 0.863157, 1.0],
    "button_green": [0.102241, 0.571125, 0.102242, 1.0],
    "button_red": [0.520996, 0.008023, 0.013702, 1.0],
    "button_blue": [0.024157, 0.445201, 0.737911, 1.0],
    "matplane": [1.0, 1.0, 1.0, 1.0],
}

ORIGINAL_GEOMS = {
    "box_geom": [0.5, 0.5, 0.5, 1.0],
    "table_top": [0.8, 0.6, 0.4, 1.0],
    "table_leg1": [0.95, 0.95, 0.95, 1.0],
    "table_leg2": [0.95, 0.95, 0.95, 1.0],
    "table_leg3": [0.95, 0.95, 0.95, 1.0],
    "table_leg4": [0.95, 0.95, 0.95, 1.0],
}


def load_action_sequence(action_json_path):
    with open(action_json_path, "r") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    if not frames:
        raise ValueError(f"action sequence가 비어 있습니다: {action_json_path}")
    return payload, frames


def set_named_material(model, material_name, rgba, emission=None, specular=None, shininess=None, reflectance=None):
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
    if mat_id == -1:
        return

    model.mat_rgba[mat_id] = np.array(rgba, dtype=np.float32)
    if emission is not None:
        model.mat_emission[mat_id] = emission
    if specular is not None:
        model.mat_specular[mat_id] = specular
    if shininess is not None:
        model.mat_shininess[mat_id] = shininess
    if reflectance is not None:
        model.mat_reflectance[mat_id] = reflectance


def set_named_geom(model, geom_name, rgba):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id == -1:
        return
    model.geom_rgba[geom_id] = np.array(rgba, dtype=np.float32)


def restore_rgb_appearance(sim):
    model = sim.model

    set_named_material(model, "black", ORIGINAL_MATERIALS["black"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "white", ORIGINAL_MATERIALS["white"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "red", ORIGINAL_MATERIALS["red"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "gray", ORIGINAL_MATERIALS["gray"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "button_green", ORIGINAL_MATERIALS["button_green"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "button_red", ORIGINAL_MATERIALS["button_red"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "button_blue", ORIGINAL_MATERIALS["button_blue"], emission=0.0, specular=0.1, shininess=0.1, reflectance=0.0)
    set_named_material(model, "matplane", ORIGINAL_MATERIALS["matplane"], emission=0.0, specular=0.05, shininess=0.05, reflectance=0.3)

    for geom_name, rgba in ORIGINAL_GEOMS.items():
        set_named_geom(model, geom_name, rgba)

    if model.nlight > 0:
        model.light_diffuse[0] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        model.light_specular[0] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        model.light_ambient[0] = np.array([0.15, 0.15, 0.15], dtype=np.float32)

    try:
        model.vis.headlight.ambient[:] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        model.vis.headlight.diffuse[:] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        model.vis.headlight.specular[:] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        model.vis.rgba.haze[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    except AttributeError:
        pass


def interpolate_actions(start, end, steps):
    if steps <= 1:
        return [np.array(end, dtype=np.float64)]

    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    return [(1.0 - alpha) * start + alpha * end for alpha in alphas]


def record_action_video(
    action_json_path,
    output_video_path,
    scene_xml_path,
    calibration_json_path,
    width,
    height,
    fps,
    target_duration_sec,
    interpolation_steps,
    settle_steps,
):
    payload, frames = load_action_sequence(action_json_path)

    sim = MuJoCoRGBSim(
        model_path=scene_xml_path,
        calibration_path=calibration_json_path,
        width=width,
        height=height,
    )

    try:
        restore_rgb_appearance(sim)
        sim.reset()

        total_output_frames = 1 + max(0, len(frames) - 1) * max(1, interpolation_steps)
        resolved_fps = fps
        if target_duration_sec is not None and target_duration_sec > 0:
            resolved_fps = total_output_frames / target_duration_sec

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            resolved_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {output_video_path}")

        previous_action = np.array(frames[0]["joint_positions"], dtype=np.float64)
        sim.set_joint_positions(previous_action, settle_steps=settle_steps)
        first_frame_bgr = cv2.cvtColor(sim.render_rgb(), cv2.COLOR_RGB2BGR)
        
        
        cv2.imwrite(str(SAVE_DIR) + "/frame_000000.png", first_frame_bgr)

        print(f"[DEBUG] current_action : {previous_action}")

        # writer.write(first_frame_bgr)

        for frame_idx, frame in enumerate(frames[1:], start=1):
            current_action = np.array(frame["joint_positions"], dtype=np.float64)
            print(f"[DEBUG] current_action : {current_action}")

            # action = np.array(frames[0]["joint_positions"], dtype=np.float64)
            sim.set_joint_positions(current_action, settle_steps=settle_steps)
            frame_bgr = cv2.cvtColor(sim.render_rgb(), cv2.COLOR_RGB2BGR)
            save_path = str(SAVE_DIR) + "/frame_" +frame["binary_image_file"]

            cv2.imwrite(save_path, frame_bgr)
            

            action_path = interpolate_actions(previous_action, current_action, interpolation_steps)

            for action in action_path:
                sim.set_joint_positions(action, settle_steps=settle_steps)
                frame_bgr = cv2.cvtColor(sim.render_rgb(), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            previous_action = current_action

            print(
                f"[RENDER {frame_idx:04d}] {frame.get('image_file', frame_idx)} "
                f"score={frame.get('similarity_score', 0.0):.4f}"
            )

        # writer.release()
        print(f"[DONE] 비디오 저장: {output_video_path}")
        print(f"[INFO] source_json: {action_json_path}")
        print(f"[INFO] frame_count: {len(frames)}")
        print(f"[INFO] interpolation_steps: {interpolation_steps}")
        print(f"[INFO] output_frame_count: {total_output_frames}")
        print(f"[INFO] output_fps: {resolved_fps:.4f}")
        if target_duration_sec is not None and target_duration_sec > 0:
            print(f"[INFO] target_duration_sec: {target_duration_sec}")
        return payload
    finally:
        sim.close()


def parse_args():
    parser = argparse.ArgumentParser(description="예측된 joint action sequence를 RGB MuJoCo 영상으로 재생하여 비디오로 저장합니다.")
    parser.add_argument("--action-json", type=Path, default=DEFAULT_ACTION_JSON)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--scene-xml", type=Path, default=SCENE_XML_PATH)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--target-duration-sec", type=float, default=DEFAULT_TARGET_DURATION_SEC)
    parser.add_argument("--interpolation-steps", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    record_action_video(
        action_json_path=args.action_json,
        output_video_path=args.output_video,
        scene_xml_path=args.scene_xml,
        calibration_json_path=args.calibration_json,
        width=args.width,
        height=args.height,
        fps=args.fps,
        target_duration_sec=args.target_duration_sec,
        interpolation_steps=args.interpolation_steps,
        settle_steps=args.settle_steps,
    )


if __name__ == "__main__":
    main()
