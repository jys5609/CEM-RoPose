import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import math

from Simulation.Mujoco.mujoco_sim_rgb import DEFAULT_HOME_QPOS
from Simulation.Mujoco.mujoco_sim import MuJoCoSim
from Simulation.Mujoco.mujoco_sim_robot_depth import MuJoCoRobotDepthSim

BASE_DIR = Path(__file__).resolve()
BASE_DIR = BASE_DIR.parent.parent.parent
print(BASE_DIR)
DREAM_ROOT = BASE_DIR / "Data/Dream"
OUTPUT_ROOT = BASE_DIR / "Data/Output"
GEMMA_INIT_ROOT = BASE_DIR / "Data/GemmaInit"

BENCHMARK_DIR_MAP = {
    "azure": "Azure",
    "realsense": "RealSense",
    "kinect": "Kinect",
    "kinect360": "Kinect",
}

SCENE_XML_BY_BENCHMARK = {
    "realsense": GEMMA_INIT_ROOT / "RealSense/InitPose/scene.xml",
    "azure": GEMMA_INIT_ROOT / "Azure/InitPose/scene.xml",
    "kinect": GEMMA_INIT_ROOT / "Kinect360/InitPose/scene.xml",
    "kinect360": GEMMA_INIT_ROOT / "Kinect360/InitPose/scene.xml",
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".npy")
DEPTH_EXCLUDED_VALUES = np.array([0], dtype=np.uint8)


def normalize_benchmark_name(name):
    if name is None:
        return None

    normalized = str(name).strip().lower()
    if normalized in {"kinect", "kinect360"}:
        return "kinect"
    if normalized == "azure":
        return "azure"
    if normalized == "realsense":
        return "realsense"
    return normalized


def benchmark_name_from_input_root(input_root):
    input_root = Path(input_root)
    return normalize_benchmark_name(input_root.name)


def resolve_scene_xml_path(scene_xml_path, benchmark):
    if scene_xml_path is not None:
        return Path(scene_xml_path)

    normalized_benchmark = normalize_benchmark_name(benchmark)
    resolved = SCENE_XML_BY_BENCHMARK.get(normalized_benchmark)
    if resolved is None:
        raise ValueError(f"지원하지 않는 benchmark입니다: {benchmark}")
    return resolved


def load_binary_image(image_path, width, height):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    if image.shape != (height, width):
        print(f"IMAGE SIZE MISMATCHING! img_path :{image_path}, width :{width}, height : {height}")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    return np.where(image > 127, 255, 0).astype(np.uint8)


def collect_image_paths(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"디렉터리를 찾을 수 없습니다: {directory}")

    paths_by_stem = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in paths_by_stem:
            raise ValueError(f"같은 stem을 가진 파일이 중복되었습니다: {path.stem}")
        paths_by_stem[path.stem] = path

    if not paths_by_stem:
        raise FileNotFoundError(f"이미지 파일이 없습니다: {directory}")

    return paths_by_stem


def build_frame_pairs(binary_dir):
    binary_paths = collect_image_paths(binary_dir)

    return [
        {
            "stem": stem,
            "binary_path": binary_paths[stem],
        }
        for stem in sorted(binary_paths)
    ]


def compute_binary_score(gt_mask, pred_mask, width, height):
    
    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0

    # Converted into (1,-1) matrix. 1 : seg, -1 : !seg
    gt_bool_converted = np.where(gt_bool, 1, -1)
    pred_bool_converted = np.where(pred_bool, 1, -1)

    gt_bool_converted = gt_bool_converted.flatten()
    pred_bool_converted = pred_bool_converted.flatten()

    score = np.dot(gt_bool_converted, pred_bool_converted)

    return {
        "score": float(score)
    }


def combine_score(binary_loss, depth_loss, binary_weight, depth_weight):
    
    score = binary_weight * binary_loss + depth_weight * depth_loss

    return float(score)


_WORKER_SIM = None
_WORKER_WIDTH = None
_WORKER_HEIGHT = None
_WORKER_SETTLE_STEPS = None
_WORKER_BINARY_WEIGHT = None
_WORKER_DEPTH_WEIGHT = None


def _init_mujoco_worker(
    scene_xml,
    calibration_json,
    best_camera_setting_json,
    width,
    height,
    settle_steps,
    binary_weight,
    depth_weight,
    fixed_joint_positions,
    fixed_camera_parameter,
    robot_pose_parameter
):
    global _WORKER_SIM
    global _WORKER_WIDTH
    global _WORKER_HEIGHT
    global _WORKER_SETTLE_STEPS
    global _WORKER_BINARY_WEIGHT
    global _WORKER_DEPTH_WEIGHT

    os.environ.setdefault("MUJOCO_GL", "egl")

    _WORKER_SIM = MuJoCoRobotDepthSim(
        model_path=Path(scene_xml),
        calibration_path=None if calibration_json is None else Path(calibration_json),
        width=int(width),
        height=int(height),
    )
    if best_camera_setting_json is not None:
        _WORKER_SIM.apply_best_camera_setting_json(best_camera_setting_json)

    _WORKER_WIDTH = int(width)
    _WORKER_HEIGHT = int(height)
    _WORKER_SETTLE_STEPS = int(settle_steps)
    _WORKER_BINARY_WEIGHT = float(binary_weight)
    _WORKER_DEPTH_WEIGHT = float(depth_weight)

    if fixed_joint_positions is not None:
        _WORKER_SIM.set_joint_positions(
            np.asarray(fixed_joint_positions, dtype=np.float64),
            settle_steps=_WORKER_SETTLE_STEPS,
        )

    if fixed_camera_parameter is not None:
        fixed_camera_parameter = np.asarray(fixed_camera_parameter, dtype=np.float64)
        _WORKER_SIM.set_camera_positions(
            fixed_camera_parameter[:3],
            fixed_camera_parameter[3:],
            settle_steps=_WORKER_SETTLE_STEPS,
        )
    
    if robot_pose_parameter is not None:
        robot_pose_parameter = np.asarray(robot_pose_parameter, dtype=np.float64)
        _WORKER_SIM.set_robot_base_pose(
            robot_pose_parameter[:3],
            robot_pose_parameter[3:],
            settle_steps=_WORKER_SETTLE_STEPS,
        )


def _worker_mask_to_result(pred_mask, parameter, result_key, target_mask):
    binary_metrics = compute_binary_score(
        target_mask,
        pred_mask,
        _WORKER_WIDTH,
        _WORKER_HEIGHT,
    )
    reward = combine_score(
        binary_loss=binary_metrics["score"],
        depth_loss=0,
        binary_weight=_WORKER_BINARY_WEIGHT,
        depth_weight=_WORKER_DEPTH_WEIGHT,
    )

    return {
        result_key: np.asarray(parameter, dtype=np.float64).tolist(),
        "score": float(reward),
        "binary_reward": float(binary_metrics["score"]),
        "depth_loss": 0,
    }

def _worker_evaluate_joint_one(joint_positions, target_mask):
    clipped = _WORKER_SIM.set_joint_positions(
        joint_positions,
        settle_steps=_WORKER_SETTLE_STEPS,
    )
    robot_mask = _WORKER_SIM.binary_segmentation(camera_name="real_view_cam")
    robot_mask = robot_mask.astype(bool)
    pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)

    return _worker_mask_to_result(
        pred_mask=pred_mask,
        parameter=clipped,
        result_key="joint_positions",
        target_mask=target_mask,
    )

def _worker_evaluate_pose_with_joint_one(sample, target_mask):
    sample = np.asarray(sample, dtype=np.float64)

    joint_dim = _WORKER_SIM.arm_joint_count
    pose_dim = 6

    if sample.size != joint_dim + pose_dim:
        raise ValueError(
            f"expected {joint_dim + pose_dim} dims "
            f"({joint_dim} joint + {pose_dim} pose), got {sample.size}"
        )

    joint_positions = sample[:joint_dim]
    robot_pose = sample[joint_dim:joint_dim + pose_dim]

    clipped_joints = _WORKER_SIM.set_joint_positions(
        joint_positions,
        settle_steps=_WORKER_SETTLE_STEPS,
    )

    _WORKER_SIM.set_robot_base_pose(
        robot_pose[:3],
        robot_pose[3:],
        settle_steps=_WORKER_SETTLE_STEPS,
    )

    robot_mask = _WORKER_SIM.binary_segmentation(camera_name="real_view_cam")
    robot_mask = robot_mask.astype(bool)
    pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)

    binary_metrics = compute_binary_score(
        target_mask,
        pred_mask,
        _WORKER_WIDTH,
        _WORKER_HEIGHT,
    )
    reward = combine_score(
        binary_loss=binary_metrics["score"],
        depth_loss=0,
        binary_weight=_WORKER_BINARY_WEIGHT,
        depth_weight=_WORKER_DEPTH_WEIGHT,
    )

    return {
        "joint_positions": np.asarray(clipped_joints, dtype=np.float64).tolist(),
        "robot_pose": robot_pose.tolist(),
        "score": float(reward),
        "binary_reward": float(binary_metrics["score"]),
        "depth_loss": 0,
    }

def _worker_evaluate_joint_batch(batch_args):
    samples_chunk, target_mask = batch_args
    results = []
    for joint_positions in samples_chunk:
        results.append(_worker_evaluate_joint_one(joint_positions, target_mask))
    return results

def _worker_evaluate_pose_with_joint_batch(batch_args):
    samples_chunk, target_mask = batch_args
    results = []
    for sample in samples_chunk:
        results.append(_worker_evaluate_pose_with_joint_one(sample, target_mask))
    return results

class CEMActionPredictor:
    def __init__(
        self,
        sim,
        width = 640,
        height = 480,
        population_size=500,
        elite_fraction=0.10,
        smoothing=0.1,
        settle_steps=1,
        local_refine_steps=1,
        random_seed=0,
        binary_weight=1.0,
        depth_weight=0.0,
        start_frame=0,
        max_frames=None,
        log_interval=1,
        quiet=False,
        test_mode = 1,
        num_workers=1,
        seg_model="ground-sam2",
        benchmark="realsense",
        cem_mode="pose",
        input_root=None,
        dream_root=None,
        output_root=None,
        scene_xml_path=None,
        calibration_json_path=None,
        save_result_path=None
    ):
        self.sim = sim

        self.width = int(width)
        self.height = int(height)
        self.population_size = int(population_size)
        self.elite_fraction = float(elite_fraction)
        self.smoothing = float(smoothing)
        self.settle_steps = int(settle_steps)
        self.local_refine_steps = int(local_refine_steps)
        self.binary_weight = float(binary_weight)
        self.depth_weight = float(depth_weight)
        self.start_frame = int(start_frame)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.log_interval = int(log_interval)
        self.quiet = bool(quiet)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.attempt_seed_offset = 0
        self.frame_num = 0
        self.num_workers = max(1, int(num_workers))
        self._shared_executor = None
        self._shared_worker_count = None

        if scene_xml_path is None:
            self.scene_xml_path = Path(self.sim.model_path)
        else:
            self.scene_xml_path = Path(scene_xml_path)

        if calibration_json_path is None:
            self.calibration_json_path = None
        else:
            self.calibration_json_path = Path(calibration_json_path)
        self.save_result_path=save_result_path

        self.dream_root = Path(dream_root) if dream_root is not None else DREAM_ROOT
        self.input_root = Path(input_root) if input_root is not None else None
        self.benchmark = (
            benchmark_name_from_input_root(self.input_root)
            if self.input_root is not None
            else normalize_benchmark_name(benchmark)
        )
        self.output_root = Path(output_root) if output_root is not None else OUTPUT_ROOT
        self.current_task = None
        self.current_task_camera_setting = None
        self.current_best_camera_setting_json_path = None
        self.current_frame_pair = None

        base_position = self.sim.get_robot_base_position()
        self.base_coordination = None if base_position is None else base_position.tolist()

        self.input_binary = None


        self.output_json_path = self.output_root / "result.json"

        self.cem_output_frames = self.output_root / "Frames"
        self.cem_output_init = self.output_root / "Init"
        self.cem_output_debug = self.output_root / "Debug"

        self.note_path = self.output_root / "note.txt"

        (self.cem_output_frames).mkdir(parents=True, exist_ok=True)
        (self.cem_output_init).mkdir(parents=True, exist_ok=True)
        (self.cem_output_debug).mkdir(parents=True, exist_ok=True)

        # For Cold Starter
        self.is_cold_start = True
        self.is_middle_start = True

        self.cold_std=1
        
        self.cold_joint_std = 0.5
        self.cold_pose_std = 1

        self.cold_min_std=0.2

        self.cold_first_iteration = 30
        self.cold_second_iteration = 0

        self.cold_iteration = self.cold_first_iteration + self.cold_second_iteration


        self.middle_joint_std=0.05
        self.middle_pose_std = 0.001

        self.middle_min_std=0.01
        self.middle_iteration = 50
        # dvide middle and unocld

        # todo, reduce std to 0.001 or 0.01
        self.uncold_joint_std=0.01
        self.uncold_pose_std = 0.001

        self.uncold_min_std=0.001
        self.uncold_iteration = 15

        self.cem_mode=cem_mode
                
        with open(self.note_path, "w") as f:
            f.write(
                f"test_name : {benchmark}-{seg_model}\n"
                f"note_path : {self.note_path}\n"
                f"population_size : {self.population_size}\n"
                f"elite_fraction : {self.elite_fraction}\n"
                f"smoothing : {self.smoothing}\n"
                f"num_workers : {self.num_workers}\n\n"

                f"binary_weight : {self.binary_weight}\n"
                f"depth_weight : {self.depth_weight}\n\n"
            )

        if self.binary_weight < 0.0 or self.depth_weight < 0.0:
            raise ValueError("binary_weight와 depth_weight는 음수일 수 없습니다.")
        if self.start_frame < 0:
            raise ValueError("start_frame은 0 이상이어야 합니다.")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames는 1 이상이어야 합니다.")
        if self.log_interval <= 0:
            raise ValueError("log_interval은 1 이상이어야 합니다.")

        self.joint_bounds = self.sim.get_joint_bounds()
        self.joint_lower = self.joint_bounds[:, 0]
        self.joint_upper = self.joint_bounds[:, 1]
        self.joint_range = self.joint_upper - self.joint_lower

        self.current_joint_positions = np.asarray(
            self.sim.data.qpos[: self.joint_lower.size],
            dtype=np.float64,
        ).copy()

        self.current_camera_parameter = None
        self.elite_count = max(1, int(round(self.population_size * self.elite_fraction))) # number of elite

        self.pose_lower = np.array([-0.02, -0.02, -0.02, -0.0035, -0.0035, -np.pi], dtype=np.float64)
        self.pose_upper = np.array([ 0.02,  0.02, 0.02,  0.0035,  0.0035,  np.pi], dtype=np.float64)

        self.pose_range = self.pose_upper - self.pose_lower


        self.video_bench_tasks = self._discover_video_bench_tasks()
        self.frame_pairs = []

        if self.cem_mode != "pose":
            if self.input_root is None:
                raise ValueError("cem_mode가 'pose'가 아닐 때는 --input-root가 필요합니다.")
            self.input_binary = self.input_root
            frame_pairs = build_frame_pairs(
                binary_dir=self.input_binary,
            )

            self.frame_pairs = frame_pairs[self.start_frame :]
            if self.max_frames is not None:
                self.frame_pairs = self.frame_pairs[: self.max_frames]


    def log(self, message):
        if not self.quiet:
            print(message, flush=True)

    def _get_dream_benchmark_root(self):
        if self.input_root is not None:
            benchmark_root = self.input_root
        else:
            benchmark_dir_name = BENCHMARK_DIR_MAP.get(self.benchmark, self.benchmark)
            benchmark_root = self.dream_root / benchmark_dir_name
        if not benchmark_root.exists():
            raise FileNotFoundError(f"Dream benchmark 디렉터리를 찾을 수 없습니다: {benchmark_root}")
        return benchmark_root

    def _discover_video_bench_tasks(self):
        benchmark_root = self._get_dream_benchmark_root()
        return [
            {
                "task_name": benchmark_root.name.lower(),
                "sensor": benchmark_root.name,
                "version": "sequence",
                "task_root": benchmark_root,
            }
        ]

    def _build_dream_frame_pairs(self, task_root):
        frame_pairs = []
        for frame_dir in sorted(task_root.iterdir()):
            if not frame_dir.is_dir():
                continue

            camera_json_path = frame_dir / "best_camera_setting.json"
            if not camera_json_path.exists():
                raise FileNotFoundError(f"camera json을 찾을 수 없습니다: {camera_json_path}")

            binary_candidates = [
                path
                for path in sorted(frame_dir.iterdir())
                if path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and path.name != "best_camera_setting.json"
            ]
            if len(binary_candidates) != 1:
                raise ValueError(
                    f"{frame_dir} 에서 binary segmentation 파일은 1개여야 합니다. "
                    f"현재 {len(binary_candidates)}개를 찾았습니다."
                )

            binary_path = binary_candidates[0]
            frame_pairs.append(
                {
                    "frame_id": frame_dir.name,
                    "stem": binary_path.stem,
                    "frame_dir": frame_dir,
                    "binary_path": binary_path,
                    "camera_json_path": camera_json_path,
                }
            )

        if not frame_pairs:
            raise FileNotFoundError(f"Dream frame 폴더를 찾을 수 없습니다: {task_root}")

        return frame_pairs

    def _write_task_note_header(self, task):
        with open(self.note_path, "w") as f:
            f.write(
                f"task_name : {task['task_name']}\n"
                f"sensor : {task['sensor']}\n"
                f"version : {task['version']}\n"
                f"task_root : {task['task_root']}\n"
                f"note_path : {self.note_path}\n"
                f"population_size : {self.population_size}\n"
                f"elite_fraction : {self.elite_fraction}\n"
                f"smoothing : {self.smoothing}\n"
                f"num_workers : {self.num_workers}\n\n"
                f"binary_weight : {self.binary_weight}\n"
                f"depth_weight : {self.depth_weight}\n\n"
            )

    def _configure_pose_task(self, task):
        self.current_task = task
        self.current_task_camera_setting = None
        self.current_best_camera_setting_json_path = None
        self.current_frame_pair = None
        self.input_binary = task["task_root"]

        task_output_root = self.output_root / "Dream" / task["sensor"]
        self.output_json_path = task_output_root / "result.json"
        self.cem_output_frames = task_output_root / "Frames"
        self.cem_output_init = task_output_root / "Init"
        self.cem_output_debug = task_output_root / "Debug"
        self.note_path = task_output_root / "note.txt"

        self.cem_output_frames.mkdir(parents=True, exist_ok=True)
        self.cem_output_init.mkdir(parents=True, exist_ok=True)
        self.cem_output_debug.mkdir(parents=True, exist_ok=True)
        self._write_task_note_header(task)

        frame_pairs = self._build_dream_frame_pairs(task["task_root"])
        self.frame_pairs = frame_pairs[self.start_frame :]
        if self.max_frames is not None:
            self.frame_pairs = self.frame_pairs[: self.max_frames]

        self.frame_num = 0
        self.is_cold_start = True
        self.is_middle_start = False

    def _apply_pose_task_camera(self, frame_pair):
        self.current_frame_pair = frame_pair
        self.current_best_camera_setting_json_path = frame_pair["camera_json_path"]
        camera_setting = self.sim.apply_best_camera_setting_json(frame_pair["camera_json_path"])
        self.current_task_camera_setting = camera_setting

        if camera_setting["width"] != self.sim.width or camera_setting["height"] != self.sim.height:
            self.log(
                f"[TASK {self.current_task['task_name']}] "
                f"[FRAME {frame_pair['frame_id']}] JSON image size "
                f"{camera_setting['width']}x{camera_setting['height']} != sim size "
                f"{self.sim.width}x{self.sim.height}. Pose/fovy only applied."
            )

        self.log(
            f"[TASK {self.current_task['task_name']}] "
            f"[FRAME {frame_pair['frame_id']}] camera applied "
            f"{camera_setting['json_path']} "
            f"(fovy={camera_setting['fovy']:.4f})"
        )
        return camera_setting

    def set_sim_joint_positions(self, joint_positions):
        clipped = self.sim.set_joint_positions(
            joint_positions,
            settle_steps=self.settle_steps,
        )
        self.current_joint_positions = np.asarray(clipped, dtype=np.float64).copy()
        return clipped


    def render_observation(self, joint_positions):
        clipped = self.set_sim_joint_positions(joint_positions)
        robot_mask = self.sim.binary_segmentation(camera_name="real_view_cam")

        robot_mask = robot_mask.astype(bool)

        pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)
        
        return clipped, pred_mask
    
    def render_pose(self, joint_positions, robot_pose):
        clipped = self.set_sim_joint_positions(joint_positions)
        clipped = self.sim.set_robot_base_pose(robot_pose[:3], robot_pose[3:])
        robot_mask = self.sim.binary_segmentation(camera_name="real_view_cam")

        robot_mask = robot_mask.astype(bool)

        pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)
        
        return clipped, pred_mask

    def evaluate_action(self, target_mask, joint_positions):
        # [TODO] Check render_observation
        clipped, pred_mask = self.render_observation(joint_positions)

        save_sim_bin = self.cem_output_debug / f"{self.frame_num:06d}.png"

        # cv2.imwrite(save_sim_bin, pred_mask)
        
        # [TODO] Check binary similarity
        binary_metrics = compute_binary_score(target_mask, pred_mask, self.width, self.height)

        reward = combine_score(
            binary_loss=binary_metrics["score"],
            depth_loss=0,
            binary_weight=self.binary_weight,
            depth_weight=self.depth_weight,
        )

        return {
            "joint_positions": clipped.tolist(),
            "score": reward,
            "binary_reward" : binary_metrics["score"],
            "depth_loss" : 0
        }

    def evaluate_pose_with_joint(self, target_mask, sample):
        sample = np.asarray(sample, dtype=np.float64)

        joint_dim = self.sim.arm_joint_count
        pose_dim = 6

        if sample.size != joint_dim + pose_dim:
            raise ValueError(
                f"expected {joint_dim + pose_dim} dims "
                f"({joint_dim} joint + {pose_dim} pose), got {sample.size}"
            )

        joint_positions = sample[:joint_dim]
        robot_pose = sample[joint_dim:joint_dim + pose_dim]

        clipped_joints = self.set_sim_joint_positions(joint_positions)
        self.sim.set_robot_base_pose(robot_pose[:3], robot_pose[3:], settle_steps=self.settle_steps)
        robot_mask = self.sim.binary_segmentation(camera_name="real_view_cam")
        robot_mask = robot_mask.astype(bool)
        pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)

        binary_metrics = compute_binary_score(
            target_mask,
            pred_mask,
            self.width,
            self.height,
        )
        reward = combine_score(
            binary_loss=binary_metrics["score"],
            depth_loss=0,
            binary_weight=self.binary_weight,
            depth_weight=self.depth_weight,
        )

        return {
            "joint_positions": np.asarray(clipped_joints, dtype=np.float64).tolist(),
            "robot_pose": np.asarray(robot_pose, dtype=np.float64).tolist(),
            "score": float(reward),
            "binary_reward": float(binary_metrics["score"]),
            "depth_loss": 0,
        }

    def sample_normalized_population(self, mean, std, lower=None, upper=None):
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        z = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.population_size, mean.size),
        )
        samples = mean + std * z

        if lower is not None or upper is not None:
            samples = np.clip(samples, lower, upper)

        return samples
        
    
    def sample_population(self, mean, std, lower=None, upper=None):
        
        if self.is_cold_start==True:
            size = 1000
            self.population_size = 1000
            self.elite_count = max(1, int(round(self.population_size * self.elite_fraction))) # number of elite
        else:
            size = 1000
            self.population_size = 1000
            self.elite_count = max(1, int(round(self.population_size * self.elite_fraction))) # number of elite

        samples = self.rng.normal(
            loc = mean,
            scale = std,
            size=(size, mean.size)
        )

        if lower is not None or upper is not None:
            samples = np.clip(samples, lower, upper)

        return samples

    def _make_executor(
        self,
        target_mask=None,
        fixed_joint_positions=None,
        fixed_camera_parameter=None,
        robot_pose_parameter=None
    ):
        if self.num_workers <= 1:
            return None, 1

        worker_count = min(self.num_workers, self.population_size)

        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_init_mujoco_worker,
            initargs=(
                str(self.scene_xml_path),
                None if self.calibration_json_path is None else str(self.calibration_json_path),
                None if self.current_best_camera_setting_json_path is None else str(self.current_best_camera_setting_json_path),
                self.sim.width,
                self.sim.height,
                self.settle_steps,
                self.binary_weight,
                self.depth_weight,
                None if fixed_joint_positions is None else np.asarray(fixed_joint_positions, dtype=np.float64),
                None if fixed_camera_parameter is None else np.asarray(fixed_camera_parameter, dtype=np.float64),
                None if robot_pose_parameter is None else np.asarray(robot_pose_parameter, dtype=np.float64),
            ),
        )

        return executor, worker_count

    def _open_shared_executor(self):
        self._shared_executor, self._shared_worker_count = self._make_executor()

    def _close_shared_executor(self):
        if self._shared_executor is not None:
            self._shared_executor.shutdown(wait=True, cancel_futures=True)
        self._shared_executor = None
        self._shared_worker_count = None

    def evaluate_population_parallel(
        self,
        executor,
        worker_count,
        samples,
        target_mask,
        mode,
    ):
        if executor is None:
            if mode == "joint":
                return [
                    self.evaluate_action(target_mask, sample)
                    for sample in samples
                ]
            if mode == "pose":
                return [
                    self.evaluate_pose_with_joint(target_mask, sample)
                    for sample in samples
                ]
            
            raise ValueError(f"지원하지 않는 mode입니다: {mode}")

        sample_chunks = [
            chunk
            for chunk in np.array_split(samples, worker_count)
            if len(chunk) > 0
        ]

        batch_args = [
            (chunk, target_mask)
            for chunk in sample_chunks
        ]

        if mode == "joint":
            chunk_results = list(executor.map(_worker_evaluate_joint_batch, batch_args))
        elif mode == "pose":
            chunk_results = list(executor.map(_worker_evaluate_pose_with_joint_batch, batch_args))
        else:
            raise ValueError(f"지원하지 않는 mode입니다: {mode}")

        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def optimize_robot_pose(self, target_mask, initial_joint_mean, initial_pose_mean,  absolute_frame_idx, frame_stem):

        frame_start_time = time.perf_counter()

        if self.is_cold_start == True:
            iterations = self.cold_iteration

            joint_mean = np.clip(DEFAULT_HOME_QPOS.copy(), self.joint_lower, self.joint_upper)
            pose_mean = np.array([initial_pose_mean[0], initial_pose_mean[1], initial_pose_mean[2], 0.0, 0.0, 0.0])

            standard_joint_mean = (joint_mean - self.joint_lower) / self.joint_range
            standard_pose_mean = (pose_mean - self.pose_lower) / self.pose_range

            joint_std = np.array(self.cold_joint_std)
            pose_std = np.array(self.cold_std)

            joint_min_std = self.cold_min_std
            pose_min_std = self.cold_min_std

        elif self.is_middle_start == True:
            iterations = self.middle_iteration
            
            joint_mean = np.clip(initial_joint_mean, self.joint_lower, self.joint_upper)
            pose_mean = np.array(initial_pose_mean, dtype=np.float64)

            standard_joint_mean = (joint_mean - self.joint_lower) / self.joint_range
            standard_pose_mean = (pose_mean - self.pose_lower) / self.pose_range

            joint_std = np.array(self.middle_joint_std)
            pose_std = np.array(self.middle_pose_std)
            
            joint_min_std = self.middle_min_std
            pose_min_std = self.middle_min_std

        else:
            iterations = self.uncold_iteration
            
            joint_mean = np.clip(initial_joint_mean, self.joint_lower, self.joint_upper)
            pose_mean = np.array(initial_pose_mean, dtype=np.float64)

            standard_joint_mean = (joint_mean - self.joint_lower) / self.joint_range
            standard_pose_mean = (pose_mean - self.pose_lower) / self.pose_range

            joint_std = np.array(self.uncold_joint_std)
            pose_std = np.array(self.uncold_pose_std)
            
            joint_min_std = self.uncold_min_std
            pose_min_std = self.uncold_min_std


        best_result = None

        executor = self._shared_executor
        worker_count = self._shared_worker_count
        owns_executor = worker_count is None

        # self.joint_mode 추가하기

        if owns_executor:
            executor, worker_count = self._make_executor(
                fixed_joint_positions=joint_mean,
                robot_pose_parameter=pose_mean,
            )


        try:
            for iteration in range(iterations):

                if self.is_cold_start==True:
                    
                    ratio = min(iteration / iterations, 1.0)
                    cosine = 0.5 * (1 + math.cos(math.pi * ratio))

                    joint_std = self.cold_min_std + (self.cold_joint_std - self.cold_min_std) * cosine
                    pose_std = self.cold_min_std + (self.cold_pose_std - self.cold_min_std) * cosine
                    # joint_std = self.cold_min_std + (self.cold_joint_std - self.cold_min_std) * 0.975 ** iteration
                    # pose_std = self.cold_min_std + (self.cold_pose_std - self.cold_min_std) * 0.975 ** iteration
                    # print(f"[DEBUG] sampling joint_std : {joint_std}")
                    # print(f"[DEBUG] sampling poses td : {pose_std}")
                                

                iteration_start_time = time.perf_counter()

                # print(f"[!] Sampling joint std : {joint_std}")
                # print(f"[!] Sampling pose std : {pose_std}")
                standard_joint_samples = self.sample_population(standard_joint_mean, joint_std)
                standard_pose_samples = self.sample_population(standard_pose_mean, pose_std)

                # rescale to origin
                joint_samples = (standard_joint_samples * self.joint_range) + self.joint_lower
                pose_samples = (standard_pose_samples * self.pose_range) + self.pose_lower

                joint_samples = np.clip(joint_samples, self.joint_lower, self.joint_upper)

                samples = np.column_stack((joint_samples,pose_samples))

                results = self.evaluate_population_parallel(
                    executor=executor,
                    worker_count=worker_count,
                    samples=samples,
                    target_mask=target_mask,
                    mode="pose",
                )

                scores = np.array([result["score"] for result in results], dtype=np.float64)
                
                elite_count = max(1, int(round(len(results) * self.elite_fraction)))
                self.elite_count = elite_count
                elite_indices = np.argsort(scores)[-elite_count :]
                elite_joint_samples = np.array([results[idx]["joint_positions"] for idx in elite_indices], dtype=np.float64)
                elite_pose_samples = np.array([results[idx]["robot_pose"] for idx in elite_indices], dtype=np.float64)
                
                elite_scores = scores[elite_indices]

                standard_elite_joint_samples = (elite_joint_samples - self.joint_lower) / self.joint_range
                standard_elite_pose_samples = (elite_pose_samples - self.pose_lower) / self.pose_range

                standard_joint_new_mean = standard_elite_joint_samples.mean(axis=0) 
                standard_joint_new_std = standard_elite_joint_samples.std(axis=0)

                standard_pose_new_mean = standard_elite_pose_samples.mean(axis=0) 
                standard_pose_new_std = standard_elite_pose_samples.std(axis=0)

                joint_new_mean = elite_joint_samples.mean(axis=0)
                pose_new_mean = elite_pose_samples.mean(axis=0)

               
                # Debugging, Mean of Elite
                # _, robot_mask = self.render_pose(joint_new_mean, pose_new_mean)
                # robot_mask = robot_mask.astype(bool)
                # pred_mask = np.where(robot_mask, 255, 0).astype(np.uint8)

                # save_path = f"{self.cem_output_debug}/debug.png"
                # cv2.imwrite(save_path, pred_mask)

                # raw data update
                joint_mean = self.smoothing * joint_mean + (1.0 - self.smoothing) * joint_new_mean
                pose_mean = self.smoothing * pose_mean + (1.0 - self.smoothing) * pose_new_mean

                # standard data update
                standard_joint_mean = self.smoothing * standard_joint_mean + (1.0 - self.smoothing) * standard_joint_new_mean
                standard_joint_std = self.smoothing * joint_std + (1.0 - self.smoothing) * standard_joint_new_std

                standard_pose_mean = self.smoothing * standard_pose_mean + (1.0 - self.smoothing) * standard_pose_new_mean
                standard_pose_std = self.smoothing * pose_std + (1.0 - self.smoothing) * standard_pose_new_std
                    
                joint_std = standard_joint_std
                pose_std = standard_pose_std

                joint_std = np.clip(joint_std, joint_min_std, None)
                pose_std = np.clip(pose_std, pose_min_std, None)

                # print("=====DEBUG=====")
                # print(f"joint min-std : {joint_min_std}")
                # print(f"pose min-std : {pose_min_std}")

                # print("=====DEBUG=====")
                # print(f"elite joint std : {standard_joint_std}")
                # print(f"elite pose std : {standard_pose_std}")

                iteration_best = results[int(np.argmax(scores))]

                # print(f"Total Reward : {iteration_best['score']}")

                if best_result is None or iteration_best["score"] > best_result["score"]:
                    best_result = iteration_best

                    # clipped, pred_mask= self.render_pose(best_result["joint_positions"], best_result["robot_pose"])

                    if self.is_cold_start:
                        save_best_bin = f"{self.cem_output_init}/Init.png"
                    else:
                        save_best_bin = self.cem_output_frames / f"{self.frame_num:06d}.png"
                        
                    # cv2.imwrite(save_best_bin, pred_mask)


                iteration_elapsed_sec = time.perf_counter() - iteration_start_time

                if (iteration + 1) % self.log_interval == 0:
                    # self.log(
                    #     f"[FRAME {absolute_frame_idx:04d}] "
                    #     f"{frame_stem} iter={iteration + 1}/{iterations} "
                    #     f"iter_reward={iteration_best['score']:.4f} "
                    #     f"running_best_reward={best_result['score']:.4f} "
                    #     f"iter_elapsed={iteration_elapsed_sec:.2f}s"
                    # )

                    with open(self.note_path, "a") as f:
                        f.write(
                            f"[FRAME {absolute_frame_idx:04d}] "
                            f"{frame_stem} iter={iteration + 1}/{iterations} "
                            f"iter_reward={iteration_best['score']:.4f} "
                            f"running_best_reward={best_result['score']:.4f} "
                            f"iter_elapsed={iteration_elapsed_sec:.2f}s\n"
                        )
                   

        finally:
            if owns_executor and executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        # [TODO] Modify self.refine_best_result
        # best_result = self.refine_best_result(target_mask, target_depth, best_result)
        frame_elapsed_sec = time.perf_counter() - frame_start_time

        with open(self.note_path, "a") as f:
            f.write("\n")

        return best_result, frame_elapsed_sec
    

    def build_payload(self, sequence, total_elapsed_sec):
        return {
            "created_at": datetime.now().isoformat(),
            "scene_xml_path": str(self.sim.model_path),
            "dream_root": str(self.dream_root),
            "benchmark": self.benchmark,
            "segmentation_binary_dir": str(self.input_binary),
            "output_json_path": str(self.output_json_path),
            "sequence_name": None if self.current_task is None else self.current_task["task_name"],
            "sequence_sensor": None if self.current_task is None else self.current_task["sensor"],
            "sequence_root": None if self.current_task is None else str(self.current_task["task_root"]),
            "camera_name": "real_view_cam",
            "image_width": self.sim.width,
            "image_height": self.sim.height,
            "input_frame_layout": "Data/Dream/<Benchmark>/<frame_id>/{best_camera_setting.json,<binary_mask_image>}",
            "objective": "maximize weighted binary segmentation alignment score",
            "binary_score_metric": "dot product over flattened (+1,-1) binary masks",
            "depth_loss_metric": "currently unused in pose mode; worker depth term is fixed to 0",
            "depth_excluded_values": DEPTH_EXCLUDED_VALUES.tolist(),
            "total_cem_elapsed_sec": total_elapsed_sec,
            "num_frames_processed": len(sequence),
            "cem_config": {
                "population_size": self.population_size,
                "elite_fraction": self.elite_fraction,
                "elite_count": self.elite_count,
                "smoothing": self.smoothing,
                "settle_steps": self.settle_steps,
                "local_refine_steps": self.local_refine_steps,
                "binary_weight": self.binary_weight,
                "depth_weight": self.depth_weight,
                "start_frame": self.start_frame,
                "max_frames": self.max_frames,
                "log_interval": self.log_interval,
                "quiet": self.quiet,
                "num_workers": self.num_workers,
            },
            "frames": sequence,
        }

    def save_payload(self, payload):
        #self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        
        with open(self.save_result_path / 'result.json', "w") as f:
            json.dump(payload, f, indent=2)

    def run(self):
        self._open_shared_executor()
        try:
            return self._run_impl()
        finally:
            self._close_shared_executor()

    def _run_impl(self):
        if not self.frame_pairs:
            raise FileNotFoundError("GT 프레임 쌍이 없습니다.")

        run_start_time = time.perf_counter()
        sequence = []

        joint_current_mean = np.clip(DEFAULT_HOME_QPOS.copy(), self.joint_lower, self.joint_upper)
        pose_current_mean = np.array([0 ,0, 0.7, 0.0, 0.0, 0.0])

        best_cam_parameter = None

        self.log(
            f"[START] frames={len(self.frame_pairs)} "
        )

        ### Area of Cold Start(for start frame)
        frame_pair = self.frame_pairs[0]
        frame_pair = self.frame_pairs[731]



        target_mask = load_binary_image(
                frame_pair["binary_path"],
                width=self.sim.width,
                height=self.sim.height,
            )


        self.log(
                f"[COLD START CAM]"
                f"binary={frame_pair['binary_path'].name}"
            )

        best_result, frame_elapsed_sec = self.optimize_robot_pose(
            target_mask=target_mask,
            initial_joint_mean=joint_current_mean,
            initial_pose_mean=pose_current_mean,
            absolute_frame_idx=0,
            frame_stem=frame_pair["stem"],
        )

        self.set_sim_joint_positions(best_result["joint_positions"])
        ee_pose = self.sim.get_end_effector_pose()

        frame_result = {
            "frame_index": "COLD-START, ROBOT-MOVING",
            "frame_stem": frame_pair["stem"],
            "binary_image_file": frame_pair["binary_path"].name,
            "joint_positions": best_result["joint_positions"],
            "robot_pose" : best_result["robot_pose"],
            "reward": best_result["score"],
            "end_effector_position": ee_pose["position"],
            "end_effector_rotation_matrix": ee_pose["rotation_matrix"],
            "cem_elapsed_sec": frame_elapsed_sec,
        }

        sequence.append(frame_result)

        joint_current_mean = np.array(best_result["joint_positions"], dtype=np.float64)
        robot_current_pose = np.array(best_result["robot_pose"], dtype=np.float64)
            
        payload = self.build_payload(
            sequence=sequence,
            total_elapsed_sec=time.perf_counter() - run_start_time,
        )
        self.save_payload(payload)
        self.log(
            f"[COLD START RESULT]"
            f"reward={best_result['score']:.4f} "
            f"elapsed={frame_elapsed_sec:.2f}s "
            f"checkpoint={self.output_json_path}"
        )

        self.is_cold_start = False
        
        # Same logic as above(for all frames)
        for frame_offset, frame_pair in enumerate(self.frame_pairs[731:]):

            absolute_frame_idx = self.start_frame + frame_offset

            target_mask = load_binary_image(
                frame_pair["binary_path"],
                width=self.sim.width,
                height=self.sim.height,
            )

            self.log(
                f"[FRAME {absolute_frame_idx:04d}] start stem={frame_pair['stem']} "
                f"binary={frame_pair['binary_path'].name}"
            )

            best_result, frame_elapsed_sec = self.optimize_robot_pose(
            target_mask=target_mask,
            initial_joint_mean=joint_current_mean,
            initial_pose_mean=robot_current_pose,
            absolute_frame_idx=0,
            frame_stem=frame_pair["stem"],
            )   

            self.set_sim_joint_positions(best_result["joint_positions"])
            ee_pose = self.sim.get_end_effector_pose()

            frame_result = {
                "frame_index": absolute_frame_idx,
                "frame_stem": frame_pair["stem"],
                "binary_image_file": frame_pair["binary_path"].name,
                "joint_positions": best_result["joint_positions"],
                "robot_pose" : best_result["robot_pose"],
                "reward": best_result["score"],
                "end_effector_position": ee_pose["position"],
                "end_effector_rotation_matrix": ee_pose["rotation_matrix"],
                "cem_elapsed_sec": frame_elapsed_sec,
            }

            sequence.append(frame_result)

            joint_current_mean = np.array(best_result["joint_positions"], dtype=np.float64)
            robot_current_pose = np.array(best_result["robot_pose"], dtype=np.float64)

            payload = self.build_payload(
                sequence=sequence,
                total_elapsed_sec=time.perf_counter() - run_start_time,
            )
            self.save_payload(payload)
            self.log(
                f"[FRAME {absolute_frame_idx:04d}] {frame_pair['stem']} "
                f"reward={best_result['score']:.4f} "
                f"elapsed={frame_elapsed_sec:.2f}s "
                f"checkpoint={self.output_json_path}"
            )
            self.frame_num += 1

        total_elapsed_sec = time.perf_counter() - run_start_time
        payload = self.build_payload(sequence=sequence, total_elapsed_sec=total_elapsed_sec)
        self.save_payload(payload)

        self.log(f"[DONE] 결과 저장: {self.output_json_path}")
        self.log(f"[TIME] total_cem_elapsed_sec={total_elapsed_sec:.2f}")
        return payload


    def pose_run(self):
        return self._pose_run_impl()


    def _finalize_pose_frame_result(self, frame_pair, absolute_frame_idx, best_result, frame_elapsed_sec):
        self.set_sim_joint_positions(best_result["joint_positions"])
        _, pred_mask = self.render_pose(best_result["joint_positions"], best_result["robot_pose"])
        rendered_frame_path = self.cem_output_frames / f"{frame_pair['frame_id']}.png"
        cv2.imwrite(rendered_frame_path, pred_mask)
        ee_pose = self.sim.get_end_effector_pose()

        return {
            "frame_index": absolute_frame_idx,
            "frame_id": frame_pair["frame_id"],
            "frame_stem": frame_pair["stem"],
            "frame_dir": str(frame_pair["frame_dir"]),
            "binary_image_file": frame_pair["binary_path"].name,
            "binary_image_path": str(frame_pair["binary_path"]),
            "camera_json_path": str(frame_pair["camera_json_path"]),
            "rendered_frame_path": str(rendered_frame_path),
            "joint_positions": best_result["joint_positions"],
            "robot_pose": best_result["robot_pose"],
            "reward": best_result["score"],
            "end_effector_position": ee_pose["position"],
            "end_effector_rotation_matrix": ee_pose["rotation_matrix"],
            "cem_elapsed_sec": frame_elapsed_sec,
        }

    def _run_first_frame_pose_cem(self, frame_pair, absolute_frame_idx, pose_attempt_count):
        initial_joint_mean = np.clip(DEFAULT_HOME_QPOS.copy(), self.joint_lower, self.joint_upper)
        
        initial_pose_mean = np.array([0, 0, 0.7, 0.0, 0.0, 0.0], dtype=np.float64)

        target_mask = load_binary_image(
            frame_pair["binary_path"],
            width=self.sim.width,
            height=self.sim.height,
        )

        # self.log(
        #     f"[TASK {self.current_task['task_name']}] "
        #     f"[FRAME {absolute_frame_idx:04d}] start stem={frame_pair['stem']} "
        #     f"binary={frame_pair['binary_path'].name}"
        # )

        frame_start_time = time.perf_counter()
        best_result = None
        best_attempt_idx = None

        for attempt_idx in range(1, pose_attempt_count + 1):
            if self.random_seed is None:
                attempt_seed = None
                self.rng = np.random.default_rng()
            else:
                attempt_seed = self.random_seed + self.attempt_seed_offset
                self.rng = np.random.default_rng(attempt_seed)
                self.attempt_seed_offset += 1

            self.is_cold_start = True
            self.is_middle_start = False

            joint_current_mean = initial_joint_mean.copy()
            pose_current_mean = initial_pose_mean.copy()

            self.log(
                f"[TASK {self.current_task['task_name']}] "
                f"[FRAME {absolute_frame_idx:04d}] "
                f"attempt={attempt_idx}/{pose_attempt_count} "
                f"seed={attempt_seed} start"
            )

            attempt_result, cold_elapsed_sec = self.optimize_robot_pose(
                target_mask=target_mask,
                initial_joint_mean=joint_current_mean,
                initial_pose_mean=pose_current_mean,
                absolute_frame_idx=absolute_frame_idx,
                frame_stem=frame_pair["stem"],
            )

            self.is_cold_start = False

            joint_current_mean = np.array(attempt_result["joint_positions"], dtype=np.float64)
            robot_current_pose = np.array(attempt_result["robot_pose"], dtype=np.float64)

            self.is_middle_start = True

            attempt_result, refine_elapsed_sec = self.optimize_robot_pose(
                target_mask=target_mask,
                initial_joint_mean=joint_current_mean,
                initial_pose_mean=robot_current_pose,
                absolute_frame_idx=absolute_frame_idx,
                frame_stem=frame_pair["stem"],
            )

            attempt_elapsed_sec = cold_elapsed_sec + refine_elapsed_sec
            self.log(
                f"[TASK {self.current_task['task_name']}] "
                f"[FRAME {absolute_frame_idx:04d}] "
                f"attempt={attempt_idx}/{pose_attempt_count} "
                f"reward={attempt_result['score']:.4f} "
                f"elapsed={attempt_elapsed_sec:.2f}s"
            )

            if best_result is None or attempt_result["score"] > best_result["score"]:
                best_result = attempt_result
                best_attempt_idx = attempt_idx

        frame_elapsed_sec = time.perf_counter() - frame_start_time
        self.is_cold_start = False
        self.is_middle_start = False

        return {
            "best_result": best_result,
            "best_attempt_idx": best_attempt_idx,
            "frame_elapsed_sec": frame_elapsed_sec,
        }

    def _run_pose_task(self, task):
        self._configure_pose_task(task)
        if not self.frame_pairs:
            raise FileNotFoundError(f"GT 프레임 쌍이 없습니다: {task['task_root']}")

        run_start_time = time.perf_counter()
        sequence = []
        pose_attempt_count = 15

        self.log(
            f"[SEQUENCE] {task['task_name']} "
            f"frames={len(self.frame_pairs)}"
        )

        for frame_offset, frame_pair in enumerate(self.frame_pairs):
            absolute_frame_idx = self.start_frame + frame_offset
            self._apply_pose_task_camera(frame_pair)

            self._open_shared_executor()
            try:
                frame_run = self._run_first_frame_pose_cem(
                    frame_pair=frame_pair,
                    absolute_frame_idx=absolute_frame_idx,
                    pose_attempt_count=pose_attempt_count,
                )
            finally:
                self._close_shared_executor()

            frame_result = self._finalize_pose_frame_result(
                frame_pair=frame_pair,
                absolute_frame_idx=absolute_frame_idx,
                best_result=frame_run["best_result"],
                frame_elapsed_sec=frame_run["frame_elapsed_sec"],
            )
            sequence.append(frame_result)

            payload = self.build_payload(
                sequence=sequence,
                total_elapsed_sec=time.perf_counter() - run_start_time,
            )
            self.save_payload(payload)
            self.log(
                f"[TASK {task['task_name']}] "
                f"[FRAME {absolute_frame_idx:04d}] {frame_pair['frame_id']} "
                f"reward={frame_run['best_result']['score']:.4f} "
                f"best_attempt={frame_run['best_attempt_idx']}/{pose_attempt_count} "
                f"elapsed={frame_run['frame_elapsed_sec']:.2f}s "
                f"checkpoint={self.output_json_path}"
            )
            self.frame_num += 1

        total_elapsed_sec = time.perf_counter() - run_start_time
        payload = self.build_payload(sequence=sequence, total_elapsed_sec=total_elapsed_sec)
        self.save_payload(payload)

        self.log(f"[TASK {task['task_name']}] 결과 저장: {self.output_json_path}")
        self.log(f"[TASK {task['task_name']}] total_cem_elapsed_sec={total_elapsed_sec:.2f}")
        return payload

    def _pose_run_impl(self):
        if not self.video_bench_tasks:
            raise FileNotFoundError(f"Dream sequence를 찾을 수 없습니다: {self.dream_root}")

        task = self.video_bench_tasks[0]
        self.log(f"[SEQUENCE] {task['task_name']} start")
        return self._run_pose_task(task)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary GT와 depth GT를 함께 사용해 FR3 joint action sequence를 CEM으로 추정합니다."
    )
    parser.add_argument("--scene-xml", type=Path, default=None)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--dream-root", type=Path, default=DREAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--population-size", type=int, default=10000)
    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--min-std", type=float, default=0.01)
    parser.add_argument("--cold-joint-iteration", type=int, default=150)
    parser.add_argument("--uncold-joint-iteration", type=int, default=30)
    parser.add_argument("--cold-entire-iteration", type=int, default=150)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--global-sample-fraction", type=float, default=0.20)
    parser.add_argument("--local-refine-steps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--binary-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.000001)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exp-set", type=str)
    parser.add_argument("--is-min-std", type=bool, default=True)
    parser.add_argument("--test-mode", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seg-model", type=str, default="sam3")
    parser.add_argument("--benchmark", type=str, default="azure")
    parser.add_argument("--cem-mode", type=str, default="pose")
    parser.add_argument("--pid", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    pid = args.pid
    pid_to_data = []
    # Azure / 000000 - 006393
    for i in range(6394):
        pid_to_data.append(('Azure', i, str(i).zfill(6)))
    # Kinect / 000000 - 004965
    for i in range(4966):
        pid_to_data.append(('Kinect', i, str(i).zfill(6)))
    # Realsense / 000000 - 005943
    for i in range(5944):
        pid_to_data.append(('Realsense', i, str(i).zfill(6)))

    input_root = None if args.input_root is None else Path(args.input_root)

    input_root = input_root / pid_to_data[pid][0]
    start_frame = pid_to_data[pid][1]
    max_frames = 1
    save_result_path = Path('/mnt/nfs/Results/' + pid_to_data[pid][0] +'/' + pid_to_data[pid][2])
    (save_result_path).mkdir(parents=True, exist_ok=True)

    benchmark = (
        benchmark_name_from_input_root(input_root)
        if input_root is not None
        else normalize_benchmark_name(args.benchmark)
    )
    xml = resolve_scene_xml_path(args.scene_xml, benchmark)

    sim = MuJoCoRobotDepthSim(
        model_path=xml,
        calibration_path=args.calibration_json,
        width=args.width,
        height=args.height,
    )

    try:
        predictor = CEMActionPredictor(
            sim=sim,
            population_size=args.population_size,
            elite_fraction=args.elite_fraction,
            smoothing=args.smoothing,
            settle_steps=args.settle_steps,
            local_refine_steps=args.local_refine_steps,
            random_seed=args.random_seed,
            binary_weight=args.binary_weight,
            depth_weight=args.depth_weight,
            start_frame=start_frame,
            max_frames=max_frames,
            log_interval=args.log_interval,
            quiet=args.quiet,
            test_mode=args.test_mode,
            num_workers=args.num_workers,
            seg_model=args.seg_model,
            benchmark=benchmark,
            cem_mode=args.cem_mode,
            input_root=input_root,
            dream_root=args.dream_root,
            output_root=args.output_root,
            scene_xml_path=xml,
            calibration_json_path=args.calibration_json,
            save_result_path=save_result_path,
        )

        # predictor = CEMActionPredictor(
        #     sim=sim,
        #     population_size=args.population_size,
        #     elite_fraction=args.elite_fraction,
        #     smoothing=args.smoothing,
        #     settle_steps=args.settle_steps,
        #     local_refine_steps=args.local_refine_steps,
        #     random_seed=args.random_seed,
        #     binary_weight=args.binary_weight,
        #     depth_weight=args.depth_weight,
        #     start_frame=args.start_frame,
        #     max_frames=args.max_frames,
        #     log_interval=args.log_interval,
        #     quiet=args.quiet,
        #     test_mode=args.test_mode,
        #     num_workers=args.num_workers,
        #     seg_model=args.seg_model,
        #     benchmark=benchmark,
        #     cem_mode=args.cem_mode,
        #     input_root=input_root,
        #     dream_root=args.dream_root,
        #     output_root=args.output_root,
        #     scene_xml_path=xml,
        #     calibration_json_path=args.calibration_json,
        # )

        if args.cem_mode == "pose":
            predictor.pose_run()
        else:
            predictor.run()
    finally:
        sim.close()

if __name__ == "__main__":
    main()
