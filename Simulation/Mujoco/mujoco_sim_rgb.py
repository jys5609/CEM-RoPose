import json
import os
from pathlib import Path

import cv2

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np

BASE_DIR = Path(__file__).resolve()
BASE_DIR = BASE_DIR.parent.parent.parent

SCENE_XML_PATH = BASE_DIR / "Data/Model/robots/fr3/scene.xml"
CALIB_PATH = BASE_DIR / "Data/CheckerBoard/camera_calibration_result.json"

# GT Posture
# DEFAULT_HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)

# Noised Init Posture
DEFAULT_HOME_QPOS = np.array([0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0], dtype=np.float64)


class MuJoCoRGBSim:
    @staticmethod
    def read_best_camera_setting(json_path):
        json_path = Path(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)

        best = data["best"]
        pos = np.asarray(best["pos"], dtype=np.float64)
        quat = np.asarray(best["quat"], dtype=np.float64)
        xyaxes = np.fromstring(best["xyaxes"], sep=" ", dtype=np.float64)

        if pos.shape != (3,):
            raise ValueError(f"best.pos must have shape (3,), got {pos.shape} from {json_path}")
        if quat.shape != (4,):
            raise ValueError(f"best.quat must have shape (4,), got {quat.shape} from {json_path}")
        if xyaxes.shape != (6,):
            raise ValueError(f"best.xyaxes must have 6 values, got {xyaxes.shape} from {json_path}")

        return {
            "json_path": json_path,
            "camera_name": data.get("camera_name", "real_view_cam"),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "pos": pos,
            "quat": quat,
            "xyaxes": xyaxes,
            "fovy": float(best["fovy"]),
            "xml_camera_tag": best.get("xml_camera_tag", ""),
        }

    def __init__(
        self,
        model_path,
        calibration_path=None,
        width=640,
        height=480,
        home_qpos=None,
    ):
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path) if calibration_path is not None else None
        self.width = width
        self.height = height
        self.home_qpos = np.array(home_qpos if home_qpos is not None else DEFAULT_HOME_QPOS, dtype=np.float64)

        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            print(f"[SUCCESS] {self.model_path.name}")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            raise

        self.arm_joint_count = min(8, self.model.nq, self.model.nu if self.model.nu > 0 else 8)
        self.apply_basic_rgb_visuals()
        self.load_camera_params(self.calibration_path)
        self.setup_camera()
        self.arm_joint_bounds = self._build_joint_bounds()
        self.reset()

    def load_camera_params(self, path):
        self.fovy_deg = 60.0
        if path is None or not Path(path).exists():
            self.fovy_deg = 60.0
            return

        with open(path, "r") as f:
            data = json.load(f)
        fy = data["camera_matrix"][1][1]
        image_height = data["image_height"]
        self.fovy_deg = np.rad2deg(2 * np.arctan(image_height / (2 * fy)))

    def setup_camera(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "real_view_cam")
        self.real_view_cam_id = cam_id
        if cam_id != -1 and self.calibration_path is not None:
            self.model.cam_fovy[cam_id] = self.fovy_deg

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr3_hand_tcp")
        if self.ee_site_id == -1:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    def apply_basic_rgb_visuals(self):
        fr3_material_colors = {
            "black": np.array([0.18, 0.18, 0.18, 1.0], dtype=np.float32),
            "white": np.array([0.92, 0.93, 0.95, 1.0], dtype=np.float32),
            "red": np.array([0.72, 0.11, 0.12, 1.0], dtype=np.float32),
            "gray": np.array([0.62, 0.64, 0.67, 1.0], dtype=np.float32),
            "button_green": np.array([0.18, 0.55, 0.24, 1.0], dtype=np.float32),
            "button_red": np.array([0.78, 0.16, 0.16, 1.0], dtype=np.float32),
            "button_blue": np.array([0.15, 0.34, 0.78, 1.0], dtype=np.float32),
        }
        table_color = np.array([0.45, 0.30, 0.18, 1.0], dtype=np.float32)
        stand_color = np.array([0.72, 0.74, 0.78, 1.0], dtype=np.float32)
        floor_color = np.array([0.82, 0.82, 0.80, 1.0], dtype=np.float32)

        if self.model.nmat > 0:
            if hasattr(self.model, "mat_emission"):
                self.model.mat_emission[:] = 0.0
            if hasattr(self.model, "mat_specular"):
                self.model.mat_specular[:] = 0.3
            if hasattr(self.model, "mat_shininess"):
                self.model.mat_shininess[:] = 0.45
            if hasattr(self.model, "mat_reflectance"):
                self.model.mat_reflectance[:] = 0.05

            for mat_id in range(self.model.nmat):
                mat_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
                if mat_name in fr3_material_colors:
                    self.model.mat_rgba[mat_id] = fr3_material_colors[mat_name]

        if self.model.ngeom > 0:
            for geom_idx in range(self.model.ngeom):
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx)
                if geom_name == "box_geom":
                    self.model.geom_rgba[geom_idx] = stand_color
                elif geom_name and geom_name.startswith("table_"):
                    self.model.geom_rgba[geom_idx] = table_color
                elif geom_name == "floor":
                    self.model.geom_rgba[geom_idx] = floor_color

        if self.model.nlight > 0:
            self.model.light_diffuse[:] = np.array([0.9, 0.9, 0.9], dtype=np.float32)
            self.model.light_ambient[:] = np.array([0.35, 0.35, 0.35], dtype=np.float32)
            self.model.light_specular[:] = np.array([0.15, 0.15, 0.15], dtype=np.float32)

        self.model.vis.headlight.ambient[:] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self.model.vis.headlight.diffuse[:] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.model.vis.headlight.specular[:] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.model.vis.rgba.haze[:] = np.array([0.15, 0.18, 0.22, 1.0], dtype=np.float32)

    def _build_joint_bounds(self):
        bounds = np.zeros((self.arm_joint_count, 2), dtype=np.float64)
        for joint_idx in range(self.arm_joint_count):
            bounds[joint_idx] = self.model.jnt_range[joint_idx]
        return bounds

    def get_joint_bounds(self):
        return self.arm_joint_bounds.copy()

    def reset(self, joint_positions=None):
        mujoco.mj_resetData(self.model, self.data)
        target_qpos = np.array(joint_positions if joint_positions is not None else self.home_qpos, dtype=np.float64)
        self.set_joint_positions(target_qpos)

    def set_joint_positions(self, joint_positions, settle_steps=0):
        joint_positions = np.array(joint_positions, dtype=np.float64)
        clipped = np.clip(joint_positions, self.arm_joint_bounds[:, 0], self.arm_joint_bounds[:, 1])

        arm_q = clipped[:7]
        gripper_q = clipped[7]

        self.data.qpos[:7] = arm_q
        self.data.qpos[7] = gripper_q
        self.data.qpos[8] = gripper_q

        self.data.qpos[: self.arm_joint_count] = clipped
        
        if self.model.nu > 0:
            self.data.ctrl[:7] = arm_q
            self.data.ctrl[7] = np.clip(gripper_q * (255.0 / 0.04), 0.0, 255.0)

        mujoco.mj_forward(self.model, self.data)

        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        return clipped

    def rot_to_quant(self, r):
        theta = np.linalg.norm(r)

        if theta < 1e-8:
            # Small-angle approximation
            return np.array([
                0.5 * r[0],
                0.5 * r[1],
                0.5 * r[2],
                1.0
            ])

        axis = r / theta
        half_theta = 0.5 * theta

        sin_half = np.sin(half_theta)
        cos_half = np.cos(half_theta)

        q = np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])

        return q

    def set_camera_positions(self, pos, rot, settle_steps):

        quant = self.rot_to_quant(rot)
        self.model.cam_pos[self.real_view_cam_id] = np.array(pos, dtype=np.float64)
        self.model.cam_quat[self.real_view_cam_id] = np.array(quant, dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        return None

    def apply_camera_setting(self, camera_setting, camera_name=None):
        target_camera_name = camera_name or camera_setting["camera_name"]
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, target_camera_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{target_camera_name}' does not exist in {self.model_path}")

        self.model.cam_pos[cam_id] = np.asarray(camera_setting["pos"], dtype=np.float64)
        self.model.cam_quat[cam_id] = np.asarray(camera_setting["quat"], dtype=np.float64)
        self.model.cam_fovy[cam_id] = float(camera_setting["fovy"])
        mujoco.mj_forward(self.model, self.data)

        return cam_id

    def apply_best_camera_setting_json(self, json_path, camera_name=None):
        camera_setting = self.read_best_camera_setting(json_path)
        self.apply_camera_setting(camera_setting, camera_name=camera_name)
        return camera_setting
    
    def render_rgb(self, camera_name="real_view_cam"):
        render_camera = camera_name
        if render_camera is None:
            render_camera = -1
        elif isinstance(render_camera, str):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, render_camera)
            render_camera = render_camera if cam_id != -1 else -1

        self.renderer.update_scene(self.data, camera=render_camera)
        return self.renderer.render()

    def get_end_effector_pose(self):
        if self.ee_site_id == -1:
            return {"position": None, "rotation_matrix": None}

        position = self.data.site_xpos[self.ee_site_id].copy()
        rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return {
            "position": position.tolist(),
            "rotation_matrix": rotation.tolist(),
        }

    def get_robot_base_position(self):
        if self.base_body_id == -1:
            return None

        return self.data.xpos[self.base_body_id].copy()

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None

            
    def rotvec_to_mj_quat(self, rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64)
        theta = np.linalg.norm(rotvec)

        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        axis = rotvec / theta
        half_theta = 0.5 * theta
        s = np.sin(half_theta)

        # MuJoCo quaternion order: [w, x, y, z]
        return np.array([
            np.cos(half_theta),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
        ], dtype=np.float64)
    

    def set_robot_base_pose(self, pos, rot, settle_steps=0):
        if self.base_body_id == -1:
            raise ValueError("base body not found")

        self.model.body_pos[self.base_body_id] = np.asarray(pos, dtype=np.float64)
        self.model.body_quat[self.base_body_id] = self.rotvec_to_mj_quat(rot)

        mujoco.mj_forward(self.model, self.data)

        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        return {
            "position": self.model.body_pos[self.base_body_id].copy(),
            "quat": self.model.body_quat[self.base_body_id].copy(),
        }

    def run(self):
        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            while viewer.is_running():
                viewer.sync()

                img_rgb = self.render_rgb(camera_name="real_view_cam")
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Real View RGB", img_bgr)

                if cv2.waitKey(1) == 27:
                    break

        cv2.destroyAllWindows()
        self.close()


if __name__ == "__main__":
    if SCENE_XML_PATH.exists():
        sim = MuJoCoRGBSim(SCENE_XML_PATH)
        sim.set_robot_base_pose([0, 0, 0.7,],[0.0, 0.0, 1.0])
        sim.run()
    else:
        print(f"[ERROR] scene.xml 파일을 찾을 수 없습니다: {SCENE_XML_PATH}")
