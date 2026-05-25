import json
import os
from pathlib import Path

import cv2

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np

BASE_DIR = Path("/home/taeyeong/VSCODE/Mujoco/AI_video2action")
SCENE_XML_PATH = BASE_DIR / "data/model/robots/fr3/scene.xml"
CALIB_PATH = BASE_DIR / "data/checker_board/camera_calibration_result.json"
DEFAULT_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854], dtype=np.float64)


class MuJoCoSim:
    def __init__(
        self,
        model_path,
        calibration_path,
        width=1280,
        height=720,
        home_qpos=None,
    ):
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path)
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

        self.arm_joint_count = min(7, self.model.nq, self.model.nu if self.model.nu > 0 else 7)
        self.load_camera_params(self.calibration_path)
        self.setup_camera()
        self.arm_joint_bounds = self._build_joint_bounds()
        self.reset()

    def load_camera_params(self, path):
        if not Path(path).exists():
            self.fovy_deg = 60.0
            return

        with open(path, "r") as f:
            data = json.load(f)
        fy = data["camera_matrix"][1][1]
        height = data["image_height"]
        self.fovy_deg = np.rad2deg(2 * np.arctan(height / (2 * fy)))

    def setup_camera(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "real_view_cam")
        self.real_view_cam_id = cam_id
        if cam_id != -1:
            self.model.cam_fovy[cam_id] = self.fovy_deg

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fr3_hand_tcp")
        if self.ee_site_id == -1:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

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

        self.data.qpos[: self.arm_joint_count] = clipped
        if self.model.nu > 0:
            self.data.ctrl[: self.arm_joint_count] = clipped

        mujoco.mj_forward(self.model, self.data)

        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        return clipped

    def render_rgb(self, camera_name="real_view_cam"):
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def render_binary_mask(self, camera_name="real_view_cam", threshold=127):
        img_rgb = self.render_rgb(camera_name=camera_name)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.where(img_gray > threshold, 255, 0).astype(np.uint8)

    def get_end_effector_pose(self):
        if self.ee_site_id == -1:
            return {"position": None, "rotation_matrix": None}

        position = self.data.site_xpos[self.ee_site_id].copy()
        rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return {
            "position": position.tolist(),
            "rotation_matrix": rotation.tolist(),
        }

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def run(self):
        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            if self.real_view_cam_id != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = self.real_view_cam_id

            while viewer.is_running():
                viewer.sync()

                img_rgb = self.render_rgb(camera_name="real_view_cam")
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Real View Camera", img_bgr)

                if cv2.waitKey(1) == 27:
                    break

        cv2.destroyAllWindows()
        self.close()


if __name__ == "__main__":
    if SCENE_XML_PATH.exists():
        sim = MuJoCoSim(SCENE_XML_PATH, CALIB_PATH)
        sim.run()
    else:
        print(f"[ERROR] scene.xml 파일을 찾을 수 없습니다: {SCENE_XML_PATH}")
