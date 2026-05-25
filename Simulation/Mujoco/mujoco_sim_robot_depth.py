import cv2
import time
import mujoco
import mujoco.viewer
import numpy as np

from Simulation.Mujoco.mujoco_sim_rgb import SCENE_XML_PATH, MuJoCoRGBSim


class MuJoCoRobotDepthSim(MuJoCoRGBSim):
    def __init__(
        self,
        model_path,
        calibration_path=None,
        width=640,
        height=480,
        home_qpos=None,
    ):
        super().__init__(
            model_path=model_path,
            calibration_path=calibration_path,
            width=width,
            height=height,
            home_qpos=home_qpos,
        )

        self.robot_visual_geom_group = 2
        self.robot_scene_option = self._build_robot_scene_option()

    def _resolve_render_camera(self, camera_name):
        render_camera = camera_name
        if render_camera is None:
            return -1
        if isinstance(render_camera, str):
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, render_camera)
            return render_camera if cam_id != -1 else -1
        return render_camera

    def _build_robot_scene_option(self):
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[self.robot_visual_geom_group] = 1

        if hasattr(scene_option, "sitegroup"):
            scene_option.sitegroup[:] = 0
        if hasattr(scene_option, "flexgroup"):
            scene_option.flexgroup[:] = 0
        if hasattr(scene_option, "frame"):
            scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE

        return scene_option

    def render_depth_map(self, camera_name="real_view_cam", scene_option=None):
        render_camera = self._resolve_render_camera(camera_name)

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=render_camera, scene_option=scene_option)
        depth_map = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        # print(depth_map)
        # print(np.max(depth_map), np.min(depth_map))

        # [[147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]
        # [147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]
        # [147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]
        # ...
        # [147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]
        # [147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]
        # [147.9476 147.9476 147.9476 ... 147.9476 147.9476 147.9476]]
        # 147.9476 0.9904433
        return depth_map

    def compute_robot_camera_distance(self, camera_name="real_view_cam"):
        depth_map = self.render_depth_map(
            camera_name=camera_name,
            scene_option=self.robot_scene_option,
        )
        segmentation_map = self.render_segmentation_map(
            camera_name=camera_name,
            scene_option=self.robot_scene_option,
        )
        robot_mask = self.build_robot_mask(segmentation_map)
        return depth_map, robot_mask
    def render_segmentation_map(self, camera_name="real_view_cam", scene_option=None):
        
        render_camera = self._resolve_render_camera(camera_name)

        self.renderer.enable_segmentation_rendering()

        self.renderer.update_scene(self.data, camera=render_camera, scene_option=scene_option)

        # Critical
        seg_map = self.renderer.render().copy()

        self.renderer.disable_segmentation_rendering()

        return seg_map

    def build_robot_mask(self, segmentation_map):
        if segmentation_map.ndim != 3 or segmentation_map.shape[-1] < 2:
            return np.zeros(segmentation_map.shape[:2], dtype=bool)

        # MuJoCo segmentation output is (obj_id, obj_type), not the reverse.
        obj_ids = segmentation_map[..., 0]
        obj_types = segmentation_map[..., 1]

        geom_pixels = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)
        valid_geom_pixels = geom_pixels & (obj_ids >= 0)
        return valid_geom_pixels

    def binary_segmentation(self, camera_name="real_view_cam"):

        segmentation_map = self.render_segmentation_map(
            camera_name=camera_name,
            scene_option=self.robot_scene_option,
        )

        robot_mask = self.build_robot_mask(segmentation_map)

        return robot_mask

    def depth_to_color(self, depth_map, robot_mask, near_percentile=5.0, far_percentile=95.0):
        depth_vis = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        valid_mask = robot_mask & np.isfinite(depth_map) & (depth_map > 0.0)
        if not np.any(valid_mask):
            return depth_vis

        valid_depth = depth_map[valid_mask]
        near_value = float(np.percentile(valid_depth, near_percentile))
        far_value = float(np.percentile(valid_depth, far_percentile))

        if far_value - near_value < 1e-8:
            normalized = np.zeros_like(depth_map, dtype=np.float32)
        else:
            normalized = (depth_map - near_value) / (far_value - near_value)
            normalized = np.clip(normalized, 0.0, 1.0)

        depth_uint8 = ((1.0 - normalized) * 255.0).astype(np.uint8)
        colored_depth_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        colored_depth_rgb = cv2.cvtColor(colored_depth_bgr, cv2.COLOR_BGR2RGB)
        depth_vis[valid_mask] = colored_depth_rgb[valid_mask]
        return depth_vis

    def render_robot_depth_colormap(self, camera_name="real_view_cam"):
        depth_map, robot_mask = self.compute_robot_camera_distance(camera_name=camera_name)
        robot_depth_rgb = self.depth_to_color(depth_map, robot_mask)
        return robot_depth_rgb, robot_mask

    def render_robot_depth_overlay(self, camera_name="real_view_cam"):
        base_rgb = self.render_rgb(camera_name=camera_name).copy()
        robot_depth_rgb, robot_mask = self.render_robot_depth_colormap(camera_name=camera_name)
        if np.any(robot_mask):
            base_rgb[robot_mask] = robot_depth_rgb[robot_mask]
        return base_rgb

    def render_robot_rgb_depth_segmentation(self, camera_name="real_view_cam", rgb_weight=0.0):
        robot_depth_rgb, robot_mask = self.render_robot_depth_colormap(camera_name=camera_name)

        segmented_rgb = np.zeros_like(robot_depth_rgb, dtype=np.uint8)
        if not np.any(robot_mask):
            return segmented_rgb, robot_mask

        rgb_weight = float(np.clip(rgb_weight, 0.0, 1.0))
        depth_weight = 1.0 - rgb_weight

        blended_robot = robot_depth_rgb[robot_mask].astype(np.float32) * depth_weight
        if rgb_weight > 0.0:
            robot_rgb = self.render_rgb(camera_name=camera_name).copy()
            blended_robot += robot_rgb[robot_mask].astype(np.float32) * rgb_weight

        segmented_rgb[robot_mask] = np.clip(blended_robot, 0.0, 255.0).astype(np.uint8)
        return segmented_rgb, robot_mask

    def render_robot_depth_segmentation(self, camera_name="real_view_cam"):
        segmented_rgb, _ = self.render_robot_rgb_depth_segmentation(camera_name=camera_name)
        return segmented_rgb

    def run(self):
        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            while viewer.is_running():
                viewer.sync()

                segmented_rgb = self.render_robot_depth_segmentation(camera_name="real_view_cam")
                segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("FR3 RGB-Depth Segmentation", segmented_bgr)

                if cv2.waitKey(1) == 27:
                    break

        cv2.destroyAllWindows()
        self.close()


if __name__ == "__main__":
    if SCENE_XML_PATH.exists():
        sim = MuJoCoRobotDepthSim(SCENE_XML_PATH)
        sim.run()
    else:
        print(f"[ERROR] scene.xml 파일을 찾을 수 없습니다: {SCENE_XML_PATH}")
