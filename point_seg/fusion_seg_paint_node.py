#!/usr/bin/env python3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from openvino import Core
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from ultralytics import YOLO


class FusionSegPaintNode(Node):
    def __init__(self) -> None:
        super().__init__("point_seg_node")

        ws_root = Path("/home/nuc14/work")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("painted_cloud_topic", "/yolo_seg/painted_cloud")
        self.declare_parameter("projected_image_topic", "/yolo_seg/projected_image")
        self.declare_parameter("camera_intrinsic_yaml", "")
        self.declare_parameter("camera_extrinsic_yaml", "")
        self.declare_parameter("model_path", str(ws_root / "model" / "yolo26n-seg_openvino_model"))
        self.declare_parameter("device", "auto")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("publish_projected_image", True)
        self.declare_parameter("overlay_segmentation_mask", False)
        self.declare_parameter("overlay_segmentation_alpha", 0.35)
        self.declare_parameter("profile_timing", True)
        self.declare_parameter("passthrough_x_min", 0.0)
        self.declare_parameter("passthrough_x_max", 120.0)
        self.declare_parameter("passthrough_y_abs_max", 60.0)
        self.declare_parameter("passthrough_z_min", -5.0)
        self.declare_parameter("passthrough_z_max", 5.0)
        self.declare_parameter("voxel_leaf_size", 0.10)
        self.declare_parameter("min_track_points", 10)

        self.image_topic = self.get_parameter("image_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.painted_cloud_topic = self.get_parameter("painted_cloud_topic").value
        self.projected_image_topic = self.get_parameter("projected_image_topic").value
        self.intrinsic_yaml = self.get_parameter("camera_intrinsic_yaml").value
        self.extrinsic_yaml = self.get_parameter("camera_extrinsic_yaml").value
        self.model_path = self.get_parameter("model_path").value
        self.device_mode = self.get_parameter("device").value
        self.conf_thr = float(self.get_parameter("conf").value)
        self.iou_thr = float(self.get_parameter("iou").value)
        self.publish_projected_image = bool(self.get_parameter("publish_projected_image").value)
        self.overlay_segmentation_mask = bool(self.get_parameter("overlay_segmentation_mask").value)
        self.overlay_segmentation_alpha = float(self.get_parameter("overlay_segmentation_alpha").value)
        self.profile_timing = bool(self.get_parameter("profile_timing").value)
        self.x_min = float(self.get_parameter("passthrough_x_min").value)
        self.x_max = float(self.get_parameter("passthrough_x_max").value)
        self.y_abs_max = float(self.get_parameter("passthrough_y_abs_max").value)
        self.z_min = float(self.get_parameter("passthrough_z_min").value)
        self.z_max = float(self.get_parameter("passthrough_z_max").value)
        self.voxel_leaf_size = float(self.get_parameter("voxel_leaf_size").value)
        self.min_track_points = int(self.get_parameter("min_track_points").value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path, task="segment")
        self.infer_device = self._resolve_device()
        self.K, self.R, self.t = self._load_calibration()

        self._cloud_lock = threading.Lock()
        self._cloud_xyz = None
        self._cloud_intensity = None
        self._cloud_header = None

        self.cloud_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self._on_lidar, 10
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._on_image, 10
        )

        self.cloud_pub = self.create_publisher(PointCloud2, self.painted_cloud_topic, 10)
        self.proj_img_pub = self.create_publisher(Image, self.projected_image_topic, 10)

        self._last_log = time.time()
        self.get_logger().info(
            f"point_seg_node ready | image={self.image_topic} lidar={self.lidar_topic} "
            f"out_cloud={self.painted_cloud_topic} out_image={self.projected_image_topic} "
            f"device={self.infer_device} seg_overlay={self.overlay_segmentation_mask}"
        )

    def _resolve_device(self) -> str:
        if self.device_mode != "auto":
            return self.device_mode
        devs = set(Core().available_devices)
        if "GPU" in devs:
            return "intel:gpu"
        if "NPU" in devs:
            return "intel:npu"
        return "cpu"

    def _load_calibration(self):
        intr = yaml.safe_load(open(self.intrinsic_yaml, "r"))
        ext = yaml.safe_load(open(self.extrinsic_yaml, "r"))
        cam = np.array(intr["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        p = ext["extrinsic_calibration_by_hand"]["ros__parameters"]
        x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
        roll, pitch, yaw = np.deg2rad([float(p["roll"]), float(p["pitch"]), float(p["yaw"])])

        rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]], dtype=np.float64)
        ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float64)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float64)
        r = rz @ ry @ rx
        t = np.array([x, y, z], dtype=np.float64)
        return cam, r, t

    def _extract_cloud_fields(self, msg: PointCloud2):
        n = int(msg.width) * int(msg.height)
        if n <= 0 or msg.point_step <= 0:
            return None, None
        offsets = {f.name: f.offset for f in msg.fields}
        if not {"x", "y", "z", "intensity"}.issubset(offsets):
            return None, None
        buf = memoryview(msg.data)
        x = np.ndarray((n,), dtype=np.float32, buffer=buf, offset=offsets["x"], strides=(msg.point_step,))
        y = np.ndarray((n,), dtype=np.float32, buffer=buf, offset=offsets["y"], strides=(msg.point_step,))
        z = np.ndarray((n,), dtype=np.float32, buffer=buf, offset=offsets["z"], strides=(msg.point_step,))
        i = np.ndarray((n,), dtype=np.float32, buffer=buf, offset=offsets["intensity"], strides=(msg.point_step,))
        xyz = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)
        return xyz, i.astype(np.float32, copy=False)

    def _preprocess_cloud(self, xyz: np.ndarray, intensity: np.ndarray):
        valid = np.isfinite(xyz).all(axis=1)
        valid &= (xyz[:, 0] >= self.x_min) & (xyz[:, 0] <= self.x_max)
        valid &= (np.abs(xyz[:, 1]) <= self.y_abs_max)
        valid &= (xyz[:, 2] >= self.z_min) & (xyz[:, 2] <= self.z_max)
        xyz = xyz[valid]
        intensity = intensity[valid]
        if xyz.size == 0:
            return xyz, intensity

        if self.voxel_leaf_size > 0.0:
            vox = np.floor(xyz / self.voxel_leaf_size).astype(np.int32)
            _, idx = np.unique(vox, axis=0, return_index=True)
            idx.sort()
            xyz = xyz[idx]
            intensity = intensity[idx]
        return xyz, intensity

    def _on_lidar(self, msg: PointCloud2):
        xyz, intensity = self._extract_cloud_fields(msg)
        if xyz is None:
            return
        xyz, intensity = self._preprocess_cloud(xyz, intensity)
        with self._cloud_lock:
            self._cloud_xyz = xyz
            self._cloud_intensity = intensity
            self._cloud_header = msg.header

    def _on_image(self, msg: Image):
        with self._cloud_lock:
            if self._cloud_xyz is None or self._cloud_xyz.size == 0:
                return
            xyz = self._cloud_xyz.copy()
            intensity = self._cloud_intensity.copy()
            cloud_header = self._cloud_header

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                frame,
                device=self.infer_device,
                conf=self.conf_thr,
                iou=self.iou_thr,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"inference failed: {exc}")
            return

        h, w = frame.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint16)
        conf_mask = np.zeros((h, w), dtype=np.float32)

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            masks = r.masks
            if boxes is not None and masks is not None and masks.data is not None:
                clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.empty((0,), dtype=np.float32)
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,), dtype=np.float32)
                mask_data = masks.data.cpu().numpy()
                order = np.argsort(-confs) if len(confs) else np.arange(len(mask_data))
                for idx in order:
                    m = mask_data[idx]
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    sel = m > 0.5
                    cls_id = int(clss[idx]) + 1 if idx < len(clss) else 0
                    score = float(confs[idx]) if idx < len(confs) else 0.0
                    class_mask[sel] = cls_id
                    conf_mask[sel] = score

        # Project latest lidar to camera
        cam = (xyz.astype(np.float64) @ self.R.T) + self.t
        zc = cam[:, 2]
        front = zc > 1e-6
        idx_front = np.where(front)[0]

        painted = np.zeros((xyz.shape[0],), dtype=np.uint8)
        class_id = np.zeros((xyz.shape[0],), dtype=np.uint16)
        class_conf = np.zeros((xyz.shape[0],), dtype=np.float32)
        track_id = np.zeros((xyz.shape[0],), dtype=np.uint32)

        img_pts = np.empty((0, 2), dtype=np.int32)
        src_indices = np.empty((0,), dtype=np.int64)
        if idx_front.size > 0:
            c = cam[idx_front]
            u = (self.K[0, 0] * c[:, 0] / c[:, 2] + self.K[0, 2]).round().astype(np.int32)
            v = (self.K[1, 1] * c[:, 1] / c[:, 2] + self.K[1, 2]).round().astype(np.int32)
            inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if np.any(inb):
                uu = u[inb]
                vv = v[inb]
                src = idx_front[inb]
                cid = class_mask[vv, uu]
                sel = cid > 0
                if np.any(sel):
                    sidx = src[sel]
                    class_id[sidx] = cid[sel]
                    class_conf[sidx] = conf_mask[vv[sel], uu[sel]]
                    painted[sidx] = 1
                img_pts = np.stack((uu, vv), axis=1)
                src_indices = src

        cloud_out = self._build_painted_cloud(cloud_header, xyz, intensity, class_id, class_conf, track_id, painted)
        self.cloud_pub.publish(cloud_out)

        if self.publish_projected_image:
            overlay = frame.copy()
            if self.overlay_segmentation_mask and np.any(class_mask):
                alpha = float(np.clip(self.overlay_segmentation_alpha, 0.0, 1.0))
                seed = class_mask.astype(np.uint32) * 2654435761
                color = np.stack(
                    [(seed & 0xFF), ((seed >> 8) & 0xFF), ((seed >> 16) & 0xFF)],
                    axis=-1,
                ).astype(np.uint8)
                sel = class_mask > 0
                blended = (
                    overlay[sel].astype(np.float32) * (1.0 - alpha)
                    + color[sel].astype(np.float32) * alpha
                ).astype(np.uint8)
                overlay[sel] = blended
            for i in range(img_pts.shape[0]):
                u, v = int(img_pts[i, 0]), int(img_pts[i, 1])
                src = int(src_indices[i])
                color = (0, 255, 0) if painted[src] == 1 else (0, 200, 255)
                cv2.circle(overlay, (u, v), 1, color, cv2.FILLED)
            out_img = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            out_img.header = msg.header
            self.proj_img_pub.publish(out_img)

        if self.profile_timing and time.time() - self._last_log >= 1.0:
            infer_ms = (time.perf_counter() - t0) * 1000.0
            ratio = float((painted == 1).sum()) / max(1, painted.size) * 100.0
            self.get_logger().info(
                f"infer={infer_ms:.1f}ms painted={int((painted==1).sum())}/{painted.size} ({ratio:.2f}%)"
            )
            self._last_log = time.time()

    def _build_painted_cloud(self, header, xyz, intensity, class_id, class_conf, track_id, painted):
        n = xyz.shape[0]
        dt = np.dtype(
            {
                "names": ["x", "y", "z", "intensity", "class_id", "class_conf", "track_id", "painted"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f4", "<u4", "u1"],
                "offsets": [0, 4, 8, 12, 16, 18, 22, 26],
                "itemsize": 27,
            }
        )
        arr = np.zeros((n,), dtype=dt)
        arr["x"] = xyz[:, 0]
        arr["y"] = xyz[:, 1]
        arr["z"] = xyz[:, 2]
        arr["intensity"] = intensity
        arr["class_id"] = class_id
        arr["class_conf"] = class_conf
        arr["track_id"] = track_id
        arr["painted"] = painted

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(n)
        msg.is_bigendian = False
        msg.is_dense = False
        msg.point_step = 27
        msg.row_step = msg.point_step * msg.width
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=16, datatype=PointField.UINT16, count=1),
            PointField(name="class_conf", offset=18, datatype=PointField.FLOAT32, count=1),
            PointField(name="track_id", offset=22, datatype=PointField.UINT32, count=1),
            PointField(name="painted", offset=26, datatype=PointField.UINT8, count=1),
        ]
        msg.data = arr.tobytes()
        return msg

def main() -> None:
    rclpy.init()
    node = FusionSegPaintNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
