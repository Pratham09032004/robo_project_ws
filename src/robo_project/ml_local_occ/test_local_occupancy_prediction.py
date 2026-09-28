#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from ml_local_occ.model import SimpleLocalOccNet
from ml_local_occ.het_occupancy_model import ModelConfig, OccupancyModel


class MLLocalOccupancyNode(Node):
    def __init__(self):
        super().__init__("ml_local_occupancy_node")

        # ml_runs/ is installed to <share>/robo_project/ml_runs by setup.py;
        # PKG_DIR/ml_runs is the source tree when running the script directly.
        run_dirs = [PKG_DIR / "ml_runs"]
        try:
            from ament_index_python.packages import get_package_share_directory
            run_dirs.insert(0, Path(get_package_share_directory("robo_project")) / "ml_runs")
        except Exception:
            pass
        candidates = [d / name for name in ("het_best.pt", "local_occ_rgb4_3200samples_new.pth") for d in run_dirs]
        default_model = next((c for c in candidates if c.exists()), candidates[-1])

        self.declare_parameter("model_path", str(default_model))
        self.declare_parameter("output_topic", "/local_occupancy_ml")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("publish_hz", 5.0)

        self.model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        self.output_topic = self.get_parameter("output_topic").value
        self.threshold = float(self.get_parameter("threshold").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_kind = None
        self.required_cams = ["front"]
        self.model = self._load_model(self.model_path)

        self.latest = {
            "front": None,
            "left": None,
            "right": None,
            "rear": None,
        }

        self.create_subscription(Image, "/camera/front/image_raw", lambda msg: self.image_cb("front", msg), 10)
        self.create_subscription(Image, "/camera/left/image_raw", lambda msg: self.image_cb("left", msg), 10)
        self.create_subscription(Image, "/camera/right/image_raw", lambda msg: self.image_cb("right", msg), 10)
        self.create_subscription(Image, "/camera/back/image_raw", lambda msg: self.image_cb("rear", msg), 10)

        self.pub = self.create_publisher(Image, self.output_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_cb)

        self.pub_count = 0

        self.get_logger().info(f"Loaded ML model: {self.model_path}")
        self.get_logger().info(f"Model kind: {self.model_kind}, device={self.device}")
        self.get_logger().info(f"Required cameras: {self.required_cams}")
        self.get_logger().info(f"Publishing ML occupancy on: {self.output_topic}")

    def _torch_load(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _channels_from_config(self, config):
        mode = str(config.get("input_mode", "rgb")).lower()
        if "4cam" in mode or "four" in mode or mode in {"rgb4", "rgb_4cam", "multi_rgb"}:
            return 12
        if "rgbd" in mode or "depth" in mode:
            return 4
        return int(config.get("in_channels", 3))

    def _load_model(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        ckpt = self._torch_load(path)

        # Het/release checkpoint: {"model_state", "config", ...}
        if isinstance(ckpt, dict) and "model_state" in ckpt and "config" in ckpt:
            config = ckpt["config"]
            in_channels = self._channels_from_config(config)

            cfg = ModelConfig(
                in_channels=in_channels,
                num_classes=int(config.get("num_classes", 3)),
                grid_size=int(config.get("grid_size", 64)),
                pretrained=False,
                freeze_encoder=False,
            )

            model = OccupancyModel(cfg).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            self.model_kind = "het_resnet18_occupancy"
            self.required_cams = ["front"] if in_channels == 3 else ["front", "left", "right", "rear"]
            return model

        # Old simple conv checkpoint fallback
        model = SimpleLocalOccNet().to(self.device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        self.model_kind = "old_simple_conv"
        self.required_cams = ["front", "left", "right", "rear"]
        return model

    def image_msg_to_rgb(self, msg):
        h, w = msg.height, msg.width
        enc = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ["rgb8", "bgr8"]:
            arr = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[:, :, ::-1]
            return arr.copy()

        if enc in ["rgba8", "bgra8"]:
            arr = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
            if enc == "bgra8":
                arr = arr[:, :, [2, 1, 0, 3]]
            return arr[:, :, :3].copy()

        raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

    def image_cb(self, cam, msg):
        try:
            self.latest[cam] = self.image_msg_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"{cam} image read failed: {e}")

    def preprocess_old_conv(self, arr):
        img = PILImage.fromarray(arr).convert("RGB")
        img = img.resize((64, 64))
        x = np.asarray(img).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)

    def preprocess_het_rgb(self, arr):
        img = PILImage.fromarray(arr).convert("RGB")
        img = img.resize((224, 224))
        x = np.asarray(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std

        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x.astype(np.float32))

    def timer_cb(self):
        if any(self.latest[k] is None for k in self.required_cams):
            return

        try:
            if self.model_kind == "het_resnet18_occupancy":
                if self.required_cams == ["front"]:
                    x = self.preprocess_het_rgb(self.latest["front"]).unsqueeze(0).to(self.device)
                else:
                    x = torch.cat([
                        self.preprocess_het_rgb(self.latest["front"]),
                        self.preprocess_het_rgb(self.latest["left"]),
                        self.preprocess_het_rgb(self.latest["right"]),
                        self.preprocess_het_rgb(self.latest["rear"]),
                    ], dim=0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(x)
                    class_map = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy()

                # Het class convention:
                # 0 = unknown, 1 = occupied, 2 = free
                # PF mono8 convention:
                # 127 = unknown, 0 = occupied, 255 = free
                grid = np.empty(class_map.shape, dtype=np.uint8)
                grid[class_map == 0] = 127
                grid[class_map == 1] = 0
                grid[class_map == 2] = 255

            else:
                x = torch.cat([
                    self.preprocess_old_conv(self.latest["front"]),
                    self.preprocess_old_conv(self.latest["left"]),
                    self.preprocess_old_conv(self.latest["right"]),
                    self.preprocess_old_conv(self.latest["rear"]),
                ], dim=0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(x)
                    prob = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)

                grid = (prob > self.threshold).astype(np.uint8) * 255

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.height = int(grid.shape[0])
            msg.width = int(grid.shape[1])
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = int(grid.shape[1])
            msg.data = grid.tobytes()

            self.pub.publish(msg)
            self.pub_count += 1

            if self.pub_count == 1 or self.pub_count % 10 == 0:
                self.get_logger().info(
                    f"Published /local_occupancy_ml #{self.pub_count}: "
                    f"model={self.model_kind}, shape={grid.shape}, min={grid.min()}, max={grid.max()}"
                )

        except Exception as e:
            self.get_logger().error(f"ML inference failed: {e}")


def main():
    rclpy.init()
    node = MLLocalOccupancyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
