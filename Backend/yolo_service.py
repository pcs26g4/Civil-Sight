"""
YOLOv5 Detection Service
Handles model loading and inference for object detection
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------------------
# Project paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # mdms/
BACKEND_ROOT = Path(__file__).resolve().parents[0]  # Backend/

YOLO_ROOT = BACKEND_ROOT / "yolov5"
WEIGHTS_DIR = PROJECT_ROOT / "weights"



# ---------------------------
# Lazy imports
# ---------------------------
def _import_dependencies():
    try:
        import torch
        import cv2
        import numpy as np

        # Ensure YOLOv5 path is first
        yolo_path = str(YOLO_ROOT)
        if yolo_path not in sys.path:
            sys.path.insert(0, yolo_path)

        from models.common import DetectMultiBackend
        from utils.general import check_img_size, non_max_suppression, scale_boxes, LOGGER
        from utils.augmentations import letterbox
        from utils.torch_utils import select_device

        return {
            "torch": torch,
            "cv2": cv2,
            "np": np,
            "DetectMultiBackend": DetectMultiBackend,
            "check_img_size": check_img_size,
            "non_max_suppression": non_max_suppression,
            "scale_boxes": scale_boxes,
            "LOGGER": LOGGER,
            "letterbox": letterbox,
            "select_device": select_device,
        }

    except ImportError as e:
        raise ImportError(
            "YOLOv5 dependencies missing.\n"
            "Run:\n"
            "pip install torch torchvision opencv-python numpy\n"
            "git clone https://github.com/ultralytics/yolov5\n"
            f"\nError: {e}"
        )


# ---------------------------
# YOLOv5 Service
# ---------------------------
class YOLOv5Service:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "",
        img_size: int = 640,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
    ):
        deps = _import_dependencies()

        self.torch = deps["torch"]
        self.cv2 = deps["cv2"]
        self.np = deps["np"]
        self.DetectMultiBackend = deps["DetectMultiBackend"]
        self.check_img_size = deps["check_img_size"]
        self.non_max_suppression = deps["non_max_suppression"]
        self.scale_boxes = deps["scale_boxes"]
        self.LOGGER = deps["LOGGER"]
        self.letterbox = deps["letterbox"]
        self.select_device = deps["select_device"]

        self.device = self.select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # ---------------------------
        # Custom weights ONLY (Supabase → local)
        # ---------------------------
        if weights_path:
            self.weights_path = Path(weights_path)
        else:
            pt = WEIGHTS_DIR / "best.pt"
            onnx = WEIGHTS_DIR / "best.onnx"

            if pt.exists():
                self.weights_path = pt
            elif onnx.exists():
                self.weights_path = onnx
            else:
                raise FileNotFoundError(
                    "Model weights not found.\n"
                    "Expected best.pt or best.onnx in mdms/weights/\n"
                    "Make sure download_weights.py ran successfully."
                )

        # ---------------------------
        # Load YOLOv5 model
        # ---------------------------
        self.model = self.DetectMultiBackend(
            str(self.weights_path),
            device=self.device,
            dnn=False,
            data=None,
            fp16=self.device.type != "cpu",
        )

        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

        self.img_size = self.check_img_size(img_size, s=self.stride)

        # Warmup
        self.model.warmup(
            imgsz=(1, 3, self.img_size, self.img_size)
        )

        self.LOGGER.info(f"YOLOv5 loaded from {self.weights_path}")
        self.LOGGER.info(f"Device: {self.device}")
        self.LOGGER.info(f"Classes: {self.names}")

    # ---------------------------
    # Detect from numpy image
    # ---------------------------
    def detect_image(
        self,
        im0,
        save_annotated: bool = True,
    ) -> Tuple[List[dict], any]:

        if im0 is None:
            raise ValueError("Input image is None")

        # Preprocess
        im = self.letterbox(im0, self.img_size, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = self.np.ascontiguousarray(im)

        im_tensor = self.torch.from_numpy(im).to(self.device).float()
        im_tensor /= 255.0
        im_tensor = im_tensor.unsqueeze(0)

        # Inference
        pred = self.model(im_tensor)
        pred = self.non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            max_det=1000,
        )

        detections = []
        annotated = im0.copy()

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = self.scale_boxes(
                    im_tensor.shape[2:], det[:, :4], im0.shape
                ).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_name = self.names[int(cls)]

                    detections.append({
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    })

                    if save_annotated:
                        label = f"{class_name} {conf:.2f}"
                        self.cv2.rectangle(
                            annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
                        )
                        self.cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 10),
                            self.cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

        return detections, annotated

    # ---------------------------
    # Detect from bytes
    # ---------------------------
    def detect_from_bytes(
        self,
        image_bytes: bytes,
        save_annotated: bool = True,
    ) -> Tuple[List[dict], any]:

        arr = self.np.frombuffer(image_bytes, self.np.uint8)
        img = self.cv2.imdecode(arr, self.cv2.IMREAD_COLOR)
        return self.detect_image(img, save_annotated)


# ---------------------------
# Singleton
# ---------------------------
_yolo_service: Optional[YOLOv5Service] = None


def get_yolo_service() -> YOLOv5Service:
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOv5Service()
    return _yolo_service
