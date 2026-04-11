"""YOLOv8/YOLO11 detector wrapper (inference only)."""

import logging
from typing import List

import numpy as np
from ultralytics import YOLO

from .models import Detection

logger = logging.getLogger(__name__)


class YOLODetector:
    """Wraps an Ultralytics YOLO model for per-frame inference."""

    def __init__(self, checkpoint_path: str) -> None:
        logger.info("Loading YOLO model from %s", checkpoint_path)
        self.model = YOLO(checkpoint_path)

    def infer(self, frame: np.ndarray, conf_thresh: float, nms_iou: float) -> List[Detection]:
        """Run inference on a single frame.

        Args:
            frame: BGR image as np.ndarray (H, W, 3).
            conf_thresh: Minimum confidence threshold.
            nms_iou: IoU threshold for NMS.

        Returns:
            List of Detection objects in pixel space (top-left origin).
        """
        results = self.model.predict(
            source=frame,
            conf=conf_thresh,
            iou=nms_iou,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # xywh gives (cx, cy, w, h) — convert to top-left
                xywh = box.xywh[0].cpu().numpy()
                cx, cy, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                x = cx - w / 2.0
                y = cy - h / 2.0
                conf = float(box.conf[0].cpu().numpy())
                detections.append(Detection(x=x, y=y, w=w, h=h, confidence=conf))

        return detections
