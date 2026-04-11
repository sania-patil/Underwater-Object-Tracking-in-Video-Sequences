"""
FairMOT-style tracker: joint detection + Re-ID using YOLO detector
with appearance embeddings extracted via a lightweight CNN.
Uses the same YOLO detector but adds Re-ID features for association.
"""

import logging
from typing import List

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from ..models import Detection, ModelNotFoundError, Track
from .base import BaseTracker

logger = logging.getLogger(__name__)


class FairMOTTracker(BaseTracker):
    """
    FairMOT-style tracker that combines detection with Re-ID embeddings.
    
    Unlike DeepSORT (separate detector + Re-ID), FairMOT jointly learns
    detection and Re-ID. We approximate this by using our fine-tuned YOLO
    detector with DeepSort's appearance embeddings at a lower cosine distance
    threshold (tighter appearance matching), mimicking FairMOT's joint approach.
    
    Key differences from DeepSORT config:
    - Lower max_cosine_distance (0.2 vs 0.4) — stricter appearance matching
    - Lower max_age (20 vs 30) — faster track cleanup
    - Higher nn_budget (200 vs 100) — more appearance features stored
    """

    def __init__(
        self,
        model_path: str = "",
        conf_thresh: float = 0.4,
        max_age: int = 20,
        nn_budget: int = 200,
        max_cosine_distance: float = 0.2,
    ) -> None:
        # model_path is optional — we use YOLO detector from pipeline
        # Raise ModelNotFoundError only if explicitly provided but missing
        if model_path and not __import__("os").path.exists(model_path):
            raise ModelNotFoundError(
                f"FairMOT model weights not found at: {model_path}"
            )

        self.conf_thresh = conf_thresh
        self.max_age = max_age
        self.nn_budget = nn_budget
        self.max_cosine_distance = max_cosine_distance
        self._tracker = self._build_tracker()
        logger.info(
            "FairMOTTracker initialised (max_age=%d, max_cosine_dist=%.2f, nn_budget=%d)",
            max_age, max_cosine_distance, nn_budget,
        )

    def _build_tracker(self) -> DeepSort:
        return DeepSort(
            max_age=self.max_age,
            nn_budget=self.nn_budget,
            max_cosine_distance=self.max_cosine_distance,
        )

    def reset(self) -> None:
        self._tracker = self._build_tracker()

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracker. Detections come from the shared YOLO detector.
        FairMOT-style: uses tight appearance matching to reduce ID switches.
        """
        # Filter by confidence threshold
        filtered = [d for d in detections if d.confidence >= self.conf_thresh]

        raw = [
            ([d.x, d.y, d.w, d.h], d.confidence, 0)
            for d in filtered
        ]

        tracks = self._tracker.update_tracks(raw, frame=frame)

        result: List[Track] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x = float(ltrb[0])
            y = float(ltrb[1])
            w = float(ltrb[2] - ltrb[0])
            h = float(ltrb[3] - ltrb[1])
            result.append(Track(track_id=int(track.track_id), x=x, y=y, w=w, h=h))

        return result
