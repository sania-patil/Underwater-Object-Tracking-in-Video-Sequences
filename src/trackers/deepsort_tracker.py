"""DeepSORT tracker using deep_sort_realtime package."""

import logging
from typing import List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from ..models import Detection, Track
from .base import BaseTracker

logger = logging.getLogger(__name__)


class DeepSORTTracker(BaseTracker):
    """Tracker using DeepSORT with appearance re-identification."""

    def __init__(
        self,
        max_age: int = 30,
        nn_budget: int = 100,
        max_cosine_distance: float = 0.4,
        reid_model_path: str = "",
    ) -> None:
        self.max_age = max_age
        self.nn_budget = nn_budget
        self.max_cosine_distance = max_cosine_distance
        self.reid_model_path = reid_model_path
        self._tracker = self._build_tracker()

    def _build_tracker(self) -> DeepSort:
        kwargs = dict(
            max_age=self.max_age,
            nn_budget=self.nn_budget,
            max_cosine_distance=self.max_cosine_distance,
        )
        if self.reid_model_path:
            kwargs["embedder_model_name"] = self.reid_model_path
        return DeepSort(**kwargs)

    def reset(self) -> None:
        """Reinitialise tracker state between sequences."""
        self._tracker = self._build_tracker()

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with current frame detections.

        Args:
            detections: List of Detection objects.
            frame: Current BGR frame as np.ndarray (H, W, 3).

        Returns:
            List of active Track objects.
        """
        # deep_sort_realtime expects list of ([left, top, w, h], confidence, class_id)
        raw_detections = [
            ([d.x, d.y, d.w, d.h], d.confidence, 0)
            for d in detections
        ]

        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

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
