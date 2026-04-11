"""Minimal SORT tracker implemented inline using filterpy and scipy."""

import logging
from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..models import Detection, Track
from .base import BaseTracker

logger = logging.getLogger(__name__)


def _iou(bb_a: np.ndarray, bb_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(bb_a[0], bb_b[0])
    y1 = max(bb_a[1], bb_b[1])
    x2 = min(bb_a[2], bb_b[2])
    y2 = min(bb_a[3], bb_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    area_b = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=float)


def _xyxy_to_z(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to Kalman state [cx, cy, s, r] where s=area, r=aspect."""
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    cx = xyxy[0] + w / 2.0
    cy = xyxy[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, s, r, ...] back to [x1,y1,x2,y2]."""
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w > 0 else 0.0
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).flatten()


class _KalmanBoxTracker:
    """Single-object Kalman filter tracker (SORT-style)."""

    _count = 0

    def __init__(self, xyxy: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [cx, cy, s, r, vx, vy, vs]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _xyxy_to_z(xyxy)

        _KalmanBoxTracker._count += 1
        self.id = _KalmanBoxTracker._count
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_xyxy(self.kf.x)

    def update(self, xyxy: np.ndarray) -> None:
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(_xyxy_to_z(xyxy))

    def get_state(self) -> np.ndarray:
        return _x_to_xyxy(self.kf.x)


def _associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float,
):
    """Hungarian matching. Returns (matches, unmatched_dets, unmatched_trks)."""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = _iou(det, trk)

    # Hungarian algorithm (maximise IoU = minimise negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches_arr = np.empty((0, 2), dtype=int)
    else:
        matches_arr = np.concatenate(matches, axis=0)

    return matches_arr, np.array(unmatched_dets), np.array(unmatched_trks)


class SORTTracker(BaseTracker):
    """SORT: Simple Online and Realtime Tracking."""

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[_KalmanBoxTracker] = []
        self._frame_count = 0

    def reset(self) -> None:
        """Reinitialise state between sequences."""
        self._trackers = []
        self._frame_count = 0
        _KalmanBoxTracker._count = 0

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with current frame detections.

        Args:
            detections: List of Detection objects.
            frame: Current frame (unused by SORT but required by interface).

        Returns:
            List of active Track objects.
        """
        self._frame_count += 1

        # Convert detections to [x1,y1,x2,y2]
        dets_xyxy = np.array([
            _xywh_to_xyxy(d.x, d.y, d.w, d.h) for d in detections
        ], dtype=float) if detections else np.empty((0, 4), dtype=float)

        # Predict new locations for existing trackers
        trks_xyxy = np.zeros((len(self._trackers), 4), dtype=float)
        to_del = []
        for t, trk in enumerate(self._trackers):
            pos = trk.predict()
            trks_xyxy[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self._trackers.pop(t)
        trks_xyxy = np.ma.compress_rows(np.ma.masked_invalid(trks_xyxy))

        matched, unmatched_dets, unmatched_trks = _associate_detections_to_trackers(
            dets_xyxy, trks_xyxy, self.iou_threshold
        )

        # Update matched trackers
        for m in matched:
            self._trackers[m[1]].update(dets_xyxy[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self._trackers.append(_KalmanBoxTracker(dets_xyxy[i]))

        # Remove dead trackers
        self._trackers = [
            t for t in self._trackers if t.time_since_update <= self.max_age
        ]

        # Return active tracks
        tracks: List[Track] = []
        for trk in self._trackers:
            if trk.time_since_update > 0:
                continue
            if trk.hit_streak < self.min_hits and self._frame_count > self.min_hits:
                continue
            xyxy = trk.get_state()
            x = float(xyxy[0])
            y = float(xyxy[1])
            w = float(xyxy[2] - xyxy[0])
            h = float(xyxy[3] - xyxy[1])
            tracks.append(Track(track_id=trk.id, x=x, y=y, w=w, h=h))

        return tracks
