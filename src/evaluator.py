"""Tracker evaluation using motmetrics."""

import csv
import json
import logging
import os
from typing import Dict, List, Optional

import cv2
import motmetrics as mm
import numpy as np

from .models import AggregateMetrics, SequenceMetrics

logger = logging.getLogger(__name__)


def _read_mot_results(results_path: str) -> Dict[int, List[List[float]]]:
    """Read MOTChallenge results file.

    Returns dict mapping frame_id -> list of [id, x, y, w, h].
    """
    data: Dict[int, List[List[float]]] = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data.setdefault(frame, []).append([track_id, x, y, w, h])
    return data


def _read_yolo_gt(gt_dir: str, seq_name: str) -> Dict[int, List[List[float]]]:
    """Read YOLO-format ground truth labels for a sequence.

    Returns dict mapping frame_id -> list of [x, y, w, h] in pixel space.
    We need image dimensions to convert normalised coords.
    """
    import glob

    gt_data: Dict[int, List[List[float]]] = {}

    # Find all label files for this sequence
    pattern = os.path.join(gt_dir, f"{seq_name}_*.txt")
    label_files = glob.glob(pattern)

    for label_path in label_files:
        basename = os.path.splitext(os.path.basename(label_path))[0]
        # Extract frame number
        parts = basename.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            frame_num = int(parts[1])
        except ValueError:
            continue

        # Find corresponding image to get dimensions
        img_dir = gt_dir.replace("labels", "images")
        img_path = os.path.join(img_dir, f"{basename}.jpg")
        if not os.path.exists(img_path):
            # Try without knowing exact extension
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(img_dir, f"{basename}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        boxes = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = line.split()
                if len(vals) < 5:
                    continue
                # YOLO: class cx cy w h (normalised)
                cx = float(vals[1]) * img_w
                cy = float(vals[2]) * img_h
                bw = float(vals[3]) * img_w
                bh = float(vals[4]) * img_h
                x = cx - bw / 2.0
                y = cy - bh / 2.0
                boxes.append([x, y, bw, bh])

        gt_data[frame_num] = boxes

    return gt_data


def _iou_matrix(gt_boxes: list, pred_boxes: list, max_iou: float) -> np.ndarray:
    """Compute IoU distance matrix without using motmetrics' broken iou_matrix."""
    gt = np.asarray(gt_boxes, dtype=np.float64)   # (N, 4) xywh
    pr = np.asarray(pred_boxes, dtype=np.float64)  # (M, 4) xywh

    # Convert xywh -> xyxy
    gt_x2 = gt[:, 0] + gt[:, 2]
    gt_y2 = gt[:, 1] + gt[:, 3]
    pr_x2 = pr[:, 0] + pr[:, 2]
    pr_y2 = pr[:, 1] + pr[:, 3]

    dist = np.ones((len(gt), len(pr)), dtype=np.float64)
    for i in range(len(gt)):
        for j in range(len(pr)):
            ix1 = max(gt[i, 0], pr[j, 0])
            iy1 = max(gt[i, 1], pr[j, 1])
            ix2 = min(gt_x2[i], pr_x2[j])
            iy2 = min(gt_y2[i], pr_y2[j])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_gt = gt[i, 2] * gt[i, 3]
            area_pr = pr[j, 2] * pr[j, 3]
            union = area_gt + area_pr - inter
            iou = inter / union if union > 0 else 0.0
            dist[i, j] = 1.0 - iou  # distance = 1 - IoU
    # Set distances above threshold to NaN (no match)
    dist[dist > (1.0 - max_iou)] = np.nan
    return dist


class Evaluator:
    """Evaluates tracker results against YOLO ground truth."""

    def evaluate_tracker(
        self,
        tracker_name: str,
        results_dir: str,
        gt_dir: str,
        iou_threshold: float = 0.5,
    ) -> Dict[str, SequenceMetrics]:
        """Evaluate all sequences for a tracker.

        Args:
            tracker_name: Name of the tracker (for logging).
            results_dir: Directory containing {seq}.txt results files.
            gt_dir: Directory containing YOLO ground-truth label files.
            iou_threshold: IoU threshold for matching.

        Returns:
            Dict mapping sequence name to SequenceMetrics.
        """
        # Discover sequences from GT dir
        import glob

        seq_names = set()
        for label_path in glob.glob(os.path.join(gt_dir, "*.txt")):
            basename = os.path.splitext(os.path.basename(label_path))[0]
            parts = basename.rsplit("_", 1)
            if len(parts) == 2:
                seq_names.add(parts[0])

        metrics: Dict[str, SequenceMetrics] = {}

        for seq_name in sorted(seq_names):
            results_path = os.path.join(results_dir, f"{seq_name}.txt")
            if not os.path.exists(results_path):
                logger.warning(
                    "Results file missing for sequence %s (tracker: %s), skipping",
                    seq_name,
                    tracker_name,
                )
                continue

            seq_metrics = self._evaluate_sequence(
                seq_name, results_path, gt_dir, iou_threshold
            )
            if seq_metrics is not None:
                metrics[seq_name] = seq_metrics

        return metrics

    def _evaluate_sequence(
        self,
        seq_name: str,
        results_path: str,
        gt_dir: str,
        iou_threshold: float,
    ) -> Optional[SequenceMetrics]:
        """Evaluate a single sequence using motmetrics."""
        try:
            results = _read_mot_results(results_path)
            gt = _read_yolo_gt(gt_dir, seq_name)
        except Exception as e:
            logger.warning("Failed to read data for sequence %s: %s", seq_name, e)
            return None

        acc = mm.MOTAccumulator(auto_id=True)

        all_frames = sorted(set(list(gt.keys()) + list(results.keys())))
        if not all_frames:
            return None

        for frame_id in all_frames:
            gt_boxes = gt.get(frame_id, [])
            pred_boxes = results.get(frame_id, [])

            gt_ids = list(range(len(gt_boxes)))
            pred_ids = [int(b[0]) for b in pred_boxes]

            # Compute distance matrix (1 - IoU, capped at 1 for non-overlapping)
            if gt_boxes and pred_boxes:
                dist_matrix = _iou_matrix(
                    [b[:4] for b in gt_boxes],
                    [b[1:5] for b in pred_boxes],
                    max_iou=iou_threshold,
                )
            else:
                dist_matrix = np.empty((len(gt_boxes), len(pred_boxes)))

            acc.update(gt_ids, pred_ids, dist_matrix)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "motp", "num_switches"], name=seq_name)

        mota = float(summary["mota"].iloc[0])
        motp = float(summary["motp"].iloc[0]) if not np.isnan(summary["motp"].iloc[0]) else 0.0
        id_switches = int(summary["num_switches"].iloc[0])
        num_frames = len(all_frames)

        return SequenceMetrics(
            sequence_name=seq_name,
            mota=mota,
            motp=motp,
            id_switches=id_switches,
            num_frames=num_frames,
        )

    def aggregate(self, metrics: Dict[str, SequenceMetrics]) -> AggregateMetrics:
        """Aggregate per-sequence metrics into overall metrics.

        MOTA/MOTP are weighted averages by frame count.
        """
        if not metrics:
            return AggregateMetrics(
                tracker_name="",
                mota=0.0,
                motp=0.0,
                total_id_switches=0,
                sequence_metrics=[],
            )

        total_frames = sum(m.num_frames for m in metrics.values())
        total_id_switches = sum(m.id_switches for m in metrics.values())

        if total_frames > 0:
            weighted_mota = sum(m.mota * m.num_frames for m in metrics.values()) / total_frames
            weighted_motp = sum(m.motp * m.num_frames for m in metrics.values()) / total_frames
        else:
            weighted_mota = 0.0
            weighted_motp = 0.0

        return AggregateMetrics(
            tracker_name="",
            mota=weighted_mota,
            motp=weighted_motp,
            total_id_switches=total_id_switches,
            sequence_metrics=list(metrics.values()),
        )

    def write_report(
        self,
        all_metrics: Dict[str, Dict[str, SequenceMetrics]],
        output_path: str,
    ) -> None:
        """Write evaluation report as CSV and JSON.

        Args:
            all_metrics: Dict mapping tracker_name -> {seq_name -> SequenceMetrics}.
            output_path: Base path (without extension) for output files.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Build rows for CSV
        rows = []
        json_data = {}

        for tracker_name, seq_metrics in all_metrics.items():
            agg = self.aggregate(seq_metrics)
            agg.tracker_name = tracker_name

            tracker_json = {
                "aggregate": {
                    "mota": agg.mota,
                    "motp": agg.motp,
                    "total_id_switches": agg.total_id_switches,
                },
                "sequences": {},
            }

            for seq_name, sm in seq_metrics.items():
                rows.append({
                    "tracker": tracker_name,
                    "sequence": seq_name,
                    "mota": sm.mota,
                    "motp": sm.motp,
                    "id_switches": sm.id_switches,
                    "num_frames": sm.num_frames,
                })
                tracker_json["sequences"][seq_name] = {
                    "mota": sm.mota,
                    "motp": sm.motp,
                    "id_switches": sm.id_switches,
                    "num_frames": sm.num_frames,
                }

            # Add aggregate row
            rows.append({
                "tracker": tracker_name,
                "sequence": "AGGREGATE",
                "mota": agg.mota,
                "motp": agg.motp,
                "id_switches": agg.total_id_switches,
                "num_frames": sum(sm.num_frames for sm in seq_metrics.values()),
            })
            json_data[tracker_name] = tracker_json

        # Write CSV
        csv_path = f"{output_path}.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["tracker", "sequence", "mota", "motp", "id_switches", "num_frames"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Write JSON
        json_path = f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info("Evaluation report written to %s.csv and %s.json", output_path, output_path)
