"""Pipeline orchestrator: ties detector, tracker, writer, and renderer together."""

import logging
import os
from typing import Dict, List

import cv2
import numpy as np

from .config_loader import ConfigLoader
from .detector import YOLODetector
from .models import Config, Track
from .renderer import FrameRenderer
from .results_writer import ResultsWriter
from .sequence_loader import load_sequences
from .trackers.base import BaseTracker

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs the full tracking pipeline for a single tracker."""

    def __init__(self, cfg: Config, tracker: BaseTracker) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.detector = YOLODetector(cfg.fine_tuned_checkpoint)
        self.writer = ResultsWriter()
        self.renderer = FrameRenderer()

    def run(self, sequence_name: str, image_dir: str) -> None:
        """Process a single sequence end-to-end.

        Args:
            sequence_name: Name of the sequence (used for output file naming).
            image_dir: Directory containing the sequence's image frames.
        """
        # Load frames for this sequence
        sequences = load_sequences(image_dir)
        frame_paths = sequences.get(sequence_name)
        if not frame_paths:
            logger.warning("No frames found for sequence %s in %s", sequence_name, image_dir)
            return

        logger.info("Processing sequence %s (%d frames)", sequence_name, len(frame_paths))

        # Reset tracker state
        self.tracker.reset()

        frames: List[np.ndarray] = []
        tracks_by_frame: Dict[int, List[Track]] = {}

        for frame_idx, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning("Failed to decode frame: %s", frame_path)
                continue

            frames.append(frame)

            # Detect
            detections = self.detector.infer(
                frame,
                conf_thresh=self.cfg.conf_threshold,
                nms_iou=self.cfg.nms_iou_threshold,
            )

            # Track
            active_tracks = self.tracker.update(detections, frame)
            tracks_by_frame[frame_idx] = active_tracks

        # Determine tracker name from class name
        tracker_name = type(self.tracker).__name__.lower().replace("tracker", "")

        # Write results
        results_path = os.path.join(
            self.cfg.output_dir, "results", tracker_name, f"{sequence_name}.txt"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        self.writer.write(tracks_by_frame, results_path)
        logger.info("Results written to %s", results_path)

        # Render annotated video
        video_path = os.path.join(
            self.cfg.output_dir, "annotated", tracker_name, f"{sequence_name}.mp4"
        )
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        self.renderer.render_sequence(frames, tracks_by_frame, video_path, sequence_name=sequence_name)
        logger.info("Video written to %s", video_path)
