"""Frame renderer: draws bounding boxes, track IDs and object labels, encodes to MP4."""

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from .models import Track

logger = logging.getLogger(__name__)

# Map sequence name -> human-readable object label
SEQUENCE_LABELS = {
    "BlueFish2":   "Blue Fish",
    "BoySwimming": "Boy Swimming",
    "Dolphin2":    "Dolphin",
    "Fisherman":   "Fisherman",
    "HoverFish2":  "Hover Fish",
    "SeaDiver":    "Sea Diver",
    "SeaTurtle2":  "Sea Turtle",
    "SeaTurtle3":  "Sea Turtle",
}

# Colour palette for track IDs (BGR)
_COLOURS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


class FrameRenderer:
    """Renders annotated video sequences to MP4."""

    def render_sequence(
        self,
        frames: List[np.ndarray],
        tracks_by_frame: Dict[int, List[Track]],
        output_path: str,
        sequence_name: Optional[str] = None,
    ) -> None:
        """Draw bounding boxes, track IDs and object label on frames, write to MP4.

        Args:
            frames: List of BGR frames (H, W, 3).
            tracks_by_frame: Mapping of 1-based frame index to Track list.
            output_path: Destination .mp4 file path.
            sequence_name: Optional sequence name used to look up object label.
        """
        if not frames:
            logger.warning("No frames to render; skipping %s", output_path)
            return

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Resolve object label from sequence name
        object_label = SEQUENCE_LABELS.get(sequence_name, "Underwater Object") if sequence_name else "Underwater Object"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))

        for frame_idx, frame in enumerate(frames, start=1):
            annotated = frame.copy()
            for track in tracks_by_frame.get(frame_idx, []):
                colour = _COLOURS[track.track_id % len(_COLOURS)]
                x1 = int(track.x)
                y1 = int(track.y)
                x2 = int(track.x + track.w)
                y2 = int(track.y + track.h)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                # Show "Label #ID" e.g. "Sea Diver #1"
                label_text = f"{object_label} #{track.track_id}"
                cv2.putText(
                    annotated,
                    label_text,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                )
            writer.write(annotated)

        writer.release()
        logger.info("Rendered %d frames to %s", len(frames), output_path)
