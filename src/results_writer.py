"""MOTChallenge format results writer."""

import os
from typing import Dict, List

from .models import Track


class ResultsWriter:
    """Writes tracking results in MOTChallenge format."""

    def write(self, tracks_by_frame: Dict[int, List[Track]], output_path: str) -> None:
        """Write tracks to a MOTChallenge format file.

        Format per line: frame,id,x,y,w,h,conf,-1,-1,-1

        Args:
            tracks_by_frame: Mapping of frame index (1-based) to list of Track objects.
            output_path: Path to the output .txt file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        lines = []
        for frame_idx in sorted(tracks_by_frame.keys()):
            for track in tracks_by_frame[frame_idx]:
                line = (
                    f"{frame_idx},{track.track_id},"
                    f"{track.x:.3f},{track.y:.3f},"
                    f"{track.w:.3f},{track.h:.3f},"
                    f"1,-1,-1,-1"
                )
                lines.append(line)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
