import glob
import logging
import os

logger = logging.getLogger(__name__)


def load_sequences(image_dir: str) -> dict[str, list[str]]:
    """Scan image_dir for *.jpg files, group by sequence name, sort by frame number."""
    sequences: dict[str, list[tuple[int, str]]] = {}

    for path in glob.glob(os.path.join(image_dir, "*.jpg")):
        filename = os.path.splitext(os.path.basename(path))[0]
        if "_" not in filename:
            logger.warning("Skipping file with unparseable name: %s", path)
            continue
        seq_name, frame_str = filename.rsplit("_", 1)
        try:
            frame_num = int(frame_str)
        except ValueError:
            logger.warning("Skipping file with non-integer frame number: %s", path)
            continue
        sequences.setdefault(seq_name, []).append((frame_num, path))

    return {
        seq: [p for _, p in sorted(frames)]
        for seq, frames in sequences.items()
    }
