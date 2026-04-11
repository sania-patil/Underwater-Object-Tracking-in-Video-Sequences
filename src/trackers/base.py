"""Abstract base class for all trackers."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..models import Detection, Track


class BaseTracker(ABC):
    """Abstract tracker interface."""

    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Process detections for the current frame and return active tracks.

        Args:
            detections: List of Detection objects for the current frame.
            frame: Current frame as np.ndarray (H, W, 3).

        Returns:
            List of active Track objects.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reinitialise tracker state between sequences."""
