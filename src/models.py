# Data models for underwater object tracking

from dataclasses import dataclass, field


# --- Exceptions ---

class TrackingPipelineError(Exception):
    """Base exception for all tracking pipeline errors."""


class ConfigError(TrackingPipelineError):
    """Raised when configuration is invalid or missing required keys."""


class ModelNotFoundError(TrackingPipelineError):
    """Raised when a required model file cannot be found."""


# --- Dataclasses ---

@dataclass
class Detection:
    """A single object detection in pixel space (top-left origin)."""
    x: float
    y: float
    w: float
    h: float
    confidence: float


@dataclass
class Track:
    """An active tracked object."""
    track_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class SequenceMetrics:
    """Evaluation metrics for a single sequence."""
    sequence_name: str
    mota: float
    motp: float
    id_switches: int
    num_frames: int


@dataclass
class AggregateMetrics:
    """Aggregated evaluation metrics across all sequences for a tracker."""
    tracker_name: str
    mota: float
    motp: float
    total_id_switches: int
    sequence_metrics: list = field(default_factory=list)


@dataclass
class Config:
    """Pipeline configuration loaded from config.yaml."""
    # Required fields (no defaults)
    fine_tuned_checkpoint: str
    output_dir: str
    dataset_yaml: str

    # Detector
    pretrained_checkpoint: str = "yolov8s.pt"
    train_epochs: int = 50

    # Inference
    conf_threshold: float = 0.25
    nms_iou_threshold: float = 0.45

    # SORT
    sort_max_age: int = 1
    sort_min_hits: int = 3
    sort_iou_threshold: float = 0.3

    # DeepSORT
    deepsort_max_age: int = 30
    deepsort_nn_budget: int = 100
    deepsort_max_cosine_distance: float = 0.4
    deepsort_reid_model: str = ""

    # FairMOT
    fairmot_model_path: str = ""
    fairmot_conf_threshold: float = 0.4

    # Evaluation
    eval_iou_threshold: float = 0.5
    eval_output_path: str = "runs/tracking/evaluation"
