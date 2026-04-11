# Design Document: Underwater Object Tracking

## Overview

This system implements an end-to-end underwater object tracking pipeline that fine-tunes a YOLOv8 detector on a preprocessed dataset and integrates it with three multi-object trackers: SORT, DeepSORT, and FairMOT. The pipeline processes video sequences (groups of frames sharing a common filename prefix), produces MOTChallenge-format results files, evaluates MOTA/MOTP/ID-switch metrics, and renders annotated MP4 output for visual inspection.

The design is structured around a shared `Detector` abstraction and three interchangeable `Tracker` implementations, all driven by a single YAML config file. Evaluation is handled by a standalone `Evaluator` component that consumes results files and ground-truth annotations.

### Key Design Decisions

- **YOLOv8 via Ultralytics**: The `ultralytics` library provides a clean Python API for fine-tuning and inference, eliminating the need to manage training loops manually.
- **Separate detector for SORT/DeepSORT, joint model for FairMOT**: SORT and DeepSORT consume detections from the fine-tuned YOLOv8 model. FairMOT uses its own backbone that produces detections and Re-ID embeddings jointly.
- **`motmetrics` library for evaluation**: Provides standard MOTA/MOTP/ID-switch computation compatible with MOTChallenge ground-truth format.
- **Sequence grouping from filenames**: Sequences are reconstructed from the `{SequenceName}_{FrameNumber}.jpg` naming convention — no separate sequence manifest is needed.

---

## Architecture

```mermaid
flowchart TD
    subgraph Training
        DS[yolo_dataset/] --> FT[YOLOv8 Fine-Tuning]
        FT --> CKPT[best.pt checkpoint]
    end

    subgraph Inference Pipeline
        CFG[config.yaml] --> PIPE
        CKPT --> DET[YOLOv8 Detector]
        FRAMES[Video Sequence Frames] --> DET
        DET --> DETS[Per-frame Detections]

        DETS --> SORT_T[SORT Tracker]
        DETS --> DS_T[DeepSORT Tracker]
        FRAMES --> FM_T[FairMOT Tracker]

        SORT_T --> RF1[results/sort/{seq}.txt]
        DS_T --> RF2[results/deepsort/{seq}.txt]
        FM_T --> RF3[results/fairmot/{seq}.txt]

        FRAMES --> VIZ[Frame Renderer]
        SORT_T --> VIZ
        DS_T --> VIZ
        FM_T --> VIZ
        VIZ --> MP4[annotated/{tracker}/{seq}.mp4]
    end

    subgraph Evaluation
        RF1 --> EVAL[Evaluator]
        RF2 --> EVAL
        RF3 --> EVAL
        GT[yolo_dataset/labels/val] --> EVAL
        EVAL --> RPT[Evaluation Report CSV + JSON]
    end
```

---

## Components and Interfaces

### 1. ConfigLoader

Reads and validates `config.yaml`. Raises a descriptive `ConfigError` for missing required keys.

```python
class ConfigLoader:
    def load(path: str) -> Config
    def validate(cfg: Config) -> None  # raises ConfigError on missing required keys
```

### 2. SequenceLoader

Groups image files into ordered sequences using the `{SequenceName}_{FrameNumber}` filename convention.

```python
class SequenceLoader:
    def load_sequences(image_dir: str) -> dict[str, list[str]]
    # Returns: {"ArmyDiver1": ["path/ArmyDiver1_1.jpg", ...], ...}
```

### 3. YOLODetector

Wraps the Ultralytics YOLOv8 model for both fine-tuning and per-frame inference.

```python
class YOLODetector:
    def train(cfg: TrainConfig) -> None
    def infer(frame: np.ndarray, conf_thresh: float, nms_iou: float) -> list[Detection]
```

`Detection` is a dataclass: `(x, y, w, h, confidence)` in pixel space.

### 4. Tracker (abstract base)

```python
class BaseTracker(ABC):
    @abstractmethod
    def update(detections: list[Detection], frame: np.ndarray) -> list[Track]
    def reset() -> None
```

`Track` is a dataclass: `(track_id: int, x, y, w, h)`.

### 5. SORTTracker

Implements Kalman filter + Hungarian algorithm association. Uses the `sort` package (or a minimal local implementation).

```python
class SORTTracker(BaseTracker):
    def __init__(max_age: int, min_hits: int, iou_threshold: float)
    def update(detections, frame) -> list[Track]
```

### 6. DeepSORTTracker

Extends SORT with a CNN appearance descriptor. Uses `deep_sort_realtime` or equivalent.

```python
class DeepSORTTracker(BaseTracker):
    def __init__(max_age: int, nn_budget: int, max_cosine_distance: float, reid_model_path: str)
    def update(detections, frame) -> list[Track]
```

### 7. FairMOTTracker

Wraps the FairMOT model for joint detection + Re-ID. Accepts raw frames directly.

```python
class FairMOTTracker(BaseTracker):
    def __init__(model_path: str, conf_thresh: float)
    def update(detections, frame) -> list[Track]
    # Note: detections param is ignored; FairMOT runs its own detection internally
```

### 8. ResultsWriter

Writes tracks to MOTChallenge format: `frame, id, x, y, w, h, conf, -1, -1, -1`.

```python
class ResultsWriter:
    def write(tracks_by_frame: dict[int, list[Track]], output_path: str) -> None
```

### 9. FrameRenderer

Draws bounding boxes and track IDs on frames and encodes to MP4 using OpenCV.

```python
class FrameRenderer:
    def render_sequence(frames: list[np.ndarray], tracks_by_frame: dict[int, list[Track]], output_path: str) -> None
```

### 10. Evaluator

Converts YOLO-format ground truth to MOTChallenge format, then uses `motmetrics` to compute MOTA, MOTP, and ID switches.

```python
class Evaluator:
    def evaluate_tracker(tracker_name: str, results_dir: str, gt_dir: str, iou_threshold: float) -> dict[str, SequenceMetrics]
    def aggregate(metrics: dict[str, SequenceMetrics]) -> AggregateMetrics
    def write_report(all_metrics: dict[str, dict], output_path: str) -> None  # writes CSV + JSON
```

### 11. Pipeline (orchestrator)

Ties all components together for a single tracker run.

```python
class Pipeline:
    def __init__(cfg: Config, tracker: BaseTracker)
    def run(sequence_name: str) -> None
```

---

## Data Models

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Detection:
    x: float          # top-left x, pixel space
    y: float          # top-left y, pixel space
    w: float          # width, pixel space
    h: float          # height, pixel space
    confidence: float

@dataclass
class Track:
    track_id: int
    x: float
    y: float
    w: float
    h: float

@dataclass
class SequenceMetrics:
    sequence_name: str
    mota: float
    motp: float
    id_switches: int
    num_frames: int

@dataclass
class AggregateMetrics:
    tracker_name: str
    mota: float
    motp: float
    total_id_switches: int
    sequence_metrics: list[SequenceMetrics]

@dataclass
class Config:
    # Detector
    pretrained_checkpoint: str          # e.g. "yolov8n.pt"
    dataset_yaml: str                   # path to dataset.yaml
    train_epochs: int                   # default 50
    output_dir: str                     # root output directory
    fine_tuned_checkpoint: str          # path to saved best.pt

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
    eval_output_path: str = "results/evaluation"
```

### MOTChallenge Results File Format

Each line: `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1`

### Ground-Truth Conversion

YOLO labels (`class cx cy w h` normalised) are converted to pixel-space `(x, y, w, h)` using frame dimensions before being passed to `motmetrics`.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

**Property Reflection**: After prework analysis, several properties were consolidated:
- Track lifecycle (max_age deletion) is the same property for SORT and DeepSORT — unified into Property 4.
- Track ID uniqueness applies to all trackers — unified into Property 5.
- Results file format is the same for all three trackers — covered by a single example test, not a property.
- MOTA, MOTP, and ID-switch computation are all pure functions — kept as separate properties (3 distinct formulas).

### Property 1: Detection output structure

*For any* input frame (any size, any content), every `Detection` returned by `YOLODetector.infer()` SHALL have `x >= 0`, `y >= 0`, `w > 0`, `h > 0`, and `confidence` in `[0.0, 1.0]`.

**Validates: Requirements 2.1**

### Property 2: Confidence threshold filtering

*For any* confidence threshold `t` in `[0.0, 1.0]` and any set of raw model outputs, no `Detection` returned by `YOLODetector.infer()` SHALL have `confidence < t`.

**Validates: Requirements 2.2**

### Property 3: Track ID persistence under continuous detection

*For any* sequence of frames where a detected object's bounding box overlaps with its previous-frame box at IoU >= the tracker's association threshold, the track ID assigned to that object SHALL remain the same across those frames.

**Validates: Requirements 3.1**

### Property 4: Track deletion after max_age misses

*For any* tracker (SORT or DeepSORT) configured with `max_age = k`, a track that receives no matching detection for `k + 1` consecutive frames SHALL NOT appear in the tracker's active track output on frame `k + 1`.

**Validates: Requirements 3.2, 4.3**

### Property 5: Track ID uniqueness

*For any* sequence of detections processed by any tracker, all currently active track IDs SHALL be distinct integers — no two simultaneously active tracks share the same ID.

**Validates: Requirements 3.3**

### Property 6: Distance combination formula

*For any* alpha in `[0.0, 1.0]`, any IoU distance `d_iou`, and any cosine distance `d_cos`, the combined association distance computed by `DeepSORTTracker` SHALL equal `alpha * d_iou + (1 - alpha) * d_cos`.

**Validates: Requirements 4.2**

### Property 7: MOTA formula correctness

*For any* sequence of ground-truth annotations and tracker results, the MOTA computed by `Evaluator` SHALL equal `1 - (FP + FN + IDSW) / num_gt_objects`, matching the standard MOTChallenge definition.

**Validates: Requirements 6.1**

### Property 8: MOTP formula correctness

*For any* set of matched detection-ground-truth pairs, the MOTP computed by `Evaluator` SHALL equal `sum(IoU_i for matched pairs) / num_matches`.

**Validates: Requirements 6.2**

### Property 9: ID switch count correctness

*For any* sequence of track-to-ground-truth assignments, the ID switch count computed by `Evaluator` SHALL equal the number of frames where a ground-truth object's assigned track ID changes from the previous frame.

**Validates: Requirements 6.3**

### Property 10: Metric aggregation consistency

*For any* collection of per-sequence `SequenceMetrics`, the `AggregateMetrics` produced by `Evaluator.aggregate()` SHALL have `total_id_switches` equal to the sum of `id_switches` across all sequences, and aggregate MOTA/MOTP computed as weighted averages by frame count.

**Validates: Requirements 6.4**

### Property 11: Config round-trip

*For any* valid configuration dictionary containing all required keys, serializing it to YAML and loading it via `ConfigLoader.load()` SHALL produce a `Config` object whose fields are equal to the original values.

**Validates: Requirements 9.1**

### Property 12: Missing required config key raises error

*For any* required configuration key `k`, loading a YAML config that omits `k` SHALL raise a `ConfigError` whose message contains the name of the missing key `k`.

**Validates: Requirements 9.2**

### Property 13: Optional config defaults

*For any* optional configuration parameter with a documented default value `d`, loading a YAML config that omits that parameter SHALL produce a `Config` object where that field equals `d`.

**Validates: Requirements 9.3**

### Property 14: Frame rendering preserves shape

*For any* input frame and any list of `Track` objects, the frame returned by `FrameRenderer` SHALL have the same `(height, width, channels)` shape as the input frame.

**Validates: Requirements 8.1**

---

## Error Handling

| Condition | Component | Behaviour |
|---|---|---|
| Training image dir missing or empty | `YOLODetector.train()` | Raises `ConfigError` with path in message |
| Validation image dir missing or empty | `YOLODetector.train()` | Raises `ConfigError` with path in message |
| FairMOT weights file missing | `FairMOTTracker.__init__()` | Raises `ModelNotFoundError` with path in message |
| Required config key absent | `ConfigLoader.validate()` | Raises `ConfigError` naming the missing key |
| Results file missing for a sequence | `Evaluator.evaluate_tracker()` | Logs `WARNING` and skips that sequence |
| Output directory does not exist | `Pipeline.run()` / `ResultsWriter` | Creates directory with `os.makedirs(exist_ok=True)` |
| Frame decode failure | `SequenceLoader` | Logs `WARNING` and skips the frame |

All custom exceptions inherit from a base `TrackingPipelineError` for easy catch-all handling.

---

## Testing Strategy

### Unit Tests (example-based)

Focus on specific behaviors, error conditions, and integration points:

- `ConfigLoader`: missing required key raises `ConfigError` naming the key; optional keys use defaults; valid config loads correctly.
- `SequenceLoader`: correctly groups `{Name}_{Frame}.jpg` filenames; handles single-frame sequences; ignores non-jpg files.
- `YOLODetector.infer()`: returns empty list when no detections above threshold; raises on missing checkpoint.
- `ResultsWriter`: output file lines match MOTChallenge regex `^\d+,\d+,[\d.]+,[\d.]+,[\d.]+,[\d.]+,[\d.]+,-1,-1,-1$`.
- `Evaluator`: missing results file logs warning and continues; report written as both CSV and JSON.
- `Pipeline`: output directory created if absent; FairMOT raises `ModelNotFoundError` on missing weights.

### Property-Based Tests

Uses **Hypothesis** (Python) with minimum **100 iterations** per property. Each test is tagged with a comment referencing the design property.

```
# Feature: underwater-object-tracking, Property 1: Detection output structure
# Feature: underwater-object-tracking, Property 2: Confidence threshold filtering
# Feature: underwater-object-tracking, Property 3: Track ID persistence
# Feature: underwater-object-tracking, Property 4: Track deletion after max_age misses
# Feature: underwater-object-tracking, Property 5: Track ID uniqueness
# Feature: underwater-object-tracking, Property 6: Distance combination formula
# Feature: underwater-object-tracking, Property 7: MOTA formula correctness
# Feature: underwater-object-tracking, Property 8: MOTP formula correctness
# Feature: underwater-object-tracking, Property 9: ID switch count correctness
# Feature: underwater-object-tracking, Property 10: Metric aggregation consistency
# Feature: underwater-object-tracking, Property 11: Config round-trip
# Feature: underwater-object-tracking, Property 12: Missing required config key raises error
# Feature: underwater-object-tracking, Property 13: Optional config defaults
# Feature: underwater-object-tracking, Property 14: Frame rendering preserves shape
```

Hypothesis strategies to implement:
- `st.floats(0, 1)` for confidence thresholds and alpha weights
- `st.lists(detection_strategy())` for detection lists
- `st.integers(1, 10)` for `max_age`
- `st.fixed_dictionaries(...)` for config dicts
- `st.numpy.arrays(dtype=np.uint8, shape=...)` for frames

### Integration Tests

Run against real data (small subset of val sequences):

- Post-training: assert `mAP@0.5 >= 0.70` from saved `results.csv`.
- End-to-end SORT/DeepSORT/FairMOT pipeline on 2–3 val sequences: results files created, MP4 written, evaluation report generated.
- NMS IoU threshold correctly passed to Ultralytics model call (verified via mock).

### Smoke Tests

- `YOLODetector` initialises from `yolov8n.pt` without error.
- Training paths resolved from `dataset.yaml` without error.
- Best checkpoint file exists at configured path after training.
- Annotated MP4 file written to output directory.
