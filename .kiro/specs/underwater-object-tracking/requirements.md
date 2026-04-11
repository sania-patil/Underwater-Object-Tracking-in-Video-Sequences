# Requirements Document

## Introduction

This feature implements an underwater object tracking system for video sequences.
The pipeline fine-tunes a YOLO detector on an existing preprocessed dataset, then
integrates it with three multi-object trackers (SORT, DeepSORT, FairMOT) to track
underwater objects across frames under challenging conditions such as occlusion and
motion blur. The system evaluates and compares tracker performance using standard
MOT metrics: MOTA, MOTP, and ID switches.

## Glossary

- **Detector**: The fine-tuned YOLOv8 model responsible for producing per-frame bounding box detections.
- **Tracker**: An algorithm that associates detections across frames and assigns persistent track IDs.
- **SORT**: Simple Online and Realtime Tracking — a Kalman-filter + Hungarian-algorithm baseline tracker.
- **DeepSORT**: An extension of SORT that adds a deep appearance descriptor (Re-ID) to reduce ID switches.
- **FairMOT**: A joint detection-and-tracking model that learns detection and Re-ID features simultaneously.
- **Track**: A sequence of bounding boxes assigned a single persistent integer ID across frames.
- **MOTA**: Multiple Object Tracking Accuracy — a composite metric combining false positives, false negatives, and ID switches.
- **MOTP**: Multiple Object Tracking Precision — average spatial overlap between matched detections and ground-truth boxes.
- **ID_Switch**: An event where a track ID changes for an object that was already being tracked.
- **Video_Sequence**: An ordered set of consecutive frames extracted from a source video, named with a common prefix (e.g., ArmyDiver1).
- **Ground_Truth**: Per-frame YOLO-format annotation files located in `yolo_dataset/labels/`.
- **Pipeline**: The end-to-end system combining the Detector and a Tracker to produce tracking output for a Video_Sequence.
- **Results_File**: A text file in MOTChallenge format containing frame index, track ID, and bounding box coordinates for each tracked object.
- **Evaluation_Report**: A structured output (CSV or JSON) containing MOTA, MOTP, and ID_Switch counts per tracker and aggregated across all Video_Sequences.


## Requirements

### Requirement 1: YOLO Detector Fine-Tuning

**User Story:** As a researcher, I want to fine-tune a YOLO model on the underwater dataset, so that the Detector produces accurate bounding box detections for underwater objects.

#### Acceptance Criteria

1. THE Detector SHALL be initialised from a pretrained YOLOv8 checkpoint before fine-tuning begins.
2. WHEN fine-tuning is started, THE Detector SHALL load training images from `yolo_dataset/images/train` and labels from `yolo_dataset/labels/train` as specified in `yolo_dataset/dataset.yaml`.
3. WHEN fine-tuning is started, THE Detector SHALL load validation images from `yolo_dataset/images/val` and labels from `yolo_dataset/labels/val` as specified in `yolo_dataset/dataset.yaml`.
4. WHEN fine-tuning completes, THE Detector SHALL achieve a mean Average Precision at IoU 0.5 (mAP@0.5) of at least 0.70 on the validation split.
5. WHEN fine-tuning completes, THE Detector SHALL save the best-performing checkpoint to a configurable output directory.
6. IF the training data directory is missing or empty, THEN THE Detector SHALL raise a descriptive error and halt training.
7. IF the validation data directory is missing or empty, THEN THE Detector SHALL raise a descriptive error and halt training.

---

### Requirement 2: Per-Frame Detection Inference

**User Story:** As a researcher, I want the fine-tuned Detector to run inference on video frames, so that downstream trackers receive per-frame bounding box proposals.

#### Acceptance Criteria

1. WHEN a Video_Sequence is provided, THE Detector SHALL produce one detection output per frame containing bounding box coordinates (x, y, width, height) in pixel space and a confidence score.
2. WHEN running inference, THE Detector SHALL apply a configurable confidence threshold (default 0.25) and discard detections below it.
3. WHEN running inference, THE Detector SHALL apply a configurable IoU threshold for non-maximum suppression (default 0.45).
4. IF a frame contains no detections above the confidence threshold, THEN THE Detector SHALL return an empty detection list for that frame without raising an error.

---

### Requirement 3: SORT Tracker Integration

**User Story:** As a researcher, I want a SORT tracker integrated with the Detector, so that I have a Kalman-filter baseline for multi-object tracking.

#### Acceptance Criteria

1. WHEN a Video_Sequence is processed by the SORT Pipeline, THE Tracker SHALL assign a persistent integer track ID to each detected object across consecutive frames.
2. WHEN a detection is not matched to any existing track for more than a configurable number of frames (default 1), THE Tracker SHALL delete that track.
3. WHEN a new detection cannot be matched to an existing track, THE Tracker SHALL initialise a new track with a unique ID.
4. WHEN the SORT Pipeline finishes processing a Video_Sequence, THE Pipeline SHALL write a Results_File in MOTChallenge format to the configured output directory.

---

### Requirement 4: DeepSORT Tracker Integration

**User Story:** As a researcher, I want a DeepSORT tracker integrated with the Detector, so that appearance-based Re-ID reduces ID switches compared to SORT.

#### Acceptance Criteria

1. WHEN a Video_Sequence is processed by the DeepSORT Pipeline, THE Tracker SHALL extract a deep appearance descriptor for each detection using a configurable Re-ID model.
2. WHEN associating detections to tracks, THE Tracker SHALL combine IoU distance and appearance cosine distance using a configurable weighting.
3. WHEN a detection is not matched to any existing track for more than a configurable number of frames (default 30), THE Tracker SHALL delete that track.
4. WHEN the DeepSORT Pipeline finishes processing a Video_Sequence, THE Pipeline SHALL write a Results_File in MOTChallenge format to the configured output directory.

---

### Requirement 5: FairMOT Tracker Integration

**User Story:** As a researcher, I want a FairMOT tracker integrated into the pipeline, so that joint detection and Re-ID learning can be compared against the separate-model approaches.

#### Acceptance Criteria

1. WHEN a Video_Sequence is processed by the FairMOT Pipeline, THE Tracker SHALL simultaneously produce bounding box detections and Re-ID embeddings from a single forward pass.
2. WHEN associating detections to tracks, THE Tracker SHALL use the joint Re-ID embeddings to resolve ambiguous matches caused by occlusion or motion blur.
3. WHEN the FairMOT Pipeline finishes processing a Video_Sequence, THE Pipeline SHALL write a Results_File in MOTChallenge format to the configured output directory.
4. IF the FairMOT model weights file is missing, THEN THE Pipeline SHALL raise a descriptive error and halt processing.

---

### Requirement 6: Tracking Evaluation

**User Story:** As a researcher, I want to evaluate each tracker using MOTA, MOTP, and ID switches, so that I can compare tracker performance objectively.

#### Acceptance Criteria

1. WHEN evaluation is run for a tracker, THE Evaluator SHALL compute MOTA per Video_Sequence using the corresponding Ground_Truth annotations and the tracker's Results_File.
2. WHEN evaluation is run for a tracker, THE Evaluator SHALL compute MOTP per Video_Sequence using the corresponding Ground_Truth annotations and the tracker's Results_File.
3. WHEN evaluation is run for a tracker, THE Evaluator SHALL count ID_Switch events per Video_Sequence.
4. WHEN evaluation is run for all three trackers, THE Evaluator SHALL aggregate MOTA, MOTP, and ID_Switch counts across all Video_Sequences and write an Evaluation_Report.
5. IF a Results_File is missing for a Video_Sequence, THEN THE Evaluator SHALL log a warning for that sequence and continue evaluating the remaining sequences.
6. THE Evaluator SHALL use an IoU threshold of 0.5 when matching predicted tracks to Ground_Truth boxes for MOTA and MOTP computation.

---

### Requirement 7: Tracker Comparison Report

**User Story:** As a researcher, I want a side-by-side comparison of SORT, DeepSORT, and FairMOT metrics, so that I can identify the best-performing tracker for underwater conditions.

#### Acceptance Criteria

1. WHEN all three trackers have been evaluated, THE Evaluator SHALL produce a single Evaluation_Report containing MOTA, MOTP, and ID_Switch counts for SORT, DeepSORT, and FairMOT in a tabular format.
2. THE Evaluation_Report SHALL include per-sequence metrics and overall aggregate metrics for each tracker.
3. WHEN the Evaluation_Report is written, THE Evaluator SHALL save it to a configurable output path in both CSV and JSON formats.

---

### Requirement 8: Annotated Video Output

**User Story:** As a researcher, I want the pipeline to produce annotated video output, so that I can visually inspect tracking results.

#### Acceptance Criteria

1. WHEN a Pipeline processes a Video_Sequence, THE Pipeline SHALL render each frame with bounding boxes and track IDs overlaid.
2. WHEN rendering is complete, THE Pipeline SHALL write the annotated frames as a video file (MP4) to the configured output directory.
3. IF the output directory does not exist, THEN THE Pipeline SHALL create it before writing any output files.

---

### Requirement 9: Configuration Management

**User Story:** As a researcher, I want all pipeline parameters to be configurable via a single config file, so that I can reproduce and modify experiments without changing source code.

#### Acceptance Criteria

1. THE Pipeline SHALL read all configurable parameters (model checkpoint path, confidence threshold, NMS IoU threshold, tracker-specific parameters, output directory, evaluation IoU threshold) from a single YAML configuration file.
2. WHEN a required configuration parameter is missing from the YAML file, THE Pipeline SHALL raise a descriptive error identifying the missing parameter and halt execution.
3. WHERE a parameter has a defined default value, THE Pipeline SHALL use that default when the parameter is absent from the configuration file.
