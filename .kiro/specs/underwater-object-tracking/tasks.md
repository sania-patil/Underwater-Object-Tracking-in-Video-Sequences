# Implementation Plan: Underwater Object Tracking

## Overview

Build the tracking pipeline and evaluation system on top of the existing fine-tuned YOLOv8 detector.
Training is handled by the existing `finetune_yolo.py` — tasks here cover everything from project
setup through tracker integration, evaluation, and reporting. All code is Python.
The fine-tuned checkpoint is expected at `runs/uot32_finetune/weights/best.pt`.

## Tasks

- [ ] 1. Project setup — dependencies, config, and directory structure
  - Create `requirements.txt` listing: `ultralytics`, `torch`, `opencv-python`, `numpy`, `pyyaml`, `motmetrics`, `hypothesis`, `pytest`, `deep-sort-realtime`, `fairmot` (or equivalent)
  - Create `config.yaml` with all parameters from the `Config` dataclass (checkpoint path, conf/NMS thresholds, SORT/DeepSORT/FairMOT params, output dirs, eval IoU threshold)
  - Create package skeleton: `src/__init__.py`, `src/models.py`, `src/config_loader.py`, `src/sequence_loader.py`, `src/detector.py`, `src/trackers/`, `src/results_writer.py`, `src/renderer.py`, `src/evaluator.py`, `src/pipeline.py`; and `tests/test_unit.py`, `tests/test_properties.py`
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 2. Data models and ConfigLoader
  - [x] 2.1 Implement data models in `src/models.py`
    - Write `Detection`, `Track`, `SequenceMetrics`, `AggregateMetrics`, `Config` dataclasses as specified in the design
    - Add `TrackingPipelineError`, `ConfigError`, `ModelNotFoundError` exception classes (all inherit from `TrackingPipelineError`)
    - _Requirements: 2.1, 3.1, 6.1_

  - [x] 2.2 Implement `ConfigLoader` in `src/config_loader.py`
    - `load(path)` reads YAML and returns a `Config` object, applying documented defaults for optional fields
    - `validate(cfg)` raises `ConfigError` naming any missing required key; required keys: `fine_tuned_checkpoint`, `output_dir`, `dataset_yaml`
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ]* 2.3 Write property tests for ConfigLoader (Properties 11, 12, 13)
    - **Property 11: Config round-trip** — serialize a valid config dict to YAML, reload via `ConfigLoader.load()`, assert all fields equal original values
    - **Property 12: Missing required config key raises error** — for each required key, omit it and assert `ConfigError` message contains the key name
    - **Property 13: Optional config defaults** — omit each optional key, assert field equals documented default
    - **Validates: Requirements 9.1, 9.2, 9.3**

  - [ ]* 2.4 Write unit tests for ConfigLoader
    - Test valid config loads without error; missing required key raises `ConfigError` with key name in message; optional keys fall back to defaults
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 3. SequenceLoader
  - [x] 3.1 Implement `SequenceLoader` in `src/sequence_loader.py`
    - `load_sequences(image_dir)` scans `*.jpg` files, groups by `{SequenceName}` prefix (split on last `_`), sorts frames by integer frame number, returns `dict[str, list[str]]`
    - Log `WARNING` and skip frames that fail to decode
    - _Requirements: 2.1_

  - [ ]* 3.2 Write unit tests for SequenceLoader
    - Test correct grouping of `{Name}_{Frame}.jpg` filenames; single-frame sequences; non-jpg files ignored
    - _Requirements: 2.1_

- [ ] 4. YOLODetector (inference only)
  - [x] 4.1 Implement `YOLODetector` in `src/detector.py`
    - `__init__(checkpoint_path)` loads the Ultralytics YOLO model from the configured checkpoint (default: `runs/uot32_finetune/weights/best.pt`)
    - `infer(frame, conf_thresh, nms_iou)` runs inference on a single `np.ndarray` frame, returns `list[Detection]` with `(x, y, w, h, confidence)` in pixel space
    - Returns empty list (no error) when no detections exceed `conf_thresh`
    - Training is handled by the existing `finetune_yolo.py` — do NOT add a train method
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 4.2 Write property tests for YOLODetector (Properties 1, 2)
    - **Property 1: Detection output structure** — for any frame, every returned `Detection` has `x >= 0`, `y >= 0`, `w > 0`, `h > 0`, `confidence` in `[0.0, 1.0]`
    - **Property 2: Confidence threshold filtering** — for any threshold `t`, no returned `Detection` has `confidence < t`
    - Use a mock/stub model to avoid requiring GPU in tests
    - **Validates: Requirements 2.1, 2.2**

  - [ ]* 4.3 Write unit tests for YOLODetector
    - Test returns empty list when no detections above threshold; NMS IoU threshold passed through to Ultralytics call (mock)
    - _Requirements: 2.2, 2.3, 2.4_

- [ ] 5. Checkpoint — ensure setup, models, config, detector, and sequence loader tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. SORT tracker and ResultsWriter
  - [x] 6.1 Implement `BaseTracker` in `src/trackers/base.py` and `SORTTracker` in `src/trackers/sort_tracker.py`
    - `BaseTracker` defines abstract `update(detections, frame) -> list[Track]` and `reset()`
    - `SORTTracker.__init__(max_age, min_hits, iou_threshold)` initialises the SORT algorithm
    - `update` converts `list[Detection]` to SORT input format, calls update, returns `list[Track]`; `reset()` reinitialises state between sequences
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 6.2 Implement `ResultsWriter` in `src/results_writer.py`
    - `write(tracks_by_frame, output_path)` writes MOTChallenge format lines: `frame,id,x,y,w,h,conf,-1,-1,-1`
    - Creates output directory if absent
    - Wire `SORTTracker` + `ResultsWriter` together: after processing a sequence, write results file to `{output_dir}/results/sort/{seq}.txt`
    - _Requirements: 3.4, 8.3_

  - [ ]* 6.3 Write property tests for SORTTracker (Properties 3, 4, 5)
    - **Property 3: Track ID persistence under continuous detection** — overlapping detections across frames yield the same track ID
    - **Property 4: Track deletion after max_age misses** — track absent for `k+1` frames does not appear in active tracks
    - **Property 5: Track ID uniqueness** — all active track IDs are distinct integers for any detection sequence
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [ ]* 6.4 Write unit tests for ResultsWriter
    - Test output lines match regex `^\d+,\d+,[\d.]+,[\d.]+,[\d.]+,[\d.]+,[\d.]+,-1,-1,-1$`; output directory created if absent
    - _Requirements: 3.4, 8.3_

- [ ] 7. DeepSORT tracker
  - [x] 7.1 Implement `DeepSORTTracker` in `src/trackers/deepsort_tracker.py`
    - `__init__(max_age, nn_budget, max_cosine_distance, reid_model_path)` initialises `deep_sort_realtime.DeepSort`
    - `update(detections, frame)` converts detections, calls DeepSORT update, returns `list[Track]`; `reset()` reinitialises between sequences
    - Wire to `ResultsWriter`: write results to `{output_dir}/results/deepsort/{seq}.txt`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 7.2 Write property tests for DeepSORTTracker (Properties 4, 5, 6)
    - **Property 4: Track deletion after max_age misses** — same as SORT but with DeepSORT instance
    - **Property 5: Track ID uniqueness** — distinct active IDs for any detection sequence
    - **Property 6: Distance combination formula** — for any alpha, d_iou, d_cos, assert combined distance equals `alpha * d_iou + (1 - alpha) * d_cos`
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 7.3 Write unit tests for DeepSORTTracker
    - Test appearance descriptor extracted per detection (mock Re-ID model); test max_age deletion with a short sequence
    - _Requirements: 4.1, 4.3_

- [ ] 8. FairMOT tracker
  - [ ] 8.1 Implement `FairMOTTracker` in `src/trackers/fairmot_tracker.py`
    - `__init__(model_path, conf_thresh)` loads FairMOT weights; raises `ModelNotFoundError` with path in message if weights file is missing
    - `update(detections, frame)` ignores `detections`, runs FairMOT forward pass on `frame`, returns `list[Track]`; `reset()` reinitialises between sequences
    - Wire to `ResultsWriter`: write results to `{output_dir}/results/fairmot/{seq}.txt`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 8.2 Write unit tests for FairMOTTracker
    - Test `ModelNotFoundError` raised with path in message when weights file is absent; test `update()` returns `list[Track]` (mock model)
    - _Requirements: 5.4_

- [ ] 9. FrameRenderer
  - [x] 9.1 Implement `FrameRenderer` in `src/renderer.py`
    - `render_sequence(frames, tracks_by_frame, output_path)` draws bounding boxes and track IDs on each frame using OpenCV, encodes to MP4 via `cv2.VideoWriter`, creates output dir if absent
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 9.2 Write property test for FrameRenderer (Property 14)
    - **Property 14: Frame rendering preserves shape** — for any input frame shape and any list of `Track` objects, the rendered frame has the same `(height, width, channels)` shape
    - **Validates: Requirements 8.1**

  - [ ]* 9.3 Write unit tests for FrameRenderer
    - Test annotated MP4 file is written to the configured output path; output directory created if absent
    - _Requirements: 8.2, 8.3_

- [ ] 10. Checkpoint — ensure tracker and renderer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Evaluator
  - [x] 11.1 Implement `Evaluator` in `src/evaluator.py`
    - `evaluate_tracker(tracker_name, results_dir, gt_dir, iou_threshold)` reads each `{seq}.txt` results file and corresponding YOLO-format ground-truth labels, converts YOLO normalised coords to pixel-space, feeds both into `motmetrics`, returns `dict[str, SequenceMetrics]`
    - Log `WARNING` and skip sequences where the results file is missing
    - `aggregate(metrics)` computes `AggregateMetrics`: `total_id_switches` = sum of per-sequence ID switches; MOTA/MOTP as frame-count-weighted averages
    - `write_report(all_metrics, output_path)` writes `{output_path}.csv` and `{output_path}.json` with per-sequence and aggregate metrics for all three trackers
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3_

  - [ ]* 11.2 Write property tests for Evaluator (Properties 7, 8, 9, 10)
    - **Property 7: MOTA formula correctness** — for any FP, FN, IDSW, num_gt, assert MOTA equals `1 - (FP + FN + IDSW) / num_gt`
    - **Property 8: MOTP formula correctness** — for any set of matched IoU values, assert MOTP equals `sum(IoU_i) / num_matches`
    - **Property 9: ID switch count correctness** — for any sequence of track-to-GT assignments, assert ID switch count equals number of frames where assigned track ID changes
    - **Property 10: Metric aggregation consistency** — for any collection of `SequenceMetrics`, assert `total_id_switches` equals sum of per-sequence `id_switches`
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

  - [ ]* 11.3 Write unit tests for Evaluator
    - Test missing results file logs warning and continues; report written as both CSV and JSON; IoU threshold of 0.5 used for matching
    - _Requirements: 6.5, 6.6, 7.3_

- [ ] 12. Pipeline orchestrator
  - [x] 12.1 Implement `Pipeline` in `src/pipeline.py`
    - `__init__(cfg, tracker)` wires together `YOLODetector`, the given `BaseTracker`, `ResultsWriter`, and `FrameRenderer`
    - `run(sequence_name)` iterates frames via `SequenceLoader`, runs `YOLODetector.infer()` per frame, calls `tracker.update()`, accumulates `tracks_by_frame`, then calls `ResultsWriter.write()` and `FrameRenderer.render_sequence()`
    - Creates output directories if absent
    - _Requirements: 3.4, 4.4, 5.3, 8.1, 8.2, 8.3_

  - [x] 12.2 Implement `run_pipeline.py` entry-point script
    - Loads `config.yaml` via `ConfigLoader`, instantiates all three trackers, runs `Pipeline.run()` for each tracker over all val sequences, then runs `Evaluator` and writes the comparison report
    - Accepts `--config` CLI argument (default `config.yaml`)
    - _Requirements: 7.1, 7.2, 7.3, 9.1_

  - [ ]* 12.3 Write unit tests for Pipeline
    - Test output directory created if absent; test FairMOT raises `ModelNotFoundError` on missing weights
    - _Requirements: 5.4, 8.3_

- [ ] 13. Final checkpoint — full test suite and end-to-end smoke test
  - Ensure all unit and property tests pass (`pytest tests/ --hypothesis-seed=0`)
  - Run `python run_pipeline.py` on 2–3 val sequences (e.g. `BlueFish2`) and verify: results `.txt` files created, annotated `.mp4` files written, `evaluation.csv` and `evaluation.json` generated
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Training is NOT part of these tasks — run `python finetune_yolo.py` separately first to produce `runs/uot32_finetune/weights/best.pt`
- Each task references specific requirements for traceability
- Property tests use Hypothesis with minimum 100 iterations per property
- Checkpoints ensure incremental validation at logical breaks
