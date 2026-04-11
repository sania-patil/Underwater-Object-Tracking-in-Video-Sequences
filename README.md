# Underwater Object Tracking in Video Sequences

> Track underwater objects across video frames under occlusion and motion blur.

---

## Overview

Complete **Detection + Tracking pipeline** for underwater video sequences. A YOLOv11m detector is fine-tuned on an underwater dataset and integrated with three trackers — **SORT**, **DeepSORT**, and **FairMOT** — evaluated using **MOTA**, **MOTP**, and **ID Switches**.

---

## Dataset

- **Source:** Fish4Knowledge Video Dataset
- **Sequences (8):** BlueFish2, BoySwimming, Dolphin2, Fisherman, HoverFish2, SeaDiver, SeaTurtle2, SeaTurtle3
- **Format:** YOLO (single class: `underwater_object`)
- **Original:** 20,748 train / 5,307 val frames
- **Subsampled (every 5th frame):** 4,159 train / 1,064 val frames

---

## Project Structure

```
├── src/
│   ├── models.py              # Data classes
│   ├── config_loader.py       # YAML config loader
│   ├── sequence_loader.py     # Groups frames by sequence
│   ├── detector.py            # YOLOv11m inference
│   ├── results_writer.py      # MOTChallenge output
│   ├── renderer.py            # Annotated MP4 output
│   ├── evaluator.py           # MOTA/MOTP/ID switches
│   ├── pipeline.py            # Pipeline orchestrator
│   └── trackers/
│       ├── sort_tracker.py    # SORT
│       ├── deepsort_tracker.py# DeepSORT
│       └── fairmot_tracker.py # FairMOT
├── run_pipeline.py            # Main entry point
├── finetune_yolo.py           # Fine-tuning script
├── subsample_dataset.py       # Frame subsampling
├── plot_results.py            # Result graphs
├── config.yaml
└── requirements.txt
```

---

## Detector: YOLOv11m

### Models Compared

| Model | Params | mAP@0.5 |
|-------|--------|---------|
| yolo26n | ~3M | 0.402 |
| yolo11s | ~9M | 0.475 |
| **yolo11m** | **~20M** | **0.504** |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| lr0 | 0.001 |
| lrf | 0.01 |
| LR Schedule | Cosine annealing |
| Warmup epochs | 5 |
| Weight decay | 0.0005 |
| Batch size | 12 |
| Image size | 640×640 |
| Epochs | 100 (early stop patience=20) |

### Augmentation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| hsv_s | 0.7 | Murky water color |
| hsv_v | 0.4 | Depth/lighting variation |
| flipud | 0.3 | Any orientation |
| mosaic | 1.0 | Image diversity |
| mixup | 0.15 | Blend images |
| copy_paste | 0.1 | Object diversity |
| degrees | ±10° | Camera tilt |

---

## Trackers

### SORT (Baseline)
Kalman Filter + Hungarian Algorithm (IoU-based association)

| Parameter | Value |
|-----------|-------|
| max_age | 1 |
| min_hits | 3 |
| iou_threshold | 0.3 |

### DeepSORT
SORT + MobileNetV2 Re-ID appearance embeddings

| Parameter | Value |
|-----------|-------|
| max_age | 30 |
| nn_budget | 100 |
| max_cosine_distance | 0.4 |

### FairMOT
Stricter Re-ID matching to reduce ID switches

| Parameter | Value |
|-----------|-------|
| max_age | 20 |
| nn_budget | 200 |
| max_cosine_distance | 0.2 |
| conf_threshold | 0.4 |

---

## Results

### Aggregate

| Tracker | MOTA | MOTP | ID Switches |
|---------|------|------|-------------|
| **SORT** | **+0.099** | 0.239 | 68 |
| FairMOT | -0.095 | 0.261 | **28** |
| DeepSORT | -0.993 | **0.267** | 43 |

### Per-Sequence MOTA

| Sequence | SORT | DeepSORT | FairMOT |
|----------|------|----------|---------|
| BlueFish2 | -0.371 | -4.976 | -1.157 |
| BoySwimming | -0.992 | -0.997 | -0.969 |
| Dolphin2 | **0.462** | -0.213 | 0.321 |
| Fisherman | **0.335** | -0.141 | 0.306 |
| HoverFish2 | **0.421** | -0.125 | 0.149 |
| SeaDiver | **0.244** | -1.304 | -0.258 |
| SeaTurtle2 | 0.266 | 0.021 | **0.317** |
| SeaTurtle3 | **0.416** | -0.436 | 0.397 |

### Single vs Multi-Object

| Tracker | Single MOTA | Multi MOTA | ID Sw (Single) | ID Sw (Multi) |
|---------|------------|-----------|----------------|---------------|
| SORT | -0.010 | **+0.202** | 31 | 37 |
| FairMOT | -0.194 | -0.003 | **13** | **15** |
| DeepSORT | -0.740 | -1.230 | 12 | 31 |

---

## Key Findings

1. **SORT is best overall** (MOTA=+0.099) — simple IoU matching outperforms appearance-based Re-ID for underwater objects
2. **FairMOT has fewest ID switches** (28) — stricter appearance matching preserves identity better
3. **DeepSORT performs worst** — Re-ID model trained on humans fails on fish/turtles
4. **All trackers perform better on multi-object sequences** — spatially separated objects are easier to track
5. **Low MOTA is primarily due to detector** (mAP50=0.504) — every missed detection counts as a false negative

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Fine-tune detector
```bash
python finetune_yolo.py
```

### Run tracking pipeline
```bash
python run_pipeline.py --config config.yaml
```

### Generate result graphs
```bash
python plot_results.py
```

---

## Environment

- Python 3.11
- PyTorch 2.12.0+cu128
- Ultralytics 8.4.35
- GPU: NVIDIA RTX PRO 2000 Blackwell (16GB)
- Docker: `izor12/deep_yolo:latest`
