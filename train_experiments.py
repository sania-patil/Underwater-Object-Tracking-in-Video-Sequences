"""
Grid search: each model x each hyperparameter config.
Runs all combinations sequentially and saves comparison CSV.
"""

import torch, csv, os, itertools
from ultralytics import YOLO

DATASET_YAML = "/workspace/yolo_dataset/dataset.yaml"
OUTPUT_DIR   = "/workspace/runs/experiments"

# --- Models to try ---
MODELS = {
    "yolo26n": "/workspace/yolo26n.pt",
    "yolov8s": "/workspace/yolov8s.pt",
    "yolov8m": "/workspace/yolov8m.pt",
}

# --- Hyperparameter grid ---
HP_GRID = [
    {
        "tag": "adamw_640",
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
        "imgsz": 640, "batch": 16, "mixup": 0.15, "mosaic": 1.0,
        "close_mosaic": 10,
    },
    {
        "tag": "adamw_832",
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
        "imgsz": 832, "batch": 8, "mixup": 0.15, "mosaic": 1.0,
        "close_mosaic": 10,
    },
    {
        "tag": "sgd_640",
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "imgsz": 640, "batch": 16, "mixup": 0.1, "mosaic": 1.0,
        "close_mosaic": 15,
    },
    {
        "tag": "sgd_832",
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "imgsz": 832, "batch": 8, "mixup": 0.1, "mosaic": 1.0,
        "close_mosaic": 15,
    },
    {
        "tag": "adamw_lowlr",
        "optimizer": "AdamW", "lr0": 0.0005, "lrf": 0.005,
        "imgsz": 640, "batch": 16, "mixup": 0.2, "mosaic": 1.0,
        "close_mosaic": 20,
    },
    {
        "tag": "adamw_highaug",
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
        "imgsz": 640, "batch": 16, "mixup": 0.3, "mosaic": 1.0,
        "close_mosaic": 20,
    },
    # --- yolov8m specific configs (medium model needs lower lr + more warmup) ---
    {
        "tag": "medium_adamw_640",
        "optimizer": "AdamW", "lr0": 0.0008, "lrf": 0.01,
        "imgsz": 640, "batch": 12, "mixup": 0.15, "mosaic": 1.0,
        "close_mosaic": 10,
    },
    {
        "tag": "medium_adamw_832",
        "optimizer": "AdamW", "lr0": 0.0008, "lrf": 0.008,
        "imgsz": 832, "batch": 6, "mixup": 0.2, "mosaic": 1.0,
        "close_mosaic": 15,
    },
    {
        "tag": "medium_sgd_cosine",
        "optimizer": "SGD", "lr0": 0.008, "lrf": 0.005,
        "imgsz": 640, "batch": 12, "mixup": 0.1, "mosaic": 1.0,
        "close_mosaic": 15,
    },
]

# --- Fixed settings (underwater-specific, don't change) ---
FIXED = dict(
    data=DATASET_YAML,
    epochs=30,          # 30 epochs per run — enough to compare, not too slow
    device=0,
    cos_lr=True,
    amp=True,
    patience=10,
    workers=2,
    cache=False,
    save=True,
    plots=False,        # skip plots to save time
    verbose=False,
    exist_ok=True,      # allow re-running without name conflicts
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.3,
    fliplr=0.5,
    copy_paste=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    weight_decay=0.0005,
    warmup_epochs=3,
    momentum=0.937,
)

def run():
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total experiments: {len(MODELS) * len(HP_GRID)}\n")

    summary = []
    total = len(MODELS) * len(HP_GRID)
    idx = 0

    for model_name, model_path in MODELS.items():
        for hp in HP_GRID:
            idx += 1
            tag = hp["tag"]
            exp_name = f"{model_name}_{tag}"
            hp_params = {k: v for k, v in hp.items() if k != "tag"}

            print(f"[{idx}/{total}] {exp_name}")

            try:
                model = YOLO(model_path)
                results = model.train(
                    **FIXED,
                    **hp_params,
                    project=OUTPUT_DIR,
                    name=exp_name,
                )
                r = results.results_dict
                row = {
                    "experiment": exp_name,
                    "model": model_name,
                    "optimizer": hp["optimizer"],
                    "lr0": hp["lr0"],
                    "imgsz": hp["imgsz"],
                    "batch": hp["batch"],
                    "mixup": hp["mixup"],
                    "mAP50": round(r.get("metrics/mAP50(B)", 0), 4),
                    "mAP50_95": round(r.get("metrics/mAP50-95(B)", 0), 4),
                    "precision": round(r.get("metrics/precision(B)", 0), 4),
                    "recall": round(r.get("metrics/recall(B)", 0), 4),
                    "status": "ok",
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                row = {
                    "experiment": exp_name, "model": model_name,
                    "optimizer": hp["optimizer"], "lr0": hp["lr0"],
                    "imgsz": hp["imgsz"], "batch": hp["batch"],
                    "mixup": hp["mixup"],
                    "mAP50": 0, "mAP50_95": 0,
                    "precision": 0, "recall": 0,
                    "status": f"error: {e}",
                }

            summary.append(row)
            print(f"  mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}  status={row['status']}\n")

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Print sorted summary
    print("\n" + "="*70)
    print("RESULTS (sorted by mAP50)")
    print("="*70)
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        print(f"{row['experiment']:40s}  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50_95']:.4f}")

    print(f"\nFull report: {report_path}")

if __name__ == "__main__":
    run()
