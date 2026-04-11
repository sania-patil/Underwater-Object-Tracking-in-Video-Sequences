"""
Hyperparameter search for underwater object detection.
Reads metrics directly from results.csv written by Ultralytics.
"""

import os, csv, shutil
import torch
from ultralytics import YOLO

DATASET_YAML = "/workspace/yolo_dataset/dataset.yaml"
BASE_DIR     = "/workspace/runs/hp_search"

EXPERIMENTS = [
    # --- yolo26n ---
    {"name": "n_adamw_001",  "model": "/workspace/yolo26n.pt",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 640, "batch": 16},
    {"name": "n_adamw_0005", "model": "/workspace/yolo26n.pt",  "lr0": 0.0005, "optimizer": "AdamW", "imgsz": 640, "batch": 16},
    {"name": "n_sgd_001",    "model": "/workspace/yolo26n.pt",  "lr0": 0.01,   "optimizer": "SGD",   "imgsz": 640, "batch": 16},
    # --- yolov8s ---
    {"name": "s_adamw_001",  "model": "/workspace/yolov8s.pt",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 640, "batch": 16},
    {"name": "s_adamw_0005", "model": "/workspace/yolov8s.pt",  "lr0": 0.0005, "optimizer": "AdamW", "imgsz": 640, "batch": 16},
    {"name": "s_adamw_832",  "model": "/workspace/yolov8s.pt",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 832, "batch": 8},
    {"name": "s_sgd_001",    "model": "/workspace/yolov8s.pt",  "lr0": 0.01,   "optimizer": "SGD",   "imgsz": 640, "batch": 16},
    # --- yolov8m ---
    {"name": "m_adamw_001",  "model": "/workspace/yolov8m.pt",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 640, "batch": 12},
    {"name": "m_adamw_0005", "model": "/workspace/yolov8m.pt",  "lr0": 0.0005, "optimizer": "AdamW", "imgsz": 640, "batch": 12},
    {"name": "m_adamw_832",  "model": "/workspace/yolov8m.pt",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 832, "batch": 6},
]

COMMON = dict(
    data=DATASET_YAML,
    epochs=25,
    device=0,
    cos_lr=True,
    amp=True,
    patience=8,
    workers=2,
    cache=False,
    save=True,
    plots=False,
    verbose=True,
    exist_ok=True,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    close_mosaic=5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    mosaic=1.0, mixup=0.15,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
)


def read_best_map(run_dir):
    """Read best mAP50 from results.csv written by Ultralytics."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return 0.0, 0.0
    best_map50 = 0.0
    best_map95 = 0.0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                m50 = float(row.get("metrics/mAP50(B)", 0))
                m95 = float(row.get("metrics/mAP50-95(B)", 0))
                if m50 > best_map50:
                    best_map50 = m50
                    best_map95 = m95
            except ValueError:
                continue
    return best_map50, best_map95


def run():
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total experiments: {len(EXPERIMENTS)}\n")

    summary = []

    for i, exp in enumerate(EXPERIMENTS):
        name      = exp["name"]
        model_path = exp["model"]
        run_dir   = os.path.join(BASE_DIR, name)

        print(f"[{i+1}/{len(EXPERIMENTS)}] {name}  model={os.path.basename(model_path)}  lr0={exp['lr0']}  opt={exp['optimizer']}  imgsz={exp['imgsz']}")

        try:
            model = YOLO(model_path)
            model.train(
                **COMMON,
                lr0=exp["lr0"],
                optimizer=exp["optimizer"],
                imgsz=exp["imgsz"],
                batch=exp["batch"],
                project=BASE_DIR,
                name=name,
            )
        except Exception as e:
            print(f"  ERROR: {e}\n")
            summary.append({"name": name, "model": os.path.basename(model_path),
                            "lr0": exp["lr0"], "optimizer": exp["optimizer"],
                            "imgsz": exp["imgsz"], "batch": exp["batch"],
                            "mAP50": 0, "mAP50_95": 0, "status": f"error: {e}"})
            continue

        map50, map95 = read_best_map(run_dir)
        print(f"  -> best mAP50={map50:.4f}  mAP50-95={map95:.4f}\n")

        summary.append({
            "name": name,
            "model": os.path.basename(model_path),
            "lr0": exp["lr0"],
            "optimizer": exp["optimizer"],
            "imgsz": exp["imgsz"],
            "batch": exp["batch"],
            "mAP50": round(map50, 4),
            "mAP50_95": round(map95, 4),
            "status": "ok",
        })

    # Save report
    os.makedirs(BASE_DIR, exist_ok=True)
    report = os.path.join(BASE_DIR, "hp_results.csv")
    with open(report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Print sorted table
    print("\n" + "="*70)
    print(f"{'EXPERIMENT':<20} {'MODEL':<12} {'LR':>7} {'OPT':<6} {'IMG':>4} {'mAP50':>7} {'mAP95':>7}")
    print("="*70)
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        print(f"{row['name']:<20} {row['model']:<12} {row['lr0']:>7} {row['optimizer']:<6} {row['imgsz']:>4} {row['mAP50']:>7.4f} {row['mAP50_95']:>7.4f}")

    best = max(summary, key=lambda x: x["mAP50"])
    print(f"\nBest: {best['name']}  mAP50={best['mAP50']}  model={best['model']}  lr0={best['lr0']}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    run()
