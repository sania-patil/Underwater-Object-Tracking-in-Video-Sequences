"""
Trains yolo11s and yolo11m on subsampled underwater dataset.
Compares results and picks the best model.
"""

import os, csv, torch
from ultralytics import YOLO

# W&B integration — Ultralytics logs automatically when wandb is installed
try:
    import wandb
    wandb.init(project="underwater-tracking", name="yolo11_compare")
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

DATASET_YAML = "/workspace/yolo_dataset_sub/dataset.yaml"
OUTPUT_DIR   = "/workspace/runs/compare"

MODELS = [
    {"name": "yolo11s_sub", "model": "/workspace/yolo11s.pt", "batch": 16},
    {"name": "yolo11m_sub", "model": "/workspace/yolo11m.pt", "batch": 12},
]

COMMON = dict(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    mosaic=1.0, mixup=0.15,
    copy_paste=0.1,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
    amp=False,  # disabled: Ultralytics check_amp downloads yolo26n.pt which fails in container
    close_mosaic=15,
    patience=20,
    save=True,
    save_period=10,
    plots=True,
    workers=2,
    cache=False,
    verbose=True,
    exist_ok=False,
)


def read_best_map(run_dir):
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return 0.0, 0.0
    best50, best95 = 0.0, 0.0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                m50 = float(row.get("metrics/mAP50(B)", 0))
                m95 = float(row.get("metrics/mAP50-95(B)", 0))
                if m50 > best50:
                    best50, best95 = m50, m95
            except ValueError:
                continue
    return best50, best95


def run():
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    summary = []

    for exp in MODELS:
        print(f"\n{'='*60}")
        print(f"Training: {exp['name']}  ({exp['model']})")
        print(f"{'='*60}")

        try:
            model = YOLO(exp["model"])
            model.train(
                **COMMON,
                batch=exp["batch"],
                project=OUTPUT_DIR,
                name=exp["name"],
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append({"name": exp["name"], "mAP50": 0, "mAP50_95": 0, "status": str(e)})
            continue

        run_dir = os.path.join(OUTPUT_DIR, exp["name"])
        map50, map95 = read_best_map(run_dir)
        print(f"\n  Best mAP50={map50:.4f}  mAP50-95={map95:.4f}")
        summary.append({
            "name": exp["name"],
            "model": exp["model"],
            "mAP50": round(map50, 4),
            "mAP50_95": round(map95, 4),
            "weights": os.path.join(run_dir, "weights", "best.pt"),
            "status": "ok",
        })

    # Report
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        print(f"{row['name']:<20}  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50_95']:.4f}  status={row['status']}")

    best = max(summary, key=lambda x: x["mAP50"])
    print(f"\nWinner: {best['name']}  mAP50={best['mAP50']}")
    print(f"Use weights: {best.get('weights', 'N/A')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    run()
