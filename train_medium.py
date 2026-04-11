"""
Hyperparameter search for yolov8m (medium) on underwater dataset.
Reads metrics from results.csv for reliable reporting.
"""

import os, csv, torch
from ultralytics import YOLO

DATASET_YAML = "/workspace/yolo_dataset/dataset.yaml"
MODEL        = "/workspace/yolov8m.pt"
BASE_DIR     = "/workspace/runs/medium_search"

EXPERIMENTS = [
    {"name": "m_adamw_001_640",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 640, "batch": 12, "mixup": 0.15},
    {"name": "m_adamw_0005_640", "lr0": 0.0005, "optimizer": "AdamW", "imgsz": 640, "batch": 12, "mixup": 0.15},
    {"name": "m_adamw_0003_640", "lr0": 0.0003, "optimizer": "AdamW", "imgsz": 640, "batch": 12, "mixup": 0.20},
    {"name": "m_adamw_001_832",  "lr0": 0.001,  "optimizer": "AdamW", "imgsz": 832, "batch": 6,  "mixup": 0.15},
    {"name": "m_sgd_005_640",    "lr0": 0.005,  "optimizer": "SGD",   "imgsz": 640, "batch": 12, "mixup": 0.10},
    {"name": "m_sgd_001_640",    "lr0": 0.001,  "optimizer": "SGD",   "imgsz": 640, "batch": 12, "mixup": 0.10},
]

COMMON = dict(
    data=DATASET_YAML,
    epochs=30,
    device=0,
    cos_lr=True,
    amp=True,
    patience=8,
    workers=2,
    cache=False,
    save=True,
    plots=False,
    verbose=True,
    exist_ok=False,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    close_mosaic=8,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    mosaic=1.0,
    copy_paste=0.1,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
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
    print(f"Model: yolov8m  |  {len(EXPERIMENTS)} experiments x 30 epochs\n")

    summary = []

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        print(f"[{i+1}/{len(EXPERIMENTS)}] {name}  lr0={exp['lr0']}  opt={exp['optimizer']}  imgsz={exp['imgsz']}  batch={exp['batch']}")

        try:
            model = YOLO(MODEL)
            model.train(
                **COMMON,
                lr0=exp["lr0"],
                optimizer=exp["optimizer"],
                imgsz=exp["imgsz"],
                batch=exp["batch"],
                mixup=exp["mixup"],
                project=BASE_DIR,
                name=name,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append({**exp, "mAP50": 0, "mAP50_95": 0, "status": str(e)})
            continue

        map50, map95 = read_best_map(os.path.join(BASE_DIR, name))
        print(f"  best mAP50={map50:.4f}  mAP50-95={map95:.4f}\n")
        summary.append({**exp, "mAP50": round(map50, 4), "mAP50_95": round(map95, 4), "status": "ok"})

    # Save report
    os.makedirs(BASE_DIR, exist_ok=True)
    report = os.path.join(BASE_DIR, "results.csv")
    with open(report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "="*65)
    print(f"{'EXPERIMENT':<25} {'LR':>7} {'OPT':<6} {'IMG':>4} {'mAP50':>7} {'mAP95':>7}")
    print("="*65)
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        print(f"{row['name']:<25} {row['lr0']:>7} {row['optimizer']:<6} {row['imgsz']:>4} {row['mAP50']:>7.4f} {row['mAP50_95']:>7.4f}")

    best = max(summary, key=lambda x: x["mAP50"])
    print(f"\nBest config: {best['name']}")
    print(f"  lr0={best['lr0']}  optimizer={best['optimizer']}  imgsz={best['imgsz']}  mAP50={best['mAP50']}")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    run()
