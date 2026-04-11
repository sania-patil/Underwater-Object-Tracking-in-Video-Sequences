"""
Extended experiments to improve accuracy beyond mAP50=0.504.
Runs 4 experiments sequentially and compares results.
"""

import os, csv, torch
from ultralytics import YOLO

BEST_PT       = "/workspace/runs/compare/yolo11m_sub5/weights/best.pt"
DATASET_SUB3  = "/workspace/yolo_dataset_sub3/dataset.yaml"   # every 3rd frame
DATASET_SUB5  = "/workspace/yolo_dataset_sub/dataset.yaml"    # every 5th frame
OUTPUT_DIR    = "/workspace/runs/experiments2"

EXPERIMENTS = [
    {
        "name": "A_retrain_sub3",
        "desc": "Resume best.pt on sub3 dataset (every 3rd frame)",
        "model": BEST_PT,
        "data": DATASET_SUB3,
        "epochs": 100, "batch": 12, "imgsz": 640,
        "lr0": 0.0003, "patience": 30, "freeze": None,
    },
    {
        "name": "B_highres_832",
        "desc": "Fresh yolo11m at 832px resolution",
        "model": BEST_PT,
        "data": DATASET_SUB3,
        "epochs": 100, "batch": 6, "imgsz": 832,
        "lr0": 0.0003, "patience": 30, "freeze": None,
    },
    {
        "name": "C_freeze_backbone",
        "desc": "Freeze backbone (layers 0-10), train head only",
        "model": BEST_PT,
        "data": DATASET_SUB3,
        "epochs": 100, "batch": 12, "imgsz": 640,
        "lr0": 0.001, "patience": 30, "freeze": 10,
    },
    {
        "name": "D_lowlr_long",
        "desc": "Very low LR, long training for fine convergence",
        "model": BEST_PT,
        "data": DATASET_SUB3,
        "epochs": 150, "batch": 12, "imgsz": 640,
        "lr0": 0.0001, "patience": 40, "freeze": None,
    },
]

COMMON = dict(
    device=0,
    optimizer="AdamW",
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    mosaic=1.0, mixup=0.15,
    copy_paste=0.1,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
    amp=False,
    close_mosaic=15,
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
    print(f"Baseline: yolo11m mAP50=0.504")
    print(f"Running {len(EXPERIMENTS)} experiments\n")

    summary = []

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(EXPERIMENTS)}] {exp['name']}")
        print(f"  {exp['desc']}")
        print(f"  lr0={exp['lr0']}  imgsz={exp['imgsz']}  batch={exp['batch']}  freeze={exp['freeze']}")
        print(f"{'='*60}")

        try:
            model = YOLO(exp["model"])
            train_kwargs = dict(
                **COMMON,
                data=exp["data"],
                epochs=exp["epochs"],
                batch=exp["batch"],
                imgsz=exp["imgsz"],
                lr0=exp["lr0"],
                patience=exp["patience"],
                project=OUTPUT_DIR,
                name=exp["name"],
            )
            if exp["freeze"] is not None:
                train_kwargs["freeze"] = exp["freeze"]
            model.train(**train_kwargs)
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append({**exp, "mAP50": 0, "mAP50_95": 0, "status": str(e)})
            continue

        run_dir = os.path.join(OUTPUT_DIR, exp["name"])
        map50, map95 = read_best_map(run_dir)
        print(f"\n  Result: mAP50={map50:.4f}  mAP50-95={map95:.4f}")
        summary.append({
            "name": exp["name"],
            "desc": exp["desc"],
            "lr0": exp["lr0"],
            "imgsz": exp["imgsz"],
            "mAP50": round(map50, 4),
            "mAP50_95": round(map95, 4),
            "weights": os.path.join(run_dir, "weights", "best.pt"),
            "status": "ok",
        })

    # Save + print report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = os.path.join(OUTPUT_DIR, "results.csv")
    with open(report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*60}")
    print(f"RESULTS (baseline mAP50=0.504)")
    print(f"{'='*60}")
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        delta = row["mAP50"] - 0.504
        sign = "+" if delta >= 0 else ""
        print(f"{row['name']:<25}  mAP50={row['mAP50']:.4f} ({sign}{delta:.4f})  status={row['status']}")

    best = max(summary, key=lambda x: x["mAP50"])
    print(f"\nBest: {best['name']}  mAP50={best['mAP50']}")
    print(f"Use: {best.get('weights', 'N/A')}")
    print(f"Report: {report}")


if __name__ == "__main__":
    run()
