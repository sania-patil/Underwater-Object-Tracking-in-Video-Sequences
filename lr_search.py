"""
Learning rate search for yolov8s on underwater dataset.
Tries 6 different lr0 values, 15 epochs each, finds the best.
"""

import torch, csv, os
from ultralytics import YOLO

DATASET_YAML = "/workspace/yolo_dataset/dataset.yaml"
OUTPUT_DIR   = "/workspace/runs/lr_search"

# LR values to try — reduced to 1 for debugging
LR_CONFIGS = [
    {"lr0": 0.001,  "lrf": 0.01},
]

FIXED = dict(
    data=DATASET_YAML,
    model="/workspace/yolov8s.pt",
    epochs=15,
    imgsz=640,
    batch=16,
    device=0,
    optimizer="AdamW",
    cos_lr=True,
    amp=True,
    patience=8,
    workers=2,
    cache=False,
    save=True,
    plots=False,
    verbose=True,
    exist_ok=True,   # allow re-running without name conflicts
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3, fliplr=0.5,
    mosaic=1.0, mixup=0.15,
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
    weight_decay=0.0005,
    warmup_epochs=3,
    momentum=0.937,
    close_mosaic=5,
)

def run():
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Running {len(LR_CONFIGS)} LR experiments\n")

    summary = []

    for cfg in LR_CONFIGS:
        name = f"lr_{str(cfg['lr0']).replace('.', 'p')}"
        print(f"Testing lr0={cfg['lr0']} ...")

        try:
            model = YOLO(FIXED["model"])
            results = model.train(
                **{k: v for k, v in FIXED.items() if k != "model"},
                **cfg,
                project=OUTPUT_DIR,
                name=name,
            )
            r = results.results_dict
            print(f"  results_dict keys: {list(r.keys())}")
            print(f"  results_dict values: {r}")
            row = {
                "lr0": cfg["lr0"],
                "mAP50": round(r.get("metrics/mAP50(B)", 0), 4),
                "mAP50_95": round(r.get("metrics/mAP50-95(B)", 0), 4),
                "precision": round(r.get("metrics/precision(B)", 0), 4),
                "recall": round(r.get("metrics/recall(B)", 0), 4),
                "status": "ok",
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = {"lr0": cfg["lr0"], "mAP50": 0, "mAP50_95": 0,
                   "precision": 0, "recall": 0, "status": str(e)}

        summary.append(row)
        print(f"  lr0={cfg['lr0']} -> mAP50={row['mAP50']}  status={row['status']}\n")

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = os.path.join(OUTPUT_DIR, "lr_results.csv")
    with open(report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "="*50)
    print("LR SEARCH RESULTS (sorted by mAP50)")
    print("="*50)
    for row in sorted(summary, key=lambda x: x["mAP50"], reverse=True):
        print(f"  lr0={row['lr0']:<8}  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50_95']:.4f}")

    best = max(summary, key=lambda x: x["mAP50"])
    print(f"\nBest lr0: {best['lr0']}  (mAP50={best['mAP50']})")
    print(f"Report: {report}")

if __name__ == "__main__":
    run()
