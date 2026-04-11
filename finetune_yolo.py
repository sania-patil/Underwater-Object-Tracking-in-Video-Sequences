"""
Fine-tunes YOLOv11s on UOT32 underwater dataset.
"""

import os, csv, torch
from ultralytics import YOLO

DATASET_YAML = "/workspace/yolo_dataset_sub/dataset.yaml"
MODEL        = "/workspace/yolo11s.pt"
OUTPUT_DIR   = "/workspace/runs"

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

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = YOLO(MODEL)

    model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        # Underwater augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,

        # Training
        amp=True,
        close_mosaic=15,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        workers=2,
        cache=False,
        verbose=True,

        project=OUTPUT_DIR,
        name="yolo11s_uot",
        exist_ok=False,
    )

    run_dir = os.path.join(OUTPUT_DIR, "yolo11s_uot")
    map50, map95 = read_best_map(run_dir)
    print(f"\nTraining complete.")
    print(f"Best mAP50:    {map50:.4f}")
    print(f"Best mAP50-95: {map95:.4f}")
    print(f"Best weights:  {run_dir}/weights/best.pt")
