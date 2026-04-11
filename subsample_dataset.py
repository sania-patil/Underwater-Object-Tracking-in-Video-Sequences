"""
Subsamples video frames to reduce redundancy.
Keeps every Nth frame per sequence for both train and val splits.
"""

import os, shutil
from pathlib import Path
from collections import defaultdict

KEEP_EVERY_N = 3   # every 3rd frame ~6900 train images

SPLITS = [
    {
        "src_images": r"C:\DL-fa\yolo_dataset\images\train",
        "src_labels": r"C:\DL-fa\yolo_dataset\labels\train",
        "dst_images": r"C:\DL-fa\yolo_dataset_sub3\images\train",
        "dst_labels": r"C:\DL-fa\yolo_dataset_sub3\labels\train",
    },
    {
        "src_images": r"C:\DL-fa\yolo_dataset\images\val",
        "src_labels": r"C:\DL-fa\yolo_dataset\labels\val",
        "dst_images": r"C:\DL-fa\yolo_dataset_sub3\images\val",
        "dst_labels": r"C:\DL-fa\yolo_dataset_sub3\labels\val",
    },
]

def subsample_split(src_images, src_labels, dst_images, dst_labels, split_name):
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    sequences = defaultdict(list)
    for f in Path(src_images).glob("*.jpg"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                sequences[parts[0]].append((int(parts[1]), f))
            except ValueError:
                continue

    total_src, total_kept = 0, 0
    for frames in sequences.values():
        frames.sort(key=lambda x: x[0])
        for i, (_, img_path) in enumerate(frames):
            total_src += 1
            if i % KEEP_EVERY_N != 0:
                continue
            shutil.copy2(img_path, Path(dst_images) / img_path.name)
            lbl = Path(src_labels) / (img_path.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, Path(dst_labels) / lbl.name)
            total_kept += 1

    print(f"[{split_name}] {total_src} -> {total_kept} frames (every {KEEP_EVERY_N}rd, {100*(1-total_kept/total_src):.1f}% reduction)")

if __name__ == "__main__":
    for split in SPLITS:
        subsample_split(**split, split_name=split["src_images"].split("\\")[-1])

    # Write dataset.yaml with Docker-compatible path
    yaml = """path: /workspace/yolo_dataset_sub3
train: images/train
val: images/val

nc: 1
names: ['underwater_object']
"""
    out = r"C:\DL-fa\yolo_dataset_sub3\dataset.yaml"
    os.makedirs(r"C:\DL-fa\yolo_dataset_sub3", exist_ok=True)
    with open(out, "w") as f:
        f.write(yaml)
    print(f"\ndataset.yaml written to {out}")
