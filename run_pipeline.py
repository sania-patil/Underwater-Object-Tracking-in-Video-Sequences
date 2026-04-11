"""Entry-point script for the underwater object tracking pipeline."""

import argparse
import csv
import json
import logging
import os

from src.config_loader import ConfigLoader
from src.evaluator import Evaluator
from src.pipeline import Pipeline
from src.sequence_loader import load_sequences
from src.trackers.deepsort_tracker import DeepSORTTracker
from src.trackers.fairmot_tracker import FairMOTTracker
from src.trackers.sort_tracker import SORTTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VAL_IMAGES = "yolo_dataset/images/val"
VAL_LABELS = "yolo_dataset/labels/val"
CHECKPOINT  = "runs/compare/yolo11m_sub5/weights/best.pt"

# Sequences with typically 1 object vs multiple objects
SINGLE_OBJECT_SEQS  = {"Dolphin2", "Fisherman", "BoySwimming", "SeaDiver"}
MULTI_OBJECT_SEQS   = {"BlueFish2", "HoverFish2", "SeaTurtle2", "SeaTurtle3"}


def analyse_single_vs_multi(all_metrics: dict, output_path: str) -> None:
    """Compare tracker performance on single-object vs multi-object sequences."""
    rows = []
    for tracker_name, seq_metrics in all_metrics.items():
        for category, seq_set in [("single_object", SINGLE_OBJECT_SEQS),
                                   ("multi_object",  MULTI_OBJECT_SEQS)]:
            cat_metrics = {k: v for k, v in seq_metrics.items() if k in seq_set}
            if not cat_metrics:
                continue
            total_frames = sum(m.num_frames for m in cat_metrics.values())
            total_ids    = sum(m.id_switches for m in cat_metrics.values())
            avg_mota = (sum(m.mota * m.num_frames for m in cat_metrics.values())
                        / total_frames if total_frames else 0)
            avg_motp = (sum(m.motp * m.num_frames for m in cat_metrics.values())
                        / total_frames if total_frames else 0)
            rows.append({
                "tracker":    tracker_name,
                "category":   category,
                "sequences":  len(cat_metrics),
                "mota":       round(avg_mota, 4),
                "motp":       round(avg_motp, 4),
                "id_switches": total_ids,
                "num_frames": total_frames,
            })

    csv_path = f"{output_path}_single_vs_multi.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Single vs multi-object analysis written to %s", csv_path)

    # Print table
    print("\n" + "="*70)
    print("SINGLE vs MULTI-OBJECT TRACKING COMPARISON")
    print("="*70)
    print(f"{'Tracker':<12} {'Category':<16} {'MOTA':>7} {'MOTP':>7} {'ID_SW':>6} {'Frames':>7}")
    print("-"*70)
    for r in rows:
        print(f"{r['tracker']:<12} {r['category']:<16} {r['mota']:>7.4f} {r['motp']:>7.4f} "
              f"{r['id_switches']:>6} {r['num_frames']:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwater Object Tracking Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip tracking, only run evaluation on existing results")
    args = parser.parse_args()

    loader = ConfigLoader()
    cfg = loader.load(args.config)
    cfg.fine_tuned_checkpoint = CHECKPOINT

    sequences = load_sequences(VAL_IMAGES)
    if not sequences:
        logger.error("No sequences found in %s", VAL_IMAGES)
        return

    logger.info("Found %d val sequences: %s", len(sequences), sorted(sequences.keys()))

    trackers = {
        "sort": SORTTracker(
            max_age=cfg.sort_max_age,
            min_hits=cfg.sort_min_hits,
            iou_threshold=cfg.sort_iou_threshold,
        ),
        "deepsort": DeepSORTTracker(
            max_age=cfg.deepsort_max_age,
            nn_budget=cfg.deepsort_nn_budget,
            max_cosine_distance=cfg.deepsort_max_cosine_distance,
            reid_model_path=cfg.deepsort_reid_model,
        ),
        "fairmot": FairMOTTracker(
            max_age=20,
            nn_budget=200,
            max_cosine_distance=0.2,
        ),
    }

    all_metrics = {}

    for tracker_name, tracker in trackers.items():
        logger.info("=== Running tracker: %s ===", tracker_name)

        if not args.skip_tracking:
            pipeline = Pipeline(cfg, tracker)
            for seq_name in sorted(sequences.keys()):
                logger.info("Processing sequence: %s", seq_name)
                try:
                    pipeline.run(seq_name, VAL_IMAGES)
                except Exception as e:
                    logger.error("Error processing %s with %s: %s", seq_name, tracker_name, e)

        results_dir = os.path.join(cfg.output_dir, "results", tracker_name)
        evaluator = Evaluator()
        seq_metrics = evaluator.evaluate_tracker(
            tracker_name=tracker_name,
            results_dir=results_dir,
            gt_dir=VAL_LABELS,
            iou_threshold=cfg.eval_iou_threshold,
        )
        all_metrics[tracker_name] = seq_metrics

        agg = evaluator.aggregate(seq_metrics)
        logger.info(
            "%s aggregate — MOTA: %.3f, MOTP: %.3f, ID switches: %d",
            tracker_name, agg.mota, agg.motp, agg.total_id_switches,
        )

    # Write main comparison report
    evaluator = Evaluator()
    report_path = cfg.eval_output_path
    evaluator.write_report(all_metrics, report_path)
    logger.info("Comparison report: %s.csv / %s.json", report_path, report_path)

    # Single vs multi-object analysis
    analyse_single_vs_multi(all_metrics, report_path)

    # Print final summary
    print("\n" + "="*60)
    print("TRACKER COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Tracker':<12} {'MOTA':>8} {'MOTP':>8} {'ID_SW':>6}")
    print("-"*60)
    for tracker_name, seq_metrics in all_metrics.items():
        agg = evaluator.aggregate(seq_metrics)
        print(f"{tracker_name:<12} {agg.mota:>8.4f} {agg.motp:>8.4f} {agg.total_id_switches:>6}")


if __name__ == "__main__":
    main()
