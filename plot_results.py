"""
Generate result graphs for the underwater object tracking evaluation.
Saves all plots to runs/tracking/plots/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "runs/tracking/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACKERS = ["SORT", "DeepSORT", "FairMOT"]
COLORS   = ["#2196F3", "#FF5722", "#4CAF50"]

# ── Per-sequence data ──────────────────────────────────────────────────────────
SEQUENCES = ["BlueFish2","BoySwimming","Dolphin2","Fisherman",
             "HoverFish2","SeaDiver","SeaTurtle2","SeaTurtle3"]

MOTA = {
    "SORT":     [-0.371, -0.992,  0.462,  0.335,  0.421,  0.244,  0.266,  0.416],
    "DeepSORT": [-4.976, -0.997, -0.213, -0.141, -0.125, -1.304,  0.021, -0.436],
    "FairMOT":  [-1.157, -0.969,  0.321,  0.306,  0.149, -0.258,  0.317,  0.397],
}
MOTP = {
    "SORT":     [0.280, 0.000, 0.238, 0.212, 0.214, 0.241, 0.382, 0.287],
    "DeepSORT": [0.344, 0.000, 0.256, 0.302, 0.262, 0.251, 0.380, 0.300],
    "FairMOT":  [0.333, 0.000, 0.253, 0.272, 0.264, 0.245, 0.390, 0.294],
}
ID_SW = {
    "SORT":     [9,  0,  9,  9,  5, 13,  8, 15],
    "DeepSORT": [12, 0,  2,  7,  6,  3,  3, 10],
    "FairMOT":  [4,  0,  2,  9,  4,  2,  2,  5],
}
AGGREGATE = {
    "SORT":     {"mota":  0.099, "motp": 0.239, "id_sw": 68},
    "DeepSORT": {"mota": -0.993, "motp": 0.267, "id_sw": 43},
    "FairMOT":  {"mota": -0.095, "motp": 0.261, "id_sw": 28},
}
SINGLE_MULTI = {
    "SORT":     {"single": -0.010, "multi":  0.202},
    "DeepSORT": {"single": -0.740, "multi": -1.230},
    "FairMOT":  {"single": -0.194, "multi": -0.003},
}

x = np.arange(len(SEQUENCES))
w = 0.25

# ── 1. Per-sequence MOTA bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for i, (t, c) in enumerate(zip(TRACKERS, COLORS)):
    ax.bar(x + i*w, MOTA[t], w, label=t, color=c, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x + w)
ax.set_xticklabels(SEQUENCES, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("MOTA")
ax.set_title("Per-Sequence MOTA Comparison (SORT vs DeepSORT vs FairMOT)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_per_sequence_mota.png", dpi=150)
plt.close()

# ── 2. Per-sequence MOTP bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for i, (t, c) in enumerate(zip(TRACKERS, COLORS)):
    ax.bar(x + i*w, MOTP[t], w, label=t, color=c, alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(SEQUENCES, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("MOTP")
ax.set_title("Per-Sequence MOTP Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_per_sequence_motp.png", dpi=150)
plt.close()

# ── 3. Per-sequence ID switches bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for i, (t, c) in enumerate(zip(TRACKERS, COLORS)):
    ax.bar(x + i*w, ID_SW[t], w, label=t, color=c, alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(SEQUENCES, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("ID Switches")
ax.set_title("Per-Sequence ID Switches Comparison (lower is better)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_per_sequence_id_switches.png", dpi=150)
plt.close()

# ── 4. Aggregate comparison (grouped bar) ─────────────────────────────────────
metrics = ["MOTA", "MOTP", "ID Switches (÷10)"]
vals = {
    "SORT":     [AGGREGATE["SORT"]["mota"],     AGGREGATE["SORT"]["motp"],     AGGREGATE["SORT"]["id_sw"]/10],
    "DeepSORT": [AGGREGATE["DeepSORT"]["mota"], AGGREGATE["DeepSORT"]["motp"], AGGREGATE["DeepSORT"]["id_sw"]/10],
    "FairMOT":  [AGGREGATE["FairMOT"]["mota"],  AGGREGATE["FairMOT"]["motp"],  AGGREGATE["FairMOT"]["id_sw"]/10],
}
xm = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(9, 6))
for i, (t, c) in enumerate(zip(TRACKERS, COLORS)):
    ax.bar(xm + i*0.25, vals[t], 0.25, label=t, color=c, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(xm + 0.25)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_title("Aggregate Tracker Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_aggregate_comparison.png", dpi=150)
plt.close()

# ── 5. Single vs Multi-object MOTA ────────────────────────────────────────────
cats = ["Single Object", "Multi Object"]
xc = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(8, 6))
for i, (t, c) in enumerate(zip(TRACKERS, COLORS)):
    vals_sm = [SINGLE_MULTI[t]["single"], SINGLE_MULTI[t]["multi"]]
    ax.bar(xc + i*0.25, vals_sm, 0.25, label=t, color=c, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(xc + 0.25)
ax.set_xticklabels(cats, fontsize=12)
ax.set_ylabel("MOTA")
ax.set_title("Single vs Multi-Object Tracking MOTA")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_single_vs_multi_mota.png", dpi=150)
plt.close()

# ── 6. ID switches comparison (horizontal bar) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
yt = np.arange(len(TRACKERS))
id_totals = [AGGREGATE[t]["id_sw"] for t in TRACKERS]
bars = ax.barh(yt, id_totals, color=COLORS, alpha=0.85)
ax.set_yticks(yt)
ax.set_yticklabels(TRACKERS, fontsize=12)
ax.set_xlabel("Total ID Switches (lower is better)")
ax.set_title("Total ID Switches per Tracker")
for bar, val in zip(bars, id_totals):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/6_id_switches_total.png", dpi=150)
plt.close()

print("All plots saved to:", OUTPUT_DIR)
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
