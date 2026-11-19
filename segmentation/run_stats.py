"""One row per training run: speed, GPU utilisation, power and energy (GPU only).

    python3 -m segmentation.run_stats                      # runs in $OUT_ROOT or /cluster/work/$USER/tdt17/runs
    python3 -m segmentation.run_stats path/to/runs

Power/utilisation come from the power_<jobid>.csv written by slurm/seg_train.slurm, or for non-slurm runs the
log scripts/train_gpu.sh writes (path in summary.json). Sampled every 30 s.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def power_stats(path):
    """Mean GPU utilisation (%) and power (W) from an nvidia-smi CSV log."""
    util, watts = [], []
    with open(path) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                util.append(float(row["utilization.gpu [%]"].split()[0]))
                watts.append(float(row["power.draw [W]"].split()[0]))
            except (KeyError, ValueError, IndexError):
                continue  # header repeats or "[N/A]"
    return (np.mean(util), np.mean(watts)) if watts else (None, None)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_ROOT", f"/cluster/work/{os.environ['USER']}/tdt17/runs"))
    rows = []
    for run in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        epochs = list(csv.DictReader(open(run / "metrics.csv")))
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        secs = [float(e["seconds"]) for e in epochs]
        job = summary.get("slurm_job_id")
        log = root / f"power_{job}.csv" if job else Path(summary["power_log"]) if summary.get("power_log") else None
        util, watts = power_stats(log) if log and log.exists() else (None, None)
        train_s = summary.get("train_seconds", sum(secs))
        best = max(epochs, key=lambda e: float(e["val_iou"])) if epochs else None
        if "best_epoch" in summary:  # the saved checkpoint (projection runs keep the last epoch, not the best)
            best = epochs[summary["best_epoch"] - 1]
        rows.append({
            "run": run.name, "job": job or "-", "workers": cfg["num_workers"], "epochs": len(epochs),
            "s/epoch": np.mean(secs[1:] or secs),  # skip epoch 1 (start-up)
            "train_min": train_s / 60, "gpu_util%": util, "gpu_W": watts,
            "gpu_kWh": watts * train_s / 3.6e6 if watts else None,
            "best_ep": best["epoch"] if best else "-", "best_val_iou": float(best["val_iou"]) if best else None,
            "done": "yes" if summary else "running/failed",
        })

    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v < 1 else f"{v:.1f}" if isinstance(v, float) else str(v if v is not None else "-")
    cols = list(rows[0]) if rows else []
    widths = [max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


if __name__ == "__main__":
    main()
