"""Estimate the GPU energy used by a run from an nvidia-smi power log, and log the run.

train.slurm samples the GPU power every few seconds while training runs. This script
integrates that log into kWh, converts it to an EV driving distance, and appends one
row per run to experiments.csv (used for the sustainability section).

Only the GPU is measured: CPU, RAM and cooling are not included. --pue scales the
number up to account for data-centre overhead if you want a fuller estimate.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

EV_KWH_PER_KM = 0.18  # typical electric car, ~18 kWh/100 km


def read_power_log(path):
    times, watts, names = [], [], set()
    for row in csv.reader(open(path)):
        if len(row) < 3:
            continue
        try:
            t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            w = float(row[2])
        except ValueError:  # header lines or "[N/A]" readings
            continue
        times.append(t)
        watts.append(w)
        names.add(row[1].strip())
    return times, watts, names


def energy_kwh(times, watts):
    """Trapezoidal integration of power over time."""
    joules = sum((t1 - t0).total_seconds() * (w0 + w1) / 2
                 for t0, t1, w0, w1 in zip(times, times[1:], watts, watts[1:]))
    return joules / 3.6e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="nvidia-smi CSV written by train.slurm")
    p.add_argument("--name", required=True)
    p.add_argument("--phase", default="train", help="train / eval / inference")
    p.add_argument("--notes", default="")
    p.add_argument("--pue", type=float, default=1.0)
    p.add_argument("--experiments", default=None, help="CSV to append to (default: next to the log)")
    args = p.parse_args()

    times, watts, names = read_power_log(args.log)
    if len(times) < 2:
        raise SystemExit(f"Not enough samples in {args.log}")
    hours = (times[-1] - times[0]).total_seconds() / 3600
    # Assumes one GPU per job (the log would interleave rows from several GPUs).
    kwh = energy_kwh(times, watts) * args.pue
    row = {
        "name": args.name, "phase": args.phase, "gpu": "+".join(sorted(names)),
        "start": times[0].isoformat(timespec="seconds"), "hours": round(hours, 3),
        "mean_watt": round(sum(watts) / len(watts), 1), "kwh": round(kwh, 3),
        "ev_km": round(kwh / EV_KWH_PER_KM, 1), "pue": args.pue, "notes": args.notes,
    }

    exp = Path(args.experiments or Path(args.log).parent / "experiments.csv")
    new = not exp.exists()
    with open(exp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{args.name} [{args.phase}]: {hours:.2f} h on {row['gpu']}, "
          f"{kwh:.2f} kWh ~ {row['ev_km']} km in an EV  -> logged to {exp}")


if __name__ == "__main__":
    main()
