"""Collect all detection results into presentation-ready tables and figures.

Reads the folders copied from IDUN (runs, eval, error_analysis, energy) and writes to <results>/summary:
  summary_table.csv / .md      one row per model: time, energy, speed, mAP overall / Norway / NRD
  per_class_ap50.csv / .md     test AP50 per class and model
  map50_per_set.png            mAP50 per country and on NRD, all models
  found_by_size.png            share of boxes found by size (same boxes, sizes at 640 px), per country
  error_types.png              missed / boxed differently / wrong class / correct, per model
  energy_vs_map.png            training energy against mAP50 (all, Norway, NRD)
  training_curves.png          val mAP50 per epoch, with the best epoch marked

Runs on a laptop in seconds:
    python detection/summarize.py --results ../results
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["yolo11s_640", "yolo26s_640", "yolo11s_1280"]
MODEL_NAMES = {"yolo11s_640": "YOLO11s @ 640 (baseline)", "yolo26s_640": "YOLO26s @ 640",
               "yolo11s_1280": "YOLO11s @ 1280"}
MODEL_COLORS = {"yolo11s_640": "#2a78d6", "yolo26s_640": "#eb6834", "yolo11s_1280": "#1baf7a"}
OUTCOMES = ["correct", "wrong class", "boxed differently", "missed"]
OUTCOME_COLORS = {"correct": "#1baf7a", "wrong class": "#eda100", "boxed differently": "#e87ba4", "missed": "#b5b4ad"}
SETS = {"rdd_test": "RDD test (all)", "rdd_test_China_MotorBike": "China_MotorBike", "rdd_test_United_States": "United_States",
        "rdd_test_Japan": "Japan", "rdd_test_Czech": "Czech", "rdd_test_India": "India",
        "rdd_test_Norway": "Norway (RDD)", "nrd_agnostic": "NRD (customer, class-agnostic)"}
SIZE_BINS = [0, 8, 16, 32, 64, 1e9]
SIZE_LABELS = ["<8", "8-16", "16-32", "32-64", ">=64"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def imgsz_of(model):
    return int(model.rsplit("_", 1)[1])


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print(f"  saved {name}")


def plot_map_per_set(ev, out):
    sets = [s for s in SETS if s in ev["set"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.8 / len(MODELS)
    y = np.arange(len(sets))[::-1]
    for i, m in enumerate(MODELS):
        vals = ev[ev["model"] == m].set_index("set").reindex(sets)["mAP50"].values
        ax.barh(y + (len(MODELS) / 2 - i - 0.5) * h, vals, height=h, color=MODEL_COLORS[m],
                label=MODEL_NAMES[m], edgecolor="white", linewidth=1)
        if m == "yolo11s_1280":  # label only the model the story is about
            for yy, v in zip(y + (len(MODELS) / 2 - i - 0.5) * h, vals):
                ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [SETS[s] for s in sets])
    ax.set_xlim(0, 1)
    ax.set_xlabel("mAP50 on the test split")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out, "map50_per_set.png")


def plot_found_by_size(ea, out):
    countries = sorted(ea["country"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for ax, c in zip(axes.ravel(), countries):
        for m in MODELS:
            g = ea[(ea["model"] == m) & (ea["country"] == c)]
            rate = g.groupby("size640", observed=False)["found"].mean().reindex(SIZE_LABELS)
            ax.plot(SIZE_LABELS, rate.values, marker="o", markersize=6, linewidth=2,
                    color=MODEL_COLORS[m], label=MODEL_NAMES[m])
        n = ea[(ea["model"] == MODELS[0]) & (ea["country"] == c)].groupby("size640", observed=False).size().reindex(SIZE_LABELS)
        ax.set_title(f"{c}   (boxes per bin: {', '.join(str(int(v)) for v in n.values)})", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Shortest box side, px at 640 input", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share found (IoU >= 0.5)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Same ground-truth boxes for every model, sizes measured at 640 px", fontsize=9, y=1.03)
    save(fig, out, "found_by_size.png")


def plot_error_types(ea, out):
    groups = [("All countries", ea), ("Norway (RDD)", ea[ea["country"] == "Norway"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
    for ax, (title, df) in zip(axes, groups):
        share = pd.crosstab(df["model"], df["outcome"], normalize="index").reindex(index=MODELS[::-1], columns=OUTCOMES, fill_value=0)
        left = np.zeros(len(share))
        for o in OUTCOMES:
            ax.barh([MODEL_NAMES[m] for m in share.index], share[o], left=left, color=OUTCOME_COLORS[o],
                    label=o, height=0.6, edgecolor="white", linewidth=2)
            for yy, (l, v) in enumerate(zip(left, share[o])):
                if v > 0.08:
                    ax.text(l + v / 2, yy, f"{v:.0%}", ha="center", va="center", fontsize=8, color=INK)
            left += share[o].values
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of ground-truth boxes (conf >= 0.25)")
    axes[1].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(-0.1, 1.3))
    save(fig, out, "error_types.png")


def plot_energy_vs_map(table, out):
    panels = [("mAP50_all", "RDD test (all)"), ("mAP50_Norway", "Norway (RDD)"), ("mAP50_NRD", "NRD (customer)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, (col, title) in zip(axes, panels):
        for m, r in table.iterrows():
            ax.scatter(r["train_kwh"], r[col], s=80, color=MODEL_COLORS[m], edgecolor="white", linewidth=2, zorder=3)
            rightmost = r["train_kwh"] == table["train_kwh"].max()
            # The two 640 models sit close together: put the lower one's label below its point.
            other = {"yolo11s_640": "yolo26s_640", "yolo26s_640": "yolo11s_640"}.get(m)
            below = other is not None and r[col] < table.loc[other, col]
            ax.annotate(MODEL_NAMES[m].split(" (")[0], (r["train_kwh"], r[col]),
                        xytext=(-8, 8) if rightmost else (8, -12 if below else 6),
                        ha="right" if rightmost else "left", textcoords="offset points", fontsize=7, color=MUTED)
        lo, hi = table[col].min(), table[col].max()
        ax.set_ylim(lo - 0.05, hi + 0.05)
        ax.set_xlim(0, table["train_kwh"].max() * 1.15)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training energy, GPU only (kWh)")
    axes[0].set_ylabel("mAP50 (test)")
    save(fig, out, "energy_vs_map.png")


def plot_training_curves(results, out):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for m in MODELS:
        f = results / m / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], linewidth=2, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
        best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]  # best.pt is chosen mainly on mAP50-95
        ax.scatter(best["epoch"], best["metrics/mAP50(B)"], s=60, color=MODEL_COLORS[m], edgecolor="white", linewidth=2, zorder=3)
    ax.axvspan(90, 100, color=GRID, alpha=0.6, linewidth=0)
    ax.text(95, 0.15, "no mosaic\n(epochs 91-100)", fontsize=7, color=MUTED, ha="center")
    ax.set_xlabel("Epoch (training stopped after 20 epochs without improvement; dot = best epoch)")
    ax.set_ylabel("Val mAP50")
    ax.legend(fontsize=8, loc="lower center")
    save(fig, out, "training_curves.png")


def to_markdown(df):
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="folder with runs, eval, error_analysis (and energy) from IDUN")
    args = p.parse_args()
    results = Path(args.results)
    out = results / "summary"
    out.mkdir(exist_ok=True)

    ev = pd.read_csv(results / "eval" / "results.csv").drop_duplicates(["model", "set"], keep="last")
    ev = ev[ev["model"].isin(MODELS)]

    ea = []
    for m in MODELS:
        f = results / "error_analysis" / m / "gt_outcomes.csv"
        if f.exists():
            ea.append(pd.read_csv(f).assign(model=m))
    ea = pd.concat(ea, ignore_index=True)
    ea["found"] = ea["best_iou"] >= 0.5
    # Rescale box sizes to 640 px input, so every model is compared on the same boxes in the same bins.
    ea["short640"] = ea["short_px"] * 640 / ea["model"].map(imgsz_of)
    ea["size640"] = pd.cut(ea["short640"], SIZE_BINS, labels=SIZE_LABELS, right=False)

    # Summary table
    piv = ev.pivot(index="model", columns="set", values="mAP50")
    table = pd.DataFrame(index=MODELS)
    energy_file = results / "energy" / "experiments.csv"
    if energy_file.exists():
        en = pd.read_csv(energy_file)
        en = en[en["phase"] == "train"].drop_duplicates("name", keep="last").set_index("name")
        table["train_hours"] = en["hours"].round(2)
        table["train_kwh"] = en["kwh"].round(2)
        table["ev_km"] = en["ev_km"]
    else:
        print(f"  ({energy_file} not found: energy columns and energy_vs_map.png skipped)")
    rdd = ev[ev["set"] == "rdd_test"].set_index("model")
    # No inference time here: eval jobs may land on different GPU types (the 1280 eval ran on an H100),
    # so their timings are not comparable. Use the A100 timings from the training logs instead.
    table["mAP50_all"] = piv["rdd_test"].round(3)
    table["mAP50-95_all"] = rdd["mAP50-95"].round(3)
    table["recall_all"] = rdd["R"].round(3)
    table["mAP50_Norway"] = piv["rdd_test_Norway"].round(3)
    table["recall_Norway"] = ev[ev["set"] == "rdd_test_Norway"].set_index("model")["R"].round(3)
    table["mAP50_NRD"] = piv["nrd_agnostic"].round(3)
    table["recall_NRD"] = ev[ev["set"] == "nrd_agnostic"].set_index("model")["R"].round(3)
    table.index.name = "model"
    table.to_csv(out / "summary_table.csv")
    (out / "summary_table.md").write_text(to_markdown(table))

    per_class = rdd[[c for c in rdd.columns if c.startswith("AP50_")]].reindex(MODELS).round(3)
    per_class.to_csv(out / "per_class_ap50.csv")
    (out / "per_class_ap50.md").write_text(to_markdown(per_class))

    print("Figures:")
    plot_map_per_set(ev, out)
    plot_found_by_size(ea, out)
    plot_error_types(ea, out)
    if "train_kwh" in table:
        plot_energy_vs_map(table, out)
    plot_training_curves(results, out)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\nSummary:\n", table)
    print("\nTest AP50 per class:\n", per_class)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
