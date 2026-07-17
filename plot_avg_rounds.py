#!/usr/bin/env python3
import argparse, csv, os
import matplotlib as mpl
import matplotlib.pyplot as plt


ALGO_COLOR = {
    "FedExpSls": "#0000FF",
    "FedAvg": "#008000",
    "FedExp": "#800080",
    "FedExpProx": "#FF0000",
    "FedSls": "#D2681D",
    "FedAdam": "#32A2A2",
}

FIG_W_PX, FIG_H_PX = 395*2, 316*2
DPI = 96*2  # must be 96 to match your properties

def read_avg_csv(path):
    xs, ys, stds = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["round"]))
            ys.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return xs, ys, stds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", action="append", required=True)  # NAME=CSV
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", choices=["loss", "test_acc"], required=True)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    # --- lock pixel size exactly (395x316 @ 96 dpi) ---
    fig, ax = plt.subplots(figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- grid and spines like default look ---
    ax.grid(True)  # use mpl defaults (matches your target better than custom alpha/linewidth)
    # for spine in ax.spines.values():
    #     spine.set_color("black")
    #     spine.set_linewidth(1.0)

    # --- axes box (spines) ---
    for spine in ax.spines.values():
        spine.set_color("0.25")   # darker than 0.6 (matches the other panels visually)
        spine.set_linewidth(1.0)  # slightly thicker
        
    # --- plot ---
    for item in args.algo:
        name, path = item.split("=")
        xs, ys, _ = read_avg_csv(path)
        ax.plot(xs, ys, label=name, color=ALGO_COLOR[name], linewidth=1.5)


    # legend placement per metric
    legend_loc = "upper right" if args.metric == "loss" else "lower right"

    # --- IMPORTANT: keep legend box, keep DEFAULT dimensions ---
    # So: don't set fontsize/handlelength/padding/etc.
    # leg = ax.legend(loc=legend_loc, frameon=True)

    # # Make the box subtle like your image (keep box, no harsh black border)
    # leg.get_frame().set_facecolor("white")
    # leg.get_frame().set_alpha(0.8)          # close to mpl default look
    # leg.get_frame().set_edgecolor("0.8")    # light grey border
    # leg.get_frame().set_linewidth(1.0)

    leg = ax.legend(loc=legend_loc, frameon=True, fontsize=12)  # increase legend font


    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("0.6")   # light gray border
    leg.get_frame().set_linewidth(0.8)
    # --- CRITICAL: do NOT use tight_layout or bbox_inches="tight" ---
    fig.savefig(args.out, dpi=DPI, facecolor="white")
    if args.pdf:
        pdf_out = os.path.splitext(args.out)[0] + ".pdf"
        fig.savefig(
            pdf_out,
            facecolor="white",
            bbox_inches="tight",   # crop the PDF page to the content
            pad_inches=0.1        # tiny padding; use 0.0 if you want zero border
        )

    print(f"Saved: {args.out}  ({FIG_W_PX}x{FIG_H_PX} @ {DPI} dpi)")

if __name__ == "__main__":
    main()