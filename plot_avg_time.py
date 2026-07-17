#!/usr/bin/env python3
import argparse, csv, os
# from click import style
import matplotlib.pyplot as plt

ALGO_COLOR = {
    "FedExpSls": "#080bb4",
    "FedAvg": "#006400",
    "FedExp": "#500C90E3",
    "FedExpProx": "#b51212",
    "FedSls": "#ff7f0e",
    "FedAdam": "#2E9787",
}

# ALGO_STYLE = {
#     "FedAvg": dict(linestyle="--", linewidth=2.6),
#     "FedExp": dict(linestyle="-",  linewidth=2.2),
# }

def read_avg_csv(path):
    xs, ys, stds = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)

        if "std" in r.fieldnames:
            for row in r:
                xs.append(float(row["time_sec"]))
                ys.append(float(row["mean"]))
                stds.append(float(row["std"]))
        else:
            for row in r:
                xs.append(float(row["time_sec"]))
                ys.append(float(row["value"]))
                stds.append(0.0)

    return xs, ys, stds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", action="append", required=True)  # NAME=CSV
    ap.add_argument("--title")
    ap.add_argument("--ylabel")
    ap.add_argument("--out")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--band", action="store_true")
    args = ap.parse_args()

    plt.figure(figsize=(8,6))

    for item in args.algo:
        name, path = item.split("=")
        xs, ys, stds = read_avg_csv(path)
        c = ALGO_COLOR.get(name)
        lw = 2.4 if name == "FedAvg" else 2.0

        # style = ALGO_STYLE.get(name, dict(linestyle="-", linewidth=2.0))

        # draw FedAvg on top so it doesn't get hidden when overlapping
        z = 10 if name == "FedAvg" else 6 if name == "FedExp" else 3
        if name == "FedAvg":
            ys_plot = [y + 1e-3 for y in ys]  # tiny visual offset
        else:
            ys_plot = ys

        plt.plot(xs, ys_plot, label=name, color=c, linewidth=2.2, zorder=z)
        # plt.plot(xs, ys, label=name, color=c, zorder=z, **style)
        # z  = 10  if name == "FedAvg" else 3
        # plt.plot(xs, ys, label=name, color=c, linewidth=2.2, zorder=z)
        #plt.plot(xs, ys, label=name, color=c, linewidth=lw, zorder=z)
        # plt.plot(xs, ys, label=name, color=c)
        # if args.band:
        #     lo = [m-s for m,s in zip(ys, stds)]
        #     hi = [m+s for m,s in zip(ys, stds)]
            # plt.fill_between(xs, lo, hi, alpha=0.15, color=c)

    plt.xlabel("Time (sec)", fontsize=12)
    plt.ylabel(args.ylabel, fontsize=12)
    plt.title(args.title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.pdf:
        plt.savefig(os.path.splitext(args.out)[0] + ".pdf")
    print("Saved:", args.out)

if __name__ == "__main__":
    main()