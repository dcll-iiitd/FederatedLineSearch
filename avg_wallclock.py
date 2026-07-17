#!/usr/bin/env python3
import argparse, glob, csv, math

def read_time_csv(path):
    xs, ys = [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["time_sec"]))
            ys.append(float(row["value"]))
    # ensure sorted
    pairs = sorted(zip(xs, ys), key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]

def step_value_at(times, values, t):
    # latest value with time <= t (step-hold)
    lo, hi = 0, len(times) - 1
    if not times or t < times[0]:
        return None
    # binary search upper bound
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid + 1
        else:
            hi = mid - 1
    return values[hi] if hi >= 0 else None

def mean_std(vals):
    m = sum(vals)/len(vals)
    if len(vals) == 1:
        return m, 0.0
    v = sum((x-m)**2 for x in vals)/(len(vals)-1)
    return m, math.sqrt(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="glob for seed csvs, e.g. wallclock_csv/FedAdam_seed*_time_train_loss.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=60.0, help="time grid step in seconds")
    ap.add_argument("--tmax", type=float, default=None, help="max time to plot/average")
    ap.add_argument("--min_seeds", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    series = [read_time_csv(f) for f in files]

    if not series:
        raise SystemExit("No files matched pattern")

    max_t = min(s[0][-1] for s in series if s[0])  # common coverage
    if args.tmax is not None:
        max_t = min(max_t, args.tmax)

    grid = []
    t = 0.0
    while t <= max_t:
        grid.append(t)
        t += args.dt

    rows = []
    for t in grid:
        vals = []
        for times, values in series:
            v = step_value_at(times, values, t)
            if v is not None:
                vals.append(v)
        if len(vals) >= args.min_seeds:
            mu, sd = mean_std(vals)
            rows.append((t, mu, sd, len(vals)))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "mean", "std", "n"])
        w.writerows(rows)

    print("Wrote:", args.out, "points=", len(rows))

if __name__ == "__main__":
    main()