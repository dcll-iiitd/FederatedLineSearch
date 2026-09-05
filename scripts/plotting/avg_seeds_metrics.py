#!/usr/bin/env python3
import argparse, glob, csv, math

def read_csv(path):
    d = {}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            d[int(float(row["round"]))] = float(row["value"])
    return d

def mean_std(xs):
    m = sum(xs)/len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x-m)**2 for x in xs)/(len(xs)-1)
    return m, math.sqrt(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_seeds", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    maps = [read_csv(f) for f in files]

    rounds = sorted(set().union(*[m.keys() for m in maps]))
    rows = []

    for r in rounds:
        vals = [m[r] for m in maps if r in m]
        if len(vals) >= args.min_seeds:
            mu, sd = mean_std(vals)
            rows.append((r, mu, sd, len(vals)))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "mean", "std", "n"])
        w.writerows(rows)

    print(f"Wrote {args.out}  (rounds={len(rows)})")

if __name__ == "__main__":
    main()