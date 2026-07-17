#!/usr/bin/env python3
import argparse, csv, re

KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--max_round", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[WALLCLOCK]" not in line:
                continue
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if "round" not in kvs or "elapsed_total_sec" not in kvs:
                continue
            r = int(kvs["round"])
            if args.max_round is not None and r > args.max_round:
                continue
            t = float(kvs["elapsed_total_sec"])
            train_loss = kvs.get("train_loss")
            test_acc   = kvs.get("test_acc")
            rows.append((r, t, train_loss, test_acc))

    rows.sort(key=lambda x: x[0])

    # write time-vs metrics (using round-ordered time)
    with open(args.out_prefix + "time_train_loss.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_sec", "value"])
        for r, t, tl, ta in rows:
            if tl is not None:
                w.writerow([t, tl])

    with open(args.out_prefix + "time_test_acc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_sec", "value"])
        for r, t, tl, ta in rows:
            if ta is not None:
                w.writerow([t, ta])

    print("Wrote:", args.out_prefix + "time_train_loss.csv")
    print("Wrote:", args.out_prefix + "time_test_acc.csv")

if __name__ == "__main__":
    main()