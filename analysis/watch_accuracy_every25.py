#!/usr/bin/env python3
import argparse
import csv
import re
import time
from pathlib import Path


def parse_text_log(path):
    values, current = {}, None
    round_re = re.compile(r"Algo\s+\S+\s+Round No\.\s+(\d+)")
    acc_re = re.compile(r"Test Loss\s+[0-9.eE+-]+\s+Test Accuracy\s+([0-9.eE+-]+)")
    for line in Path(path).read_text().splitlines():
        match = round_re.search(line)
        if match:
            current = int(match.group(1))
            continue
        match = acc_re.search(line)
        if current is not None and match:
            values[current] = float(match.group(1))
    return values


def parse_round_csv(path):
    values = {}
    if not Path(path).exists():
        return values
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            values[int(row["round"])] = float(row["global_test_accuracy"])
    return values


def write_report(output, series):
    current = series["SCAFFOLD-SLS"]
    checkpoints = [r for r in range(24, 1000, 25) if r in current]
    fields = ["round"]
    for name in series:
        fields.extend((f"{name}_test_accuracy", f"{name}_best_so_far"))
    temporary = Path(str(output) + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_id in checkpoints:
            row = {"round": round_id + 1}
            for name, values in series.items():
                row[f"{name}_test_accuracy"] = values.get(round_id, "")
                available = [acc for r, acc in values.items() if r <= round_id]
                row[f"{name}_best_so_far"] = max(available) if available else ""
            writer.writerow(row)
    temporary.replace(output)
    return bool(checkpoints and checkpoints[-1] == 999)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-csv", required=True)
    parser.add_argument("--scaffold-log", required=True)
    parser.add_argument("--fedsls-log", required=True)
    parser.add_argument("--fedexpsls-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baselines = {
        "SCAFFOLD": parse_text_log(args.scaffold_log),
        "FedSLS": parse_text_log(args.fedsls_log),
        "FedExpSLS": parse_text_log(args.fedexpsls_log),
    }
    while True:
        series = {"SCAFFOLD-SLS": parse_round_csv(args.round_csv), **baselines}
        if write_report(Path(args.output), series):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
