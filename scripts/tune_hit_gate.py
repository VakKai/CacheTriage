"""Tune a victim-hit gate on known slices, then evaluate two fresh windows."""

import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import platform
from statistics import mean
import subprocess
import sys
import tempfile

from prepare_meta_kv import prepare as prepare_meta
from retest_all import CASES, check_sqlite, prepare, reference_for
from twitter_baseline import ROOT, prepare_trace, request_keys, run_replay, validation_start


CAPACITY = 128
THRESHOLDS = (1, 2, 4, 8, 16, 32, 64)
DEVELOPMENT = {"Twitter cluster 52", "Meta KV 202206", "Wikimedia CDN upload 2019", "Tencent QQPhoto"}


def only_validation(trace, cutoff, suffix):
    with suffix.open("w", encoding="ascii") as output:
        for key in islice(request_keys(trace), cutoff - 1, None):
            output.write(f"{key}\n")


def development_runs(binary, model, data_dir, work):
    result = []
    trace = work / "development.txt"
    for name, filename, converter, reference_file in CASES:
        if name not in DEVELOPMENT:
            continue
        source = prepare(name, data_dir / filename, trace, converter, reference_for(name, reference_file))
        replay_trace = trace
        if name == "Twitter cluster 52":
            start = validation_start(source["read_count"])
            replay_trace = work / "validation.txt"
            only_validation(trace, start, replay_trace)
            source["replay_first_read"] = start
            source["replay_last_read"] = source["read_count"]

        age = run_replay(binary, CAPACITY, replay_trace, "--triage", model, "1")
        trials = []
        for threshold in THRESHOLDS:
            candidate = run_replay(binary, CAPACITY, replay_trace, "--triage-hits", model, str(threshold))
            if candidate["results"][:2] != age["results"][:2]:
                raise ValueError(f"{name}: baseline counts changed during tuning")
            trials.append({"min_victim_hits": threshold, "cachetriage": candidate["results"][-1]})
        result.append({"workload": name, "source": source, "requests": age["requests"],
                       "sieve": age["results"][1], "age_gate": age["results"][2], "trials": trials})
    return result


def summarize_development(runs):
    rows = []
    for i, threshold in enumerate(THRESHOLDS):
        candidates = [run["trials"][i]["cachetriage"] for run in runs]
        deltas = [(candidate["misses"] - run["sieve"]["misses"]) / run["requests"]
                  for candidate, run in zip(candidates, runs)]
        rows.append({
            "min_victim_hits": threshold,
            "mean_miss_rate_delta_vs_sieve": mean(deltas),
            "mean_calls_per_eviction": mean(row["model_calls_per_eviction"] for row in candidates),
        })
    return rows


def fresh_runs(binary, demo, model, threshold, data_dir, work):
    trace = work / "fresh.txt"
    cases = (
        ("Twitter cluster 24 later window", "cluster024.csv", prepare_trace, 40_000),
        ("Meta KV 202206 later window", "meta-kv-202206-prefix.csv",
         lambda path, trace, start, count: prepare_meta(path, trace, count, start), 100_000),
    )
    for name, filename, convert, rows in cases:
        source = convert(data_dir / filename, trace, 100_001, rows)
        benchmark = run_replay(binary, CAPACITY, trace, "--benchmark-hits", model, str(threshold), "16")
        sqlite_runs = [
            run_replay(demo, CAPACITY, trace, model, str(threshold), "16", "--hit-gate")
            for _ in range(3)
        ]
        for sqlite in sqlite_runs:
            check_sqlite(name, benchmark, sqlite)
        age = run_replay(binary, CAPACITY, trace, "--triage", model, "1")
        if age["results"][:2] != benchmark["results"][:2]:
            raise ValueError(f"{name}: baseline counts changed across gates")
        yield {"workload": name, "source": source, "benchmark": benchmark, "sqlite_runs": sqlite_runs,
               "age_gate": age["results"][-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build",
                        help="directory containing the downloaded trace prefixes")
    args = parser.parse_args()
    binary = ROOT / "build" / "cachetriage"
    demo = ROOT / "build" / "cachetriage_demo"
    model = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"
    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-hit-gate-") as temp:
            work = Path(temp)
            development = development_runs(binary, model, args.data_dir, work)
            summary = summarize_development(development)
            selected = min(summary, key=lambda row: (row["mean_miss_rate_delta_vs_sieve"],
                                                      row["mean_calls_per_eviction"]))
            threshold = selected["min_victim_hits"]
            print(f"Development selected {threshold} minimum victim hits. Evaluating fresh windows.",
                  file=sys.stderr, flush=True)
            fresh = list(fresh_runs(binary, demo, model, threshold, args.data_dir, work))
        report = {
            "platform": platform.platform(),
            "model_file": model.name,
            "model_sha256": sha256(model.read_bytes()).hexdigest(),
            "capacity_items": CAPACITY,
            "gate_rule": "victim has at least the threshold hits and first alternative has zero hits",
            "selection_rule": "minimize mean miss-rate delta on development; break ties by mean calls per eviction",
            "selected_min_victim_hits": threshold,
            "development_summary": summary,
            "development": development,
            "fresh_windows": fresh,
        }
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
