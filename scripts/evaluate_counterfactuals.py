"""Measure the local miss effect of overriding sampled SIEVE evictions."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from retest_all import CASES, prepare, reference_for
from tune_hit_gate import DEVELOPMENT, only_validation
from twitter_baseline import ROOT, run_replay, sieve_decisions, validation_start


def summarize(decisions):
    groups = [[c["misses_saved"] for c in decision["candidates"][1:]] for decision in decisions]
    return {
        "decisions_sampled": len(groups),
        "decisions_with_beneficial_alternative": sum(any(delta > 0 for delta in group) for group in groups),
        "decisions_with_harmful_alternative": sum(any(delta < 0 for delta in group) for group in groups),
        "alternative_pairs": sum(map(len, groups)),
        "beneficial_pairs": sum(delta > 0 for group in groups for delta in group),
        "tied_pairs": sum(delta == 0 for group in groups for delta in group),
        "harmful_pairs": sum(delta < 0 for group in groups for delta in group),
        "sum_best_local_misses_saved": sum(max([0, *group]) for group in groups),
        "sum_worst_local_extra_misses": sum(-min([0, *group]) for group in groups),
    }


def verify_shortlists(binary, trace, capacity, horizon, result):
    expected = {row["request"]: [c["key"] for c in row["candidates"]]
                for row in result["sampled_decisions"]}
    eligible = 0
    for line in sieve_decisions(binary, trace, capacity):
        request, *candidates = map(int, line.split(","))
        eligible += request + horizon <= result["requests"]
        if request in expected and expected.pop(request) != candidates:
            raise ValueError(f"shortlist disagrees with normal SIEVE replay at read {request}")
    if expected or eligible != result["eligible_evictions"]:
        raise ValueError("eligible or sampled decisions disagree with normal SIEVE replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000, help="maximum sampled evictions per development slice")
    parser.add_argument("--labels-out", type=Path, help="optional output JSONL, one row per eviction")
    args = parser.parse_args()
    if args.capacity < 2 or args.horizon < 1 or args.samples < 1:
        parser.error("capacity must be at least two, and horizon and samples must be positive")
    if args.labels_out is not None:
        if args.labels_out.exists():
            parser.error("labels output already exists; use a new filename")

    binary = ROOT / "build" / "cachetriage_counterfactual"
    replay = ROOT / "build" / "cachetriage"
    workloads = []
    labels = []
    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-counterfactual-") as temp:
            work = Path(temp)
            trace = work / "reads.txt"
            for name, filename, converter, reference_file in CASES:
                if name not in DEVELOPMENT:
                    continue
                print(f"Evaluating {name}", file=sys.stderr, flush=True)
                source = prepare(name, args.data_dir / filename, trace, converter, reference_for(name, reference_file))
                replay_trace = trace
                if name == "Twitter cluster 52":
                    start = validation_start(source["read_count"])
                    replay_trace = work / "validation.txt"
                    only_validation(trace, start, replay_trace)
                    source["replay_first_read"] = start
                    source["replay_last_read"] = source["read_count"]

                result = run_replay(binary, args.capacity, replay_trace, str(args.horizon), str(args.samples))
                verify_shortlists(replay, replay_trace, args.capacity, args.horizon, result)
                sampled = result["sampled_decisions"]
                workloads.append({
                    "workload": name,
                    "source": source,
                    "requests": result["requests"],
                    "eligible_evictions": result["eligible_evictions"],
                    **summarize(sampled),
                })
                labels.extend({"workload": name, **row} for row in sampled)
        if args.labels_out is not None:
            with args.labels_out.open("x", encoding="utf-8") as output:
                for row in labels:
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
    report = {
        "capacity_items": args.capacity,
        "horizon_reads": args.horizon,
        "max_samples_per_workload": args.samples,
        "sampling": "midpoint of each equal-count bin of eligible SIEVE evictions",
        "label": "SIEVE future misses minus alternative future misses; positive favors overriding",
        "workloads": workloads,
        "totals": summarize(labels),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
