"""Replay a bounded window from an uncompressed Twitter cache trace."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import islice
import json
import platform
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
READS = {"get", "gets"}
OPERATIONS = READS | {"set", "add", "replace", "cas", "append", "prepend", "delete", "incr", "decr"}


def prepare_trace(source, trace, start_row, row_count):
    reads = []
    operations = Counter()
    digest = hashlib.sha256()
    timestamp_descents = 0
    previous_timestamp = None

    with source.open("rb") as raw:
        for row_number, line in enumerate(islice(raw, start_row - 1, start_row - 1 + row_count), start_row):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("expected a complete newline-terminated row")
                row = next(csv.reader([line.decode("utf-8")], strict=True))
                if len(row) != 7 or not row[1] or row[5] not in OPERATIONS:
                    raise ValueError("expected seven Twitter trace fields and a recognized operation")
                timestamp, key, operation = int(row[0]), row[1], row[5]
                if timestamp < 0:
                    raise ValueError("timestamp must be nonnegative")
            except (csv.Error, ValueError) as exc:
                raise ValueError(f"{source}, row {row_number}: {exc}") from exc

            digest.update(line)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                timestamp_descents += 1
            previous_timestamp = timestamp
            operations[operation] += 1
            if operation in READS:
                reads.append((timestamp, key))

    if sum(operations.values()) != row_count:
        raise ValueError(f"{source} ended before row {start_row + row_count - 1}")
    if not reads:
        raise ValueError("the selected window contains no reads")

    reads.sort(key=lambda read: read[0])  # Python preserves source order for equal timestamps.
    key_ids = {}
    with trace.open("w", encoding="ascii") as output:
        for _, key in reads:
            key_id = key_ids.setdefault(key, len(key_ids))
            output.write(f"{key_id}\n")

    return {
        "source_file": source.name,
        "source_start_row": start_row,
        "source_end_row": start_row + row_count - 1,
        "source_slice_sha256": digest.hexdigest(),
        "source_timestamp_descents": timestamp_descents,
        "read_timestamp_range_seconds": [reads[0][0], reads[-1][0]],
        "operations": dict(operations),
        "read_count": len(reads),
        "distinct_read_keys": len(key_ids),
    }


def validation_start(read_count):
    return read_count * 7 // 10 + 1


def request_keys(trace):
    with trace.open(encoding="ascii") as reads:
        for line in reads:
            if line.strip():
                yield int(line)


def read_positions(trace):
    positions = defaultdict(list)
    for request, key in enumerate(request_keys(trace), start=1):
        positions[key].append(request)
    return positions


def reused_within(positions, key, request, lookahead):
    hits = positions[key]
    next_hit = bisect_right(hits, request)
    return next_hit < len(hits) and hits[next_hit] <= request + lookahead


def sieve_decisions(binary, trace, capacity):
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii") as decisions:
        subprocess.run([binary, str(capacity), trace, "--sieve-decisions"],
                       stdout=decisions, check=True)
        decisions.seek(0)
        yield from decisions


def count_opportunities(decisions, positions, read_count, lookahead):
    total = eligible = victim_reused = opportunities = 0
    for line in decisions:
        request, victim, *alternatives = map(int, line.split(","))
        total += 1
        if request + lookahead > read_count:
            continue
        eligible += 1
        if reused_within(positions, victim, request, lookahead):
            victim_reused += 1
            if any(not reused_within(positions, key, request, lookahead) for key in alternatives):
                opportunities += 1
    return {
        "total_evictions": total,
        "evaluated_evictions": eligible,
        "excluded_incomplete_window": total - eligible,
        "victim_reused": victim_reused,
        "opportunities": opportunities,
        "opportunity_rate": opportunities / eligible if eligible else None,
    }


def run_replay(binary, capacity, trace, *policy_args):
    return json.loads(subprocess.check_output([binary, str(capacity), trace, *policy_args]))


def sweep_gate(binary, capacity, trace, model, gaps):
    replay = run_replay(binary, capacity, trace, "--always-score", model)
    for gap in gaps:
        gated = run_replay(binary, capacity, trace, "--triage", model, str(gap))
        result = gated["results"][-1]
        result["gate_max_age_gap_reads"] = gated["gate_max_age_gap_reads"]
        replay["results"].append(result)
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="uncompressed Twitter trace or downloaded prefix")
    parser.add_argument("--start-row", type=int, default=1, help="first source row, counted from 1")
    parser.add_argument("--rows", type=int, default=100_000, help="number of source rows before filtering")
    parser.add_argument("--capacities", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--sieve-lookahead", type=int, help="count opportunities within this many future reads")
    policies = parser.add_mutually_exclusive_group()
    policies.add_argument("--sieve-oracle", type=int, help="replay a SIEVE oracle with this many future reads")
    policies.add_argument("--always-score", type=Path, metavar="MODEL", help="score every SIEVE shortlist")
    policies.add_argument("--gate-sweep", type=Path, metavar="MODEL",
                          help="compare gate thresholds with SIEVE and always-score")
    parser.add_argument("--gate-gaps", type=int, nargs="+", metavar="READS",
                        help="maximum age gaps to test with --gate-sweep")
    policies.add_argument("--benchmark", type=Path, metavar="MODEL",
                          help="measure all four policies on the same preloaded trace")
    policies.add_argument("--backend-demo", type=Path, metavar="MODEL",
                          help="measure the request path with a SQLite backing store")
    parser.add_argument("--demo-delays", type=int, nargs="+", metavar="MICROSECONDS",
                        help="compare SIEVE and CacheTriage with added fetch delays")
    parser.add_argument("--demo-rounds", type=int, help="measured rounds per demo policy")
    parser.add_argument("--benchmark-rounds", type=int, help="measured rounds per policy (default: 16)")
    parser.add_argument("--benchmark-gate-gap", type=int, help="gate age gap for the benchmark (default: 1)")
    parser.add_argument("--validation-only", action="store_true", help="replay the last 30%% used for validation")
    args = parser.parse_args()
    if args.demo_delays is not None and (args.backend_demo is None or min(args.demo_delays) < 0):
        parser.error("--demo-delays requires --backend-demo and nonnegative delays")
    if args.demo_rounds is not None and (args.backend_demo is None or args.demo_rounds < 1):
        parser.error("--demo-rounds requires --backend-demo and a positive round count")
    if args.backend_demo is not None and args.sieve_lookahead is not None:
        parser.error("--backend-demo and --sieve-lookahead are separate runs")
    if args.benchmark_rounds is not None and (args.benchmark is None or args.benchmark_rounds < 1):
        parser.error("--benchmark-rounds requires --benchmark and a positive round count")
    if args.benchmark_gate_gap is not None and (args.benchmark is None or args.benchmark_gate_gap < 0):
        parser.error("--benchmark-gate-gap requires --benchmark and a nonnegative age gap")
    if args.gate_gaps is not None and (args.gate_sweep is None or min(args.gate_gaps) < 0):
        parser.error("--gate-gaps requires --gate-sweep and nonnegative gaps")
    if args.sieve_oracle is not None and args.sieve_oracle < 1:
        parser.error("SIEVE oracle lookahead must be positive")
    if args.sieve_lookahead is not None and args.sieve_lookahead < 1:
        parser.error("SIEVE lookahead must be positive")
    if args.start_row < 1 or args.rows < 1 or min(args.capacities) < 0:
        parser.error("start row and row count must be positive, and capacities must be nonnegative")

    binary_name = "cachetriage_demo" if args.backend_demo is not None else "cachetriage"
    binary = ROOT / "build" / binary_name
    if not binary.is_file():
        target = "make demo" if args.backend_demo is not None else "make"
        parser.error(f"build the {binary_name} CLI with {target} first")

    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-") as work:
            trace = Path(work) / "reads.txt"
            report = prepare_trace(args.source, trace, args.start_row, args.rows)
            if args.validation_only:
                start = validation_start(report["read_count"])
                validation = Path(work) / "validation.txt"
                with validation.open("w", encoding="ascii") as output:
                    for key in islice(request_keys(trace), start - 1, None):
                        output.write(f"{key}\n")
                trace = validation
                report["replay_first_read"] = start
                report["replay_last_read"] = report["read_count"]

            model = args.always_score or args.gate_sweep or args.benchmark or args.backend_demo
            if model is not None:
                report["model_file"] = model.name
                report["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()

            if args.gate_sweep is not None:
                gaps = sorted(set(args.gate_gaps or [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]))
                report["replays"] = [
                    sweep_gate(binary, capacity, trace, model, gaps)
                    for capacity in args.capacities
                ]
            else:
                policy_args = []
                if args.sieve_oracle is not None:
                    policy_args = ["--sieve-oracle", str(args.sieve_oracle)]
                elif args.always_score is not None:
                    policy_args = ["--always-score", model]
                elif args.benchmark is not None:
                    rounds = args.benchmark_rounds or 16
                    gap = 1 if args.benchmark_gate_gap is None else args.benchmark_gate_gap
                    policy_args = ["--benchmark", model, str(gap), str(rounds)]
                    report["benchmark_platform"] = platform.platform()
                elif args.backend_demo is not None:
                    rounds = args.demo_rounds or (4 if args.demo_delays is not None else 8)
                    policy_args = [model, "1", str(rounds)]
                    if args.demo_delays is not None:
                        policy_args += ["--delay-sweep", *map(str, sorted(set(args.demo_delays)))]
                    report["benchmark_platform"] = platform.platform()
                report["replays"] = [
                    run_replay(binary, capacity, trace, *policy_args)
                    for capacity in args.capacities
                ]
            if args.sieve_lookahead is not None:
                report["sieve_lookahead_reads"] = args.sieve_lookahead
                positions = read_positions(trace)
                for replay in report["replays"]:
                    capacity = replay["capacity_items"]
                    decisions = sieve_decisions(binary, trace, capacity)
                    counts = count_opportunities(decisions, positions, replay["requests"],
                                                 args.sieve_lookahead)
                    sieve = next(result for result in replay["results"] if result["policy"] == "sieve")
                    if counts["total_evictions"] != sieve["evictions"]:
                        raise ValueError(f"SIEVE decision count disagrees with replay at capacity {capacity}")
                    replay["sieve_opportunities"] = counts
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
