"""Retest the four policies on all six saved production trace slices."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from twitter_baseline import ROOT, prepare_trace, run_replay


CAPACITIES = (128, 512, 2048, 8192)
GATE_GAP = "1"
REPLAY_ROUNDS = "16"
SQLITE_ROUNDS = "8"
CASES = (
    ("Twitter cluster 17", "cluster017.csv", None, "twitter-cluster017.json"),
    ("Twitter cluster 24", "cluster024.csv", None, "twitter-cluster024-holdout.json"),
    ("Twitter cluster 52", "cluster052.csv", None, "twitter-cluster052.json"),
    ("Meta KV 202206", "meta-kv-202206-prefix.csv", "prepare_meta_kv.py",
     "meta-kv-202206-score-optimization.json"),
    ("Wikimedia CDN upload 2019", "wiki-upload-2019-prefix.txt", "prepare_wiki_upload.py",
     "wikimedia-upload-2019-node-reuse.json"),
    ("Tencent QQPhoto", "tencent-photo1-prefix.zst", "prepare_tencent_photo.py",
     "tencent-photo1-access-optimization.json"),
)
COUNT_FIELDS = ("hits", "misses", "evictions", "model_calls", "victim_swaps")


def reference_for(name, filename):
    report = json.loads((ROOT / "benchmarks" / filename).read_text())
    if "workloads" in report:
        return next(work for work in report["workloads"] if work["workload"] == name)
    return {"source": report, "replays": report["replays"]}


def prepare(name, source, trace, converter, reference):
    expected = reference["source"]
    rows = expected["source_end_row"]
    if converter is None:
        prepared = prepare_trace(source, trace, 1, rows)
    else:
        prepared = json.loads(subprocess.check_output([
            sys.executable, "-B", ROOT / "scripts" / converter,
            source, trace, "--rows", str(rows),
        ]))
    for field in ("source_slice_sha256", "read_count", "distinct_read_keys"):
        if prepared[field] != expected[field]:
            raise ValueError(f"{name}: input {field} disagrees with the saved reference")
    for field in ("source_url", "download_byte_range", "source_etag"):
        if field in expected:
            prepared[field] = expected[field]
    return prepared


def check_reference(name, capacity, replay, reference):
    previous = next((r for r in reference["replays"] if r["capacity_items"] == capacity), None)
    if previous is None:
        return 0
    current = {row["policy"]: row for row in replay["results"]}
    for row in previous["results"]:
        policy = row["policy"]
        for field in COUNT_FIELDS:
            if field in row and current[policy][field] != row[field]:
                raise ValueError(f"{name} at {capacity} items: {policy} {field} changed")
    return len(previous["results"])


def check_sqlite(name, replay, sqlite):
    if sqlite["requests"] != replay["requests"] or sqlite["capacity_items"] != replay["capacity_items"]:
        raise ValueError(f"{name}: SQLite and replay used different request counts or capacities")
    rows = {row["policy"]: row for row in sqlite["results"]}
    for expected in replay["results"]:
        policy = expected["policy"]
        actual = rows[policy]
        if (actual["backend_calls"] != expected["misses"] or
                actual["model_calls"] != expected.get("model_calls", 0)):
            raise ValueError(f"{name}: {policy} SQLite call counts disagree with replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build",
                        help="directory containing the six downloaded trace prefixes")
    args = parser.parse_args()
    binary = ROOT / "build" / "cachetriage"
    demo = ROOT / "build" / "cachetriage_demo"
    model = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"
    if not binary.is_file() or not demo.is_file():
        parser.error("build the replay and SQLite demo with make all demo")

    report = {
        "platform": platform.platform(),
        "model_file": model.name,
        "model_sha256": sha256(model.read_bytes()).hexdigest(),
        "gate_max_age_gap_reads": int(GATE_GAP),
        "reference_policy_cases_matched": 0,
        "sqlite_policy_cases_matched": 0,
        "workloads": [],
    }

    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-retest-") as work:
            trace = Path(work) / "reads.txt"
            for name, filename, converter, reference_filename in CASES:
                print(f"Retesting {name}", file=sys.stderr, flush=True)
                reference = reference_for(name, reference_filename)
                source = prepare(name, args.data_dir / filename, trace, converter, reference)
                replays = []
                for capacity in CAPACITIES:
                    result = run_replay(binary, capacity, trace, "--benchmark", model,
                                        GATE_GAP, REPLAY_ROUNDS)
                    report["reference_policy_cases_matched"] += check_reference(
                        name, capacity, result, reference)
                    replays.append(result)
                sqlite = run_replay(demo, CAPACITIES[0], trace, model, GATE_GAP, SQLITE_ROUNDS)
                check_sqlite(name, replays[0], sqlite)
                report["sqlite_policy_cases_matched"] += len(sqlite["results"])
                report["workloads"].append({
                    "workload": name,
                    "source": source,
                    "reference_file": reference_filename,
                    "replays": replays,
                    "sqlite": sqlite,
                })
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
