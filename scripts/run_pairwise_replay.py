"""Replay frozen pairwise eviction policies on unused trace windows."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

from prepare_meta_kv import prepare as prepare_meta
from twitter_baseline import ROOT, prepare_trace, run_replay


FIELDS = ("age_reads", "resident_hits", "residency_reads", "previous_gap_reads")
FEATURES = tuple(f"{side}_log1p_{field}" for side in ("victim", "alternative_minus_victim")
                 for field in FIELDS)
BASIC_FEATURES = tuple(FEATURES[i] for i in (0, 1, 4, 5))
ITEMS = 128
BYTES = 40 * 1024
ROUNDS = 16


def pack_models(bundle, path):
    if bundle["format"] != "cachetriage_pairwise_v1":
        raise ValueError("unexpected pairwise model format")
    logistic = bundle["models"]["basic_pairwise_logistic"]
    boosted = bundle["models"]["history_shallow_boosted_trees"]
    if (logistic["kind"] != "logistic_regression" or tuple(logistic["feature_order"]) != BASIC_FEATURES or
            boosted["kind"] != "gradient_boosted_trees" or tuple(boosted["feature_order"]) != FEATURES or
            len(logistic["weights"]) != 4):
        raise ValueError("unexpected pairwise model feature order or shape")

    def values(*row):
        if not all(math.isfinite(value) for value in row):
            raise ValueError("nonfinite pairwise model parameter")
        return " ".join(format(value, ".17g") for value in row)

    lines = [
        "cachetriage_pairwise_replay_v1",
        values(logistic["threshold"], logistic["intercept"], *logistic["weights"]),
        values(boosted["threshold"], boosted["initial_logit"], boosted["learning_rate"],
               len(boosted["trees"])),
    ]
    for tree in boosted["trees"]:
        lines.append(str(len(tree["feature"])))
        for node in zip(tree["feature"], tree["left"], tree["right"], tree["threshold"], tree["value"]):
            lines.append(values(*node))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def check_baselines(run, trace, reuse):
    binary = ROOT / "build" / "cachetriage"
    capacities = {p["policy"]: p["capacity_items"] for p in run["performance"]}
    sieve = run_replay(binary, capacities["sieve"], trace)["results"][1]
    gate = run_replay(binary, capacities["eight_hit_gate"], trace, "--triage-hits", reuse, "8")["results"][-1]
    if run["results"][0] != sieve or run["results"][1] != {**gate, "policy": "eight_hit_gate"}:
        raise ValueError("pairwise benchmark baselines disagree with the ordinary replay CLI")


def fresh_windows(data_dir, trace):
    cases = (
        ("Twitter cluster 29", "cluster029.csv", prepare_trace, 1, 100_000),
        ("Twitter cluster 52 later window", "cluster052.csv", prepare_trace, 100_001, 40_000),
        ("Meta KV 202206 later window", "meta-kv-202206-prefix.csv",
         lambda path, trace, start, count: prepare_meta(path, trace, count, start), 200_001, 100_000),
    )
    for name, filename, convert, start, rows in cases:
        source = convert(data_dir / filename, trace, start, rows)
        if name == "Twitter cluster 29":
            source["source_url"] = ("https://raw.githubusercontent.com/twitter/cache-trace/"
                                    "1c503676d6972d4394a360e09162149a6c00f516/samples/2020Mar/cluster029")
            source["download_byte_range"] = [0, 8388607]
        yield name, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "benchmarks" / "twitter-cluster052-pairwise-v1.json")
    args = parser.parse_args()
    binary = ROOT / "build" / "cachetriage_pairwise"
    reuse = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"
    try:
        model_bytes = args.model.read_bytes()
        bundle = json.loads(model_bytes)
        with tempfile.TemporaryDirectory(prefix="cachetriage-pairwise-replay-") as temp:
            work = Path(temp)
            packed = work / "model.txt"
            pack_models(bundle, packed)
            trace = work / "reads.txt"
            workloads = []
            for name, source in fresh_windows(args.data_dir, trace):
                replays = []
                for mode, limit in (("items", ITEMS), ("bytes", BYTES)):
                    result = json.loads(subprocess.check_output([
                        binary, trace, packed, reuse, mode, str(limit), str(ROUNDS),
                    ]))
                    check_baselines(result, trace, reuse)
                    replays.append(result)
                workloads.append({"workload": name, "source": source, "replays": replays})
        report = {
            "platform": platform.platform(),
            "pairwise_model_file": args.model.name,
            "pairwise_model_sha256": sha256(model_bytes).hexdigest(),
            "reuse_model_sha256": sha256(reuse.read_bytes()).hexdigest(),
            "frozen_thresholds": {name: bundle["models"][name]["threshold"]
                                  for name in ("basic_pairwise_logistic", "history_shallow_boosted_trees")},
            "memory_scope": "inline player, peak requested cache heap, and relevant loaded model and log table",
            "workloads": workloads,
        }
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
