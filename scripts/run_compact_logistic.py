"""Measure compact logistic against SIEVE under shared cache byte budgets."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import tempfile

from run_pairwise_replay import BYTES, ITEMS, ROOT, fresh_windows, pack_models


BUDGETS_KIB = (16, 24, 40, 64, 96)
SQLITE_BUDGETS_KIB = (40, 64)
SCAN_LIMIT = 1024
ROUNDS = 16


def run(*args):
    return json.loads(subprocess.check_output(args))


def by_policy(rows):
    return {row["policy"]: row for row in rows}


def measure(trace, packed, reuse, previous):
    binary = ROOT / "build" / "cachetriage_pairwise"

    def replay(mode, limit, rounds=ROUNDS):
        return run(binary, trace, packed, reuse, mode, str(limit), str(rounds))

    equal_items = replay("compact-items", ITEMS)
    previous_items = by_policy(previous["replays"][0]["results"])
    if any(row != previous_items[row["policy"]] for row in equal_items["results"]):
        raise ValueError("compact logistic changed the equal-capacity replay statistics")

    parity = run(ROOT / "build" / "check_compact_victims", packed, trace)
    if (parity["requests"], parity["matching_evictions"]) != (
            equal_items["requests"], previous_items["pairwise_logistic"]["evictions"]):
        raise ValueError("victim parity check disagrees with equal-capacity replay")

    sweep = [replay("compact-bytes", kib * 1024) for kib in BUDGETS_KIB]
    at_40 = next(result for result in sweep if result["budget_bytes"] == BYTES)
    target = by_policy(at_40["results"])["sieve"]["misses"]
    match = replay("compact-scan", SCAN_LIMIT, target)
    sqlite = [
        run(ROOT / "build" / "cachetriage_demo", "--pairwise-budget", str(kib * 1024), trace, packed, str(ROUNDS))
        for kib in SQLITE_BUDGETS_KIB
    ]
    for demo in sqlite:
        memory = by_policy(demo["memory"])
        for row in demo["results"]:
            policy = row["policy"]
            expected = by_policy(replay("compact-items", memory[policy]["capacity_items"], 1)["results"])[policy]
            if (row["backend_calls"], row["model_calls"]) != (expected["misses"], expected.get("model_calls", 0)):
                raise ValueError("SQLite call counts disagree with the equal-item trace replay")
    return {
        "victim_parity_at_128_items": parity,
        "equal_items": equal_items,
        "memory_sweep": sweep,
        "break_even_at_40_kib": {"sieve_miss_target": target, "max_capacity_scanned": SCAN_LIMIT, "first_match": match},
        "sqlite": sqlite,
    }


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ("Twitter 29", "Twitter 52, later window", "Meta KV, later window")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout="constrained")
    for axis, workload, title in zip(axes, report["workloads"], titles):
        sweep = workload["memory_sweep"]
        for policy, label, color, style in (
            ("sieve", "SIEVE", "#334155", "-"),
            ("pairwise_logistic", "Compact logistic", "#b45309", "--"),
        ):
            misses = [100 * by_policy(point["results"])[policy]["misses"] / point["requests"] for point in sweep]
            axis.plot(BUDGETS_KIB, misses, style, color=color, marker="o", markersize=4, label=label)
        axis.set(title=title, xlabel="Cache budget (KiB)", xticks=BUDGETS_KIB)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Miss rate (%)")
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--plot", type=Path, help="save a miss-rate plot (requires Matplotlib)")
    args = parser.parse_args()

    benchmarks = ROOT / "benchmarks"
    model_path = benchmarks / "twitter-cluster052-pairwise-v1.json"
    reuse = benchmarks / "twitter-cluster052-reuse-v1.txt"
    baseline = json.loads((benchmarks / "pairwise-fresh-replay.json").read_text())
    model_bytes = model_path.read_bytes()
    digest = sha256(model_bytes).hexdigest()
    if digest != baseline["pairwise_model_sha256"]:
        raise ValueError("model differs from the recorded full-history baseline")

    with tempfile.TemporaryDirectory(prefix="cachetriage-compact-logistic-") as tmp:
        trace = Path(tmp) / "reads.txt"
        packed = Path(tmp) / "model.txt"
        pack_models(json.loads(model_bytes), packed)
        workloads = []
        previous = {workload["workload"]: workload for workload in baseline["workloads"]}
        for name, source in fresh_windows(args.data_dir, trace):
            if source != previous[name]["source"]:
                raise ValueError(f"trace source differs from the recorded baseline for {name}")
            workloads.append({"workload": name, "source": source, **measure(trace, packed, reuse, previous[name])})

    report = {
        "platform": platform.platform(),
        "pairwise_model_file": model_path.name,
        "pairwise_model_sha256": digest,
        "memory_scope": "inline cache/player, peak requested cache heap, and the relevant model and log table",
        "sqlite_cpu_scope": "request loop including per-request timing and profile checks; excludes setup and sorting",
        "workloads": workloads,
    }
    if args.plot:
        plot(report, args.plot)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
