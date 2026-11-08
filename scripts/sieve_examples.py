"""Export labeled SIEVE shortlist candidates from a numeric read trace."""

import argparse
import json
from pathlib import Path
import subprocess
from typing import NamedTuple

from twitter_baseline import ROOT, read_positions, request_keys, reused_within, sieve_decisions


class Example(NamedTuple):
    request: int
    key: int
    candidate_rank: int
    age_reads: int
    resident_hits: int
    reused: int


def examples(binary, trace, capacity, horizon):
    positions = read_positions(trace)
    read_count = sum(map(len, positions.values()))
    residents = {}
    decisions = (tuple(map(int, line.split(",")))
                 for line in sieve_decisions(binary, trace, capacity))
    decision = next(decisions, None)

    for request, key in enumerate(request_keys(trace), start=1):
        if key in residents:
            _, hits = residents[key]
            residents[key] = (request, hits + 1)
            continue

        if len(residents) == capacity:
            if decision is None or decision[0] != request:
                raise ValueError(f"missing SIEVE eviction for request {request}")
            _, victim, *alternatives = decision
            if request + horizon <= read_count:
                for rank, candidate in enumerate((victim, *alternatives)):
                    last_access, hits = residents[candidate]
                    yield Example(request, candidate, rank, request - last_access, hits,
                                  int(reused_within(positions, candidate, request, horizon)))
            del residents[victim]
            decision = next(decisions, None)
        # Admission is an access, but the miss that admitted the key is not a hit.
        residents[key] = (request, 0)

    if decision is not None:
        raise ValueError("unexpected extra SIEVE eviction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if args.capacity < 1 or args.horizon < 1:
        parser.error("capacity and horizon must be positive")

    binary = ROOT / "build" / "cachetriage"
    try:
        for example in examples(binary, args.trace, args.capacity, args.horizon):
            print(json.dumps(example._asdict()))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
