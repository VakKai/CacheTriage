"""Convert a bounded prefix of Meta's 202206 KV trace to numeric reads."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path


FIELDS = ["key", "op", "size", "op_count", "key_size"]


def prepare(source, trace, row_count, start_row=1):
    operations = Counter()
    reads = []
    keys = {}
    digest = hashlib.sha256()

    with source.open("rb") as raw:
        header = next(csv.reader([raw.readline().decode("ascii")]), [])
        if header != FIELDS:
            raise ValueError("expected the Meta 202206 KV trace header")

        window = islice(raw, start_row - 1, start_row - 1 + row_count)
        for row_number, line in enumerate(window, start=start_row):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("expected a complete newline-terminated row")
                key, op, _, count, _ = next(csv.reader([line.decode("ascii")], strict=True))
                count = int(count)
                if not key or op not in {"GET", "SET", "DELETE"} or count < 1:
                    raise ValueError("expected a key, recognized operation, and positive op_count")
            except (csv.Error, ValueError) as exc:
                raise ValueError(f"{source}, data row {row_number}: {exc}") from exc
            digest.update(line)
            operations[op] += 1
            if op == "GET":
                key_id = keys.setdefault(key, len(keys))
                reads.append((key_id, count))

    if sum(operations.values()) != row_count:
        raise ValueError(f"{source} ended before data row {start_row + row_count - 1}")
    if not reads:
        raise ValueError("selected prefix contains no GET rows")

    with trace.open("w", encoding="ascii") as output:
        for key_id, count in reads:
            output.write(f"{key_id}\n" * count)

    return {
        "source_file": source.name,
        "source_start_row": start_row,
        "source_end_row": start_row + row_count - 1,
        "source_slice_sha256": digest.hexdigest(),
        "operation_rows": dict(operations),
        "read_count": sum(count for _, count in reads),
        "distinct_read_keys": len(keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("trace", type=Path, help="output one numeric key per read")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--rows", type=int, default=100_000)
    args = parser.parse_args()
    if args.rows < 1 or args.start_row < 1:
        parser.error("start row and row count must be positive")
    if args.source.resolve() == args.trace.resolve():
        parser.error("source and output must be different files")
    try:
        report = prepare(args.source, args.trace, args.rows, args.start_row)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
