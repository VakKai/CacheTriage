"""Convert a bounded Wikimedia 2019 upload-cache sample into numeric reads."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, trace, row_count):
    rows = []
    digest = hashlib.sha256()
    descents = 0
    with source.open("rb") as raw:
        for line_number, line in enumerate(raw, start=1):
            if not rows and (not line.strip() or line.startswith(b"==> ")):
                continue
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("expected a complete newline-terminated row")
                time, key, _, _, _ = line.rstrip(b"\r\n").split(b"\t")
                time = int(time)
                if time < 0 or not key:
                    raise ValueError("expected a nonnegative timestamp and object ID")
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if rows and time < rows[-1][0]:
                descents += 1
            rows.append((time, key))
            digest.update(line)
            if len(rows) == row_count:
                break
    if len(rows) != row_count:
        raise ValueError(f"{source} ended before data row {row_count}")

    rows.sort(key=lambda row: row[0])
    keys = {}
    with trace.open("w", encoding="ascii") as output:
        for _, key in rows:
            output.write(f"{keys.setdefault(key, len(keys))}\n")
    return {
        "source_file": source.name,
        "source_start_row": 1,
        "source_end_row": row_count,
        "source_slice_sha256": digest.hexdigest(),
        "source_timestamp_descents": descents,
        "read_timestamp_range_seconds": [rows[0][0], rows[-1][0]],
        "read_count": len(rows),
        "distinct_read_keys": len(keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("trace", type=Path, help="output one numeric key per read")
    parser.add_argument("--rows", type=int, default=200_000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("row count must be positive")
    if args.source.resolve() == args.trace.resolve():
        parser.error("source and output must be different files")
    try:
        print(json.dumps(prepare(args.source, args.trace, args.rows), indent=2, sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
