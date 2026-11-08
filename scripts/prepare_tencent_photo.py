"""Convert a bounded Tencent QQPhoto trace prefix into numeric object reads."""

import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import subprocess


def prepare(raw, trace, row_count):
    rows = []
    digest = hashlib.sha256()
    skipped = 0
    descents = 0
    previous_time = None
    for row_number, line in enumerate(islice(raw, row_count), start=1):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("expected a complete newline-terminated row")
            time, photo, image_format, size_category, _, _, _, _ = line.split()
            if len(time) != 14 or not time.isdigit():
                raise ValueError("expected a YYYYMMDDHHMMSS timestamp")
        except ValueError as exc:
            raise ValueError(f"data row {row_number}: {exc}") from exc
        digest.update(line)
        if previous_time is not None and time < previous_time:
            descents += 1
        previous_time = time
        if image_format not in (b"0", b"5"):
            skipped += 1
            continue
        if size_category not in (b"l", b"a", b"o", b"m", b"c", b"b"):
            raise ValueError(f"data row {row_number}: unexpected size category")
        rows.append((time, (photo, image_format, size_category)))

    if len(rows) + skipped != row_count:
        raise ValueError(f"compressed prefix ended before data row {row_count}")
    if not rows:
        raise ValueError("selected prefix contains no valid photo reads")
    rows.sort(key=lambda row: row[0])
    keys = {}
    with trace.open("w", encoding="ascii") as output:
        for _, key in rows:
            output.write(f"{keys.setdefault(key, len(keys))}\n")
    return {
        "source_start_row": 1,
        "source_end_row": row_count,
        "source_slice_sha256": digest.hexdigest(),
        "source_timestamp_descents": descents,
        "read_timestamp_range": [rows[0][0].decode("ascii"), rows[-1][0].decode("ascii")],
        "read_count": len(rows),
        "distinct_read_keys": len(keys),
        "skipped_format_errors": skipped,
        "object_identity": ["photo_id", "image_format", "size_category"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="zstd-compressed Tencent trace prefix")
    parser.add_argument("trace", type=Path, help="output one numeric object key per read")
    parser.add_argument("--rows", type=int, default=200_000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("row count must be positive")
    if args.source.resolve() == args.trace.resolve():
        parser.error("source and output must be different files")
    try:
        with subprocess.Popen(["zstd", "-dc", "--", args.source], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL) as decoder:
            try:
                report = prepare(decoder.stdout, args.trace, args.rows)
            finally:
                # A byte-range prefix may end inside a frame. Only the complete
                # requested records are needed, so stop decoding after them.
                if decoder.poll() is None:
                    decoder.terminate()
        report["source_file"] = args.source.name
        print(json.dumps(report, indent=2, sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
