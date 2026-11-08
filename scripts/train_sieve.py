"""Train a two-feature reuse model on chronological SIEVE examples."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from sieve_examples import examples
from twitter_baseline import ROOT, prepare_trace, request_keys, validation_start


def training_arrays(binary, trace, capacity, horizon):
    rows = list(examples(binary, trace, capacity, horizon))
    labels = np.array([row.reused for row in rows])
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{trace.stem} requires both reuse labels; use a larger split")
    features = np.log1p(np.array([(row.age_reads, row.resident_hits) for row in rows],
                                 dtype=float))
    return features, labels


def export_model(path, model):
    bias = model.intercept_[0]
    age, hits = model.coef_[0]
    path.write_text(
        f"cachetriage_logistic_v1\n"
        f"bias {bias:.17g}\n"
        f"log1p_age_reads {age:.17g}\n"
        f"log1p_resident_hits {hits:.17g}\n",
        encoding="ascii",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="uncompressed Twitter trace or prefix")
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if min(args.start_row, args.rows, args.capacity, args.horizon) < 1:
        parser.error("start row, row count, capacity, and horizon must be positive")
    binary = ROOT / "build" / "cachetriage"

    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-training-") as work:
            work = Path(work)
            trace = work / "reads.txt"
            report = prepare_trace(args.source, trace, args.start_row, args.rows)
            cutoff = validation_start(report["read_count"]) - 1
            train_trace = work / "training.txt"
            validation_trace = work / "validation.txt"
            with train_trace.open("w") as train, validation_trace.open("w") as validation:
                for request, key in enumerate(request_keys(trace), start=1):
                    (train if request <= cutoff else validation).write(f"{key}\n")

            train_x, train_y = training_arrays(binary, train_trace, args.capacity, args.horizon)
            valid_x, valid_y = training_arrays(binary, validation_trace, args.capacity, args.horizon)
            model = LogisticRegression(max_iter=1000).fit(train_x, train_y)
            scores = model.predict_proba(valid_x)[:, 1]
            export_model(args.model_out, model)
            report.update({
                "capacity_items": args.capacity,
                "lookahead_reads": args.horizon,
                "scikit_learn_version": sklearn.__version__,
                "training": {
                    "first_read": 1,
                    "last_read": cutoff,
                    "examples": len(train_y),
                    "reuse_rate": float(train_y.mean()),
                },
                "validation": {
                    "first_read": cutoff + 1,
                    "last_read": report["read_count"],
                    "examples": len(valid_y),
                    "reuse_rate": float(valid_y.mean()),
                    "average_precision": float(average_precision_score(valid_y, scores)),
                    "roc_auc": float(roc_auc_score(valid_y, scores)),
                },
            })
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
