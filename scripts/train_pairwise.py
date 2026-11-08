"""Train and compare pairwise eviction models on chronological Twitter windows."""

import argparse
from math import log, log1p
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_counterfactuals import summarize, verify_shortlists
from twitter_baseline import ROOT, prepare_trace, request_keys, run_replay


FIELDS = ("age_reads", "resident_hits", "residency_reads", "previous_gap_reads")
FEATURES = tuple(f"{side}_log1p_{field}" for side in ("victim", "alternative_minus_victim")
                 for field in FIELDS)
BASIC_COLUMNS = (0, 1, 4, 5)
ALL_COLUMNS = tuple(range(len(FEATURES)))
THRESHOLDS = tuple(i / 20 for i in range(21))


def read_examples(trace, capacity, horizon, sample_limit):
    binary = ROOT / "build" / "cachetriage_counterfactual"
    replay = ROOT / "build" / "cachetriage"
    result = run_replay(binary, capacity, trace, str(horizon), str(sample_limit))
    verify_shortlists(replay, trace, capacity, horizon, result)
    decisions = result["sampled_decisions"]
    if not decisions:
        raise ValueError(f"{trace.stem} contains no complete eviction examples")

    features = []
    gains = []
    for decision in decisions:
        baseline, *alternatives = decision["candidates"]
        victim = np.array([log1p(baseline[field] or 0) for field in FIELDS])
        rows = []
        for alternative in alternatives:
            other = np.array([log1p(alternative[field] or 0) for field in FIELDS])
            rows.append(np.concatenate((victim, other - victim)))
        features.append(rows)
        gains.append([row["misses_saved"] for row in alternatives])
    return np.asarray(features), np.asarray(gains), decisions


def matrix(features, columns):
    return features[:, :, columns].reshape(-1, len(columns))


def outcome(chosen_gains):
    return {
        "overrides": len(chosen_gains),
        "helpful": int(np.count_nonzero(chosen_gains > 0)),
        "tied": int(np.count_nonzero(chosen_gains == 0)),
        "harmful": int(np.count_nonzero(chosen_gains < 0)),
        "net_local_misses_saved": int(chosen_gains.sum()),
    }


def assess(scores, gains, threshold):
    best = scores.argmax(axis=1)
    decision = np.arange(len(best))
    accepted = scores[decision, best] > threshold
    return outcome(gains[decision[accepted], best[accepted]])


def calibrate(scores, gains):
    trials = [{"threshold": threshold, **assess(scores, gains, threshold)}
              for threshold in THRESHOLDS]
    return max(trials, key=lambda row: (row["net_local_misses_saved"], -row["overrides"]))


def export_model(estimator, columns, threshold):
    record = {"feature_order": [FEATURES[i] for i in columns], "threshold": threshold}
    if isinstance(estimator, GradientBoostingClassifier):
        probability = estimator.init_.class_prior_[1]
        trees = []
        for fitted in estimator.estimators_[:, 0]:
            tree = fitted.tree_
            trees.append({"feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                          "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                          "value": tree.value[:, 0, 0].tolist()})
        record.update(kind="gradient_boosted_trees", initial_logit=log(probability / (1 - probability)),
                      learning_rate=estimator.learning_rate, trees=trees)
    else:
        scaler = estimator.named_steps["standardscaler"]
        linear = estimator.named_steps["logisticregression"]
        weights = linear.coef_[0] / scaler.scale_
        record.update(kind="logistic_regression", weights=weights.tolist(),
                      intercept=float(linear.intercept_[0] - weights @ scaler.mean_))
    return record


def exported_logits(record, inputs):
    if record["kind"] == "logistic_regression":
        return inputs @ np.array(record["weights"]) + record["intercept"]

    inputs = inputs.astype(np.float32)
    scores = np.full(len(inputs), record["initial_logit"])
    for tree in record["trees"]:
        for i, row in enumerate(inputs):
            node = 0
            while tree["left"][node] != -1:
                value = float(row[tree["feature"][node]])
                side = "left" if value <= tree["threshold"][node] else "right"
                node = tree[side][node]
            scores[i] += record["learning_rate"] * tree["value"][node]
    return scores


def original_reuse_choices(decisions):
    path = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"
    bias, age, hits = [float(line.split()[1]) for line in path.read_text().splitlines()[1:]]
    selected = []
    for decision in decisions:
        def score(row):
            return bias + age * log1p(row["age_reads"]) + hits * log1p(row["resident_hits"])
        candidate = min(decision["candidates"], key=score)
        if candidate["rank"] != 0:
            selected.append(candidate["misses_saved"])
    return outcome(np.asarray(selected))


def fit_and_evaluate(windows, capacity, horizon):
    train_x, train_gains, train_rows = read_examples(windows[0], capacity, horizon, 4000)
    calibration_x, calibration_gains, calibration_rows = read_examples(windows[1], capacity, horizon, 2000)
    train_y = (train_gains > 0).ravel()
    if len(np.unique(train_y)) != 2:
        raise ValueError("training needs helpful and non-helpful alternatives")

    candidates = (
        ("basic_pairwise_logistic", BASIC_COLUMNS,
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("history_pairwise_logistic", ALL_COLUMNS,
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("history_shallow_boosted_trees", ALL_COLUMNS,
         GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=2,
                                    min_samples_leaf=20, random_state=0)),
    )

    results, exports = {}, {}
    for name, columns, estimator in candidates:
        estimator.fit(matrix(train_x, columns), train_y)
        calibration_scores = estimator.predict_proba(matrix(calibration_x, columns))[:, 1]
        calibration = calibrate(calibration_scores.reshape(calibration_gains.shape), calibration_gains)
        results[name] = {"calibration": calibration}
        exports[name] = export_model(estimator, columns, calibration["threshold"])

    selected = max(results, key=lambda name: (results[name]["calibration"]["net_local_misses_saved"],
                                              -results[name]["calibration"]["overrides"]))
    eval_x, eval_gains, eval_rows = read_examples(windows[2], capacity, horizon, 2000)
    eval_y = (eval_gains > 0).ravel()
    serialized = json.loads(json.dumps(exports))
    if len(np.unique(eval_y)) != 2:
        raise ValueError("evaluation needs helpful and non-helpful alternatives")
    for name, columns, estimator in candidates:
        inputs = matrix(eval_x, columns)
        probabilities = estimator.predict_proba(inputs)[:, 1]
        np.testing.assert_allclose(exported_logits(serialized[name], inputs),
                                   estimator.decision_function(inputs), rtol=1e-10, atol=1e-10)
        results[name]["offline_evaluation"] = {
            "roc_auc": float(roc_auc_score(eval_y, probabilities)),
            "average_precision": float(average_precision_score(eval_y, probabilities)),
            **assess(probabilities.reshape(eval_gains.shape), eval_gains, exports[name]["threshold"]),
        }

    return ({
        "selected_on_calibration": selected,
        "models": results,
        "training_pairs": summarize(train_rows),
        "calibration_pairs": summarize(calibration_rows),
        "evaluation_pairs": summarize(eval_rows),
        "evaluation_positive_rate": float(eval_y.mean()),
        "evaluation_original_reuse_always_score": original_reuse_choices(eval_rows),
    }, {"format": "cachetriage_pairwise_v1", "selected_model": selected,
        "preprocessing": "log1p on counts; missing previous_gap_reads is zero; delta is alternative minus victim",
        "decision": "choose highest probability among alternatives and override only if strictly above threshold",
        "models": exports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="uncompressed Twitter trace or prefix")
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if args.capacity < 2 or args.horizon < 1 or args.rows < 1:
        parser.error("capacity must be at least two, and horizon and rows must be positive")
    if args.source.resolve() == args.model_out.resolve():
        parser.error("source and model output must be different files")

    try:
        with tempfile.TemporaryDirectory(prefix="cachetriage-pairwise-") as temp:
            work = Path(temp)
            trace = work / "all.txt"
            source = prepare_trace(args.source, trace, 1, args.rows)
            keys = list(request_keys(trace))
            cuts = (0, len(keys) * 3 // 5, len(keys) * 4 // 5, len(keys))
            windows = []
            splits = {}
            for name, start, stop in zip(("train", "calibration", "evaluation"), cuts, cuts[1:]):
                window = work / f"{name}.txt"
                window.write_text("".join(f"{key}\n" for key in keys[start:stop]), encoding="ascii")
                windows.append(window)
                splits[name] = {"first_read": start + 1, "last_read": stop}
            report, model = fit_and_evaluate(windows, args.capacity, args.horizon)
            report.update({"source": source, "splits": splits, "capacity_items": args.capacity,
                           "horizon_reads": args.horizon, "scikit_learn_version": sklearn.__version__,
                           "selection": "maximize net local misses saved on calibration, then prefer fewer overrides"})
            model.update({"capacity_items": args.capacity, "horizon_reads": args.horizon})
            args.model_out.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
