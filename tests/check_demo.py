"""Check that the SQLite path makes one fetch per replay miss."""

import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"


class DemoTest(unittest.TestCase):
    def compare(self, trace, capacity):
        demo = json.loads(subprocess.check_output([
            ROOT / "build" / "cachetriage_demo", str(capacity), trace, MODEL, "1", "2",
        ]))
        replay = json.loads(subprocess.check_output([
            ROOT / "build" / "cachetriage", str(capacity), trace, "--benchmark", MODEL, "1", "1",
        ]))
        self.assertEqual(demo["requests"], replay["requests"])
        self.assertEqual(demo["profile_bytes"], 256)
        self.assertEqual(len(demo["results"]), 4)
        for measured, expected in zip(demo["results"], replay["results"]):
            self.assertEqual(measured["policy"], expected["policy"])
            self.assertEqual(measured["backend_calls"], expected["misses"])
            self.assertEqual(measured["model_calls"], expected.get("model_calls", 0))
            if demo["requests"]:
                self.assertGreaterEqual(measured["p95_latency_us"], measured["p50_latency_us"])
            else:
                self.assertIsNone(measured["p95_latency_us"])

    def test_all_policies_match_replay(self):
        self.compare(ROOT / "tests" / "always_score.txt", 3)

    def test_unsigned_keys_and_zero_capacity(self):
        with tempfile.TemporaryDirectory() as work:
            trace = Path(work) / "reads.txt"
            trace.write_text("0\n9223372036854775808\n18446744073709551615\n0\n")
            self.compare(trace, 0)
            self.compare(trace, 3)

    def test_delay_changes_timing_without_changing_query_counts(self):
        trace = ROOT / "tests" / "always_score.txt"
        report = json.loads(subprocess.check_output([
            ROOT / "build" / "cachetriage_demo", "3", trace, MODEL, "1", "2",
            "--delay-sweep", "0", "10",
        ]))
        points = report["delay_sweep"]
        self.assertEqual([point["added_fetch_delay_us"] for point in points], [0, 10])
        for point in points:
            self.assertEqual([row["backend_calls"] for row in point["results"]], [6, 5])
            self.assertEqual([row["model_calls"] for row in point["results"]], [0, 1])
            for row in point["results"]:
                self.assertGreaterEqual(row["median_injected_us_per_fetch"],
                                        point["added_fetch_delay_us"])
            for metric in ("mean", "p95"):
                deltas = point[f"paired_{metric}_deltas_us"]
                self.assertEqual(len(deltas), 2)
                self.assertAlmostEqual(point[f"median_paired_{metric}_delta_us"], median(deltas))

    def test_hit_gate_matches_replay(self):
        trace = ROOT / "tests" / "always_score.txt"
        for threshold, calls in (("1", 1), ("2", 0)):
            demo = json.loads(subprocess.check_output([
                ROOT / "build" / "cachetriage_demo", "3", trace, MODEL, threshold, "2", "--hit-gate",
            ]))
            replay = json.loads(subprocess.check_output([
                ROOT / "build" / "cachetriage", "3", trace, "--benchmark-hits", MODEL, threshold, "2",
            ]))
            self.assertEqual(demo["gate_min_victim_hits"], int(threshold))
            self.assertEqual(replay["gate_min_victim_hits"], int(threshold))
            self.assertEqual([row["backend_calls"] for row in demo["results"]], [6, 6, 5, 6])
            self.assertEqual(demo["results"][-1]["model_calls"], calls)
            for measured, expected in zip(demo["results"], replay["results"]):
                self.assertEqual(measured["backend_calls"], expected["misses"])
                self.assertEqual(measured["model_calls"], expected.get("model_calls", 0))

    def test_empty_trace(self):
        with tempfile.TemporaryDirectory() as work:
            trace = Path(work) / "reads.txt"
            trace.touch()
            self.compare(trace, 0)


if __name__ == "__main__":
    unittest.main()
