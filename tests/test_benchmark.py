import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "cachetriage"
MODEL = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"


def replay(capacity, trace, *args):
    return json.loads(subprocess.check_output([BINARY, str(capacity), trace, *args]))


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_matches_individual_replays(self):
        trace = ROOT / "tests" / "always_score.txt"
        measured = replay(3, trace, "--benchmark", MODEL, "1", "4")
        always = replay(3, trace, "--always-score", MODEL)
        gated = replay(3, trace, "--triage", MODEL, "1")
        self.assertEqual(measured["results"], always["results"] + gated["results"][-1:])
        self.assertEqual(measured["benchmark_rounds"], 4)
        for result, perf in zip(measured["results"], measured["performance"]):
            self.assertEqual(result["policy"], perf["policy"])
            self.assertEqual(len(perf["cpu_seconds_samples"]), 4)
            self.assertEqual(len(perf["wall_seconds_samples"]), 4)
            self.assertAlmostEqual(perf["median_cpu_seconds"], median(perf["cpu_seconds_samples"]))
            self.assertAlmostEqual(perf["median_wall_seconds"], median(perf["wall_seconds_samples"]))
            self.assertGreater(perf["peak_requested_heap_bytes"], 0)
            self.assertEqual(perf["peak_accounted_bytes"],
                             perf["replay_inline_bytes"] + perf["peak_requested_heap_bytes"] +
                             perf["shared_model_table_bytes"])

    def test_empty_trace_at_zero_capacity(self):
        with tempfile.TemporaryDirectory() as work:
            trace = Path(work) / "empty.txt"
            trace.touch()
            measured = replay(0, trace, "--benchmark", MODEL, "0", "1")
        self.assertEqual(measured["requests"], 0)
        for result, perf in zip(measured["results"], measured["performance"]):
            self.assertIsNone(result["miss_rate"])
            self.assertEqual(result["evictions"], 0)
            self.assertGreaterEqual(perf["peak_accounted_bytes"], perf["replay_inline_bytes"])
