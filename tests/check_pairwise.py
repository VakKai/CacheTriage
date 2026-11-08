import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "cachetriage_pairwise"
REUSE = ROOT / "benchmarks" / "twitter-cluster052-reuse-v1.txt"


class PairwiseTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.model = Path(work.name) / "pairwise.txt"
        # Logistic always prefers the first alternative. The tree abstains.
        self.model.write_text("cachetriage_pairwise_replay_v1\n"
                              "0.5 1 0 0 0 0\n"
                              "0.5 -1 0.1 1\n"
                              "1\n-2 -1 -1 -2 0\n", encoding="ascii")

    def run_case(self, mode, limit, rounds_or_target=2):
        return json.loads(subprocess.check_output([
            BINARY, ROOT / "tests" / "oracle.txt", self.model, REUSE, mode, str(limit), str(rounds_or_target),
        ]))

    def test_override_changes_future_misses(self):
        report = self.run_case("items", 3)
        rows = {row["policy"]: row for row in report["results"]}
        self.assertEqual(rows["sieve"]["misses"], 6)
        self.assertEqual(rows["pairwise_logistic"]["misses"], 5)
        self.assertEqual(rows["pairwise_logistic"]["victim_swaps"], 2)
        self.assertEqual(rows["pairwise_boosted"]["misses"], 6)
        self.assertEqual(rows["pairwise_boosted"]["victim_swaps"], 0)

    def test_budget_charges_the_model_and_cache(self):
        budget = 40 * 1024
        report = self.run_case("bytes", budget)
        rows = {row["policy"]: row for row in report["performance"]}
        for row in rows.values():
            self.assertLessEqual(row["peak_accounted_bytes"], budget)
        self.assertGreater(rows["pairwise_boosted"]["model_bytes"], 0)
        self.assertGreater(rows["sieve"]["capacity_items"], rows["pairwise_boosted"]["capacity_items"])

    def test_compact_victims_match_full_history(self):
        self.model.write_text(self.model.read_text().replace("0.5 1 0 0 0 0", "0.3 1.3 -1.1 1.5 -0.2 0.67"))
        trace = self.model.with_name("parity.txt")
        trace.write_text("".join(f"{i % 211 if i % 3 else i % 47}\n" for i in range(2000)), encoding="ascii")
        report = json.loads(subprocess.check_output([ROOT / "build" / "check_compact_victims", self.model, trace]))
        self.assertEqual(report["requests"], 2000)
        self.assertGreater(report["matching_evictions"], 0)

    def test_compact_scan_finds_first_matching_capacity(self):
        expected = next(capacity for capacity in range(1, 5)
                        if self.run_case("compact-items", capacity)["results"][1]["misses"] <= 5)
        self.assertEqual(self.run_case("compact-scan", 5, 5)["capacity_items"], expected)
        self.assertIsNone(self.run_case("compact-scan", 4, 0))

    def test_sqlite_budget_matches_replay(self):
        trace = self.model.with_name("profiles.txt")
        trace.write_text("".join(f"{i % 20}\n" for i in range(160)), encoding="ascii")
        budget = 4096
        demo = json.loads(subprocess.check_output([
            ROOT / "build" / "cachetriage_demo", "--pairwise-budget", str(budget), trace, self.model, "2",
        ]))
        self.assertEqual(demo["profile_bytes"], 256)
        self.assertGreater(demo["memory"][0]["capacity_items"], demo["memory"][1]["capacity_items"])
        for i, (memory, trial) in enumerate(zip(demo["memory"], demo["results"])):
            self.assertLessEqual(memory["peak_accounted_bytes"], budget)
            replay = json.loads(subprocess.check_output([
                BINARY, trace, self.model, REUSE, "compact-items", str(memory["capacity_items"]), "1",
            ]))["results"][i]
            self.assertEqual(trial["policy"], replay["policy"])
            self.assertEqual(trial["backend_calls"], replay["misses"])
            self.assertEqual(trial["model_calls"], replay.get("model_calls", 0))
            self.assertGreaterEqual(trial["median_cpu_seconds"], 0)

    def test_invalid_tree_child_is_rejected(self):
        self.model.write_text(self.model.read_text().replace("-2 -1 -1 -2 0", "0 1 2 0 0"))
        result = subprocess.run([
            BINARY, ROOT / "tests" / "oracle.txt", self.model, REUSE, "items", "3", "1",
        ], capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"invalid tree node", result.stderr)


if __name__ == "__main__":
    unittest.main()
