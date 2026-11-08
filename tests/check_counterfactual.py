import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "cachetriage_counterfactual"


def analyze(trace, capacity, horizon, limit=100):
    return json.loads(subprocess.check_output([
        BINARY, str(capacity), trace, str(horizon), str(limit),
    ]))


class CounterfactualTest(unittest.TestCase):
    def test_one_eviction_changes_next_two_reads(self):
        result = analyze(ROOT / "tests" / "oracle.txt", 3, 2)
        self.assertEqual(result["eligible_evictions"], 1)
        decision, = result["sampled_decisions"]
        self.assertEqual(decision["request"], 4)
        self.assertEqual(decision["incoming_key"], 4)
        candidates = decision["candidates"]
        self.assertEqual([c["key"] for c in candidates], [1, 2, 3])
        self.assertEqual([c["future_misses"] for c in candidates], [2, 1, 0])
        self.assertEqual([c["misses_saved"] for c in candidates], [0, 1, 2])

    def test_snapshot_preserves_hand_and_candidate_history(self):
        result = analyze(ROOT / "traces" / "example.txt", 3, 2)
        decision, = result["sampled_decisions"]
        self.assertEqual(decision["request"], 6)
        self.assertEqual([c["key"] for c in decision["candidates"]], [2, 3, 1])
        self.assertEqual([c["age_reads"] for c in decision["candidates"]], [4, 2, 1])
        self.assertEqual([c["resident_hits"] for c in decision["candidates"]], [0, 1, 1])
        self.assertEqual([c["residency_reads"] for c in decision["candidates"]], [4, 3, 5])
        self.assertEqual([c["previous_gap_reads"] for c in decision["candidates"]], [None, 1, 4])

    def test_ties_and_harm_are_kept(self):
        with tempfile.TemporaryDirectory() as work:
            trace = Path(work) / "reads.txt"
            for last, expected in ((2, -1), (4, 0)):
                trace.write_text(f"1\n2\n3\n{last}\n")
                decision, = analyze(trace, 2, 1)["sampled_decisions"]
                self.assertEqual(decision["candidates"][1]["misses_saved"], expected)

    def test_samples_cover_the_eligible_sequence(self):
        with tempfile.TemporaryDirectory() as work:
            trace = Path(work) / "reads.txt"
            trace.write_text("".join(f"{key}\n" for key in range(10)))
            result = analyze(trace, 2, 1, 3)
            self.assertEqual(result["eligible_evictions"], 7)
            self.assertEqual([row["request"] for row in result["sampled_decisions"]], [4, 7, 9])
            self.assertEqual(analyze(trace, 2, 10)["sampled_decisions"], [])


if __name__ == "__main__":
    unittest.main()
