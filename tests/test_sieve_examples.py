from pathlib import Path
import tempfile
import unittest

from sieve_examples import Example, examples
from twitter_baseline import ROOT


BINARY = ROOT / "build" / "cachetriage"


class SieveExamplesTest(unittest.TestCase):
    def test_features_and_labels_at_eviction(self):
        rows = list(examples(BINARY, ROOT / "traces" / "example.txt", 3, 2))
        self.assertEqual(rows, [
            Example(6, 2, 0, 4, 0, 0),
            Example(6, 3, 1, 2, 1, 1),
            Example(6, 1, 2, 1, 1, 0),
        ])

    def test_readmission_resets_hits_and_incomplete_labels_are_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "reads.txt"
            trace.write_text("1\n2\n1\n3\n4\n1\n4\n5\n2\n4\n", encoding="ascii")
            rows = list(examples(BINARY, trace, 2, 2))
        self.assertEqual(rows[-2:], [Example(8, 1, 0, 2, 0, 0), Example(8, 4, 1, 1, 1, 1)])
