"""Unit tests for RegressionGuard sandbox verification."""

import unittest
from src.auto_fst.guard import RegressionGuard

class TestRegressionGuard(unittest.TestCase):
    def setUp(self):
        self.guard = RegressionGuard()

    def test_valid_candidate_passes(self):
        passed, reason = self.guard.validate_candidate("вайб")
        self.assertTrue(passed)
        self.assertEqual(reason, "Validation passed: 100% agreement on golden baseline.")

    def test_harmful_prefix_candidate_rejected(self):
        # A candidate that matches a common native stem or function word must be rejected
        passed, reason = self.guard.validate_candidate("бол")
        self.assertFalse(passed)
        self.assertIn("Reserved native root", reason)

    def test_short_length_candidate_rejected(self):
        passed, reason = self.guard.validate_candidate("а")
        self.assertFalse(passed)
        self.assertIn("minimum length", reason.lower())

    def test_golden_sentences_coverage(self):
        self.assertGreaterEqual(len(self.guard.golden_parses), 5)

if __name__ == '__main__':
    unittest.main()
