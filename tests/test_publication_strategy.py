# -*- coding: utf-8 -*-
"""
Unit tests for the Master Publication Strategy & Advisor Briefing Guide.
Verifies:
1. Document exists at docs/publication_strategy_and_advisor_briefing.md.
2. Document length is comprehensive (>10,000 characters).
3. Covers all target venues: EMNLP, COLING, ACM TALLIP, ACL Rolling Review, LREC-COLING.
4. Covers core concepts: Kazakh-FEVER 3K, Hard NEI, Dual-Submission, 55%, Wangna.
5. Contains LaTeX outline (\\section{, \\begin{table}, \\caption{).
6. Zero decorative emojis, zero placeholders (TODO, TBD).
7. Brain artifact mirror exists and matches content.
"""

import os
import re
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(PROJECT_ROOT, "docs", "publication_strategy_and_advisor_briefing.md")
BRAIN_MIRROR_PATH = os.path.join(
    r"C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c",
    "publication_strategy_and_advisor_briefing.md"
)


class TestPublicationStrategy(unittest.TestCase):

    def setUp(self):
        self.doc_path = DOC_PATH
        self.brain_path = BRAIN_MIRROR_PATH
        if os.path.isfile(self.doc_path):
            with open(self.doc_path, "r", encoding="utf-8") as f:
                self.content = f.read()
        else:
            self.content = None

    def test_01_document_exists(self):
        self.assertTrue(
            os.path.isfile(self.doc_path),
            f"Strategy document does not exist at {self.doc_path}"
        )

    def test_02_brain_artifact_mirror_exists_and_matches(self):
        self.assertTrue(
            os.path.isfile(self.brain_path),
            f"Brain artifact mirror does not exist at {self.brain_path}"
        )
        with open(self.brain_path, "r", encoding="utf-8") as f:
            brain_content = f.read()
        self.assertEqual(
            self.content,
            brain_content,
            "Brain artifact mirror content does not match docs/publication_strategy_and_advisor_briefing.md"
        )

    def test_03_document_length_is_comprehensive(self):
        self.assertIsNotNone(self.content, "Document content is None")
        self.assertGreater(
            len(self.content),
            10000,
            f"Document length ({len(self.content)} chars) is less than the required 10,000 characters"
        )

    def test_04_target_venues_covered(self):
        self.assertIsNotNone(self.content, "Document content is None")
        venues = ["EMNLP", "COLING", "ACM TALLIP", "ACL Rolling Review", "LREC-COLING"]
        for venue in venues:
            self.assertIn(
                venue,
                self.content,
                f"Document must cover target venue '{venue}'"
            )

    def test_05_core_concepts_covered(self):
        self.assertIsNotNone(self.content, "Document content is None")
        concepts = [
            "Kazakh-FEVER 3K",
            "Hard NEI",
            "Dual-Submission",
            "55%",
            "Wangna",
            "Daulet",
            "Professor Guo",
            "LLM Memorization Trap",
            "2D Trust Matrix",
            "Hybrid Retrieval"
        ]
        for concept in concepts:
            self.assertIn(
                concept,
                self.content,
                f"Document must contain core concept or entity '{concept}'"
            )

    def test_06_latex_outline_present(self):
        self.assertIsNotNone(self.content, "Document content is None")
        latex_elements = [
            r"\section{",
            r"\begin{table}",
            r"\caption{",
            r"\begin{equation}"
        ]
        for elem in latex_elements:
            self.assertIn(
                elem,
                self.content,
                f"Document must include LaTeX outline element '{elem}'"
            )

    def test_07_zero_placeholders(self):
        self.assertIsNotNone(self.content, "Document content is None")
        placeholders = ["TODO", "TBD", "FIXME", "XXX"]
        for ph in placeholders:
            pattern = rf"\b{ph}\b"
            matches = re.findall(pattern, self.content)
            self.assertEqual(
                len(matches),
                0,
                f"Document contains placeholder '{ph}': {matches}"
            )

    def test_08_zero_decorative_emojis(self):
        self.assertIsNotNone(self.content, "Document content is None")
        for ch in self.content:
            code = ord(ch)
            is_emoji = (
                (0x1F600 <= code <= 0x1F64F) or
                (0x1F300 <= code <= 0x1F5FF) or
                (0x1F680 <= code <= 0x1F6FF) or
                (0x1F700 <= code <= 0x1F77F) or
                (0x1F900 <= code <= 0x1F9FF) or
                (0x1FA00 <= code <= 0x1FA6F) or
                (0x1FA70 <= code <= 0x1FAFF) or
                (0x2600 <= code <= 0x26FF) or
                (0x2700 <= code <= 0x27BF and code not in (0x2713, 0x2714))
            )
            self.assertFalse(
                is_emoji,
                f"Document contains decorative emoji: '{ch}' (U+{code:04X})"
            )

    def test_09_arr_calendar_and_milestones(self):
        self.assertIsNotNone(self.content, "Document content is None")
        # Check bi-monthly cycles and milestones
        for cycle in ["April", "June", "August", "October", "December", "February"]:
            self.assertIn(
                cycle,
                self.content,
                f"Document must mention ARR cycle month '{cycle}'"
            )
        for milestone in ["Month 1", "Month 2", "Month 3", "Month 4"]:
            self.assertIn(
                milestone,
                self.content,
                f"Document must mention milestone '{milestone}'"
            )


if __name__ == "__main__":
    unittest.main()
