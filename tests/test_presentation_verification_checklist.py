"""Unit tests for the master thesis presentation verification checklist.

Validates that docs/thesis_presentation_verification_checklist.md and its brain
artifact mirror exist, meet size criteria, contain all 6 thematic pillars,
cover all 23 slides with image links, and contain zero placeholder markers.
"""

import os
import unittest

from scripts.verify_presentation_against_checklist import (
    verify_presentation_file,
    check_slide_count_and_dimensions,
    check_typography,
    check_sequential_toc,
    check_slide_content_anchors,
    check_banned_terms,
    check_zero_emojis,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKLIST_PATH = os.path.join(
    PROJECT_ROOT,
    "docs",
    "thesis_presentation_verification_checklist.md"
)
BRAIN_ARTIFACT_PATH = r"C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c\thesis_presentation_verification_checklist.md"
REPO_PPTX_PATH = os.path.join(
    PROJECT_ROOT,
    "Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx"
)
DESKTOP_PPTX_PATH = r"C:\Users\Roza\Desktop\AnekeshD_Progress.pptx"


class TestPresentationVerificationChecklist(unittest.TestCase):
    """Test suite for the master thesis presentation verification checklist and checker script."""

    def test_checklist_document_exists_and_complete(self):
        """Verify the primary checklist document exists, is comprehensive, and has no placeholders."""
        self.assertTrue(os.path.isfile(CHECKLIST_PATH), f"Checklist missing at {CHECKLIST_PATH}")
        with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Check size and structure
        self.assertGreater(len(content), 10000, "Checklist should be comprehensive (>10KB)")
        self.assertNotIn("TODO", content)
        self.assertNotIn("TBD", content)
        self.assertNotIn("[ ] [Placeholder", content)

        # Check 6 thematic pillars
        self.assertIn("Pillar 1: Research Storyline & Topic 2 Restoration", content)
        self.assertIn("Pillar 2: Architectural Rigor & Terminology Unification", content)
        self.assertIn("Pillar 3: Experimental Metrics & Statistical Rigor", content)
        self.assertIn("Pillar 4: Topic 3 Kazakh-FEVER & 2D Trust Matrix", content)
        self.assertIn("Pillar 5: Slide Hygiene & Typography Standardization", content)
        self.assertIn("Pillar 6: Strategic Consultation Roadmap & Speaker Notes", content)

        # Check all 23 slides are covered in Part 2 with image references
        for s_idx in range(1, 24):
            slide_tag = f"Slide {s_idx:02d}:"
            self.assertIn(slide_tag, content, f"Slide {s_idx:02d} missing from audit guide")
            self.assertIn(f"slide_{s_idx:02d}.png", content, f"Preview image link for Slide {s_idx:02d} missing")

    def test_brain_artifact_mirror_exists(self):
        """Verify the mirror artifact exists in the brain directory and is populated."""
        self.assertTrue(os.path.isfile(BRAIN_ARTIFACT_PATH), f"Mirror artifact missing at {BRAIN_ARTIFACT_PATH}")
        with open(BRAIN_ARTIFACT_PATH, "r", encoding="utf-8") as f:
            mirror_content = f.read()
        self.assertGreater(len(mirror_content), 10000, "Brain artifact should be comprehensive (>10KB)")
        self.assertNotIn("TODO", mirror_content)
        self.assertNotIn("TBD", mirror_content)

    def test_verify_presentation_structure_and_repo_file(self):
        """Verify that verify_presentation_file runs on repo PPTX and returns valid structure with 100% pass."""
        self.assertTrue(os.path.isfile(REPO_PPTX_PATH), f"Repo presentation missing at {REPO_PPTX_PATH}")
        res = verify_presentation_file(REPO_PPTX_PATH)
        self.assertIsInstance(res, dict)
        self.assertIn("path", res)
        self.assertIn("all_passed", res)
        self.assertIn("results", res)
        self.assertTrue(res["all_passed"], f"Verification failed on repo presentation: {res['results']}")

        expected_categories = {
            "slide_count_and_dimensions",
            "typography",
            "sequential_toc",
            "slide_content_anchors",
            "banned_terms",
            "zero_emojis",
        }
        result_names = {r["name"] for r in res["results"]}
        self.assertTrue(
            expected_categories.issubset(result_names),
            f"Missing check categories: {expected_categories - result_names}"
        )

        for check in res["results"]:
            self.assertTrue(
                check["passed"],
                f"Check '{check['name']}' failed on repo presentation: {check['detail']}"
            )

    def test_verify_presentation_desktop_file_if_present(self):
        """Verify that verify_presentation_file passes on Desktop AnekeshD_Progress.pptx if accessible."""
        if not os.path.isfile(DESKTOP_PPTX_PATH):
            self.skipTest(f"Desktop presentation not found at {DESKTOP_PPTX_PATH}")

        res = verify_presentation_file(DESKTOP_PPTX_PATH)
        self.assertTrue(
            res["all_passed"],
            f"Verification failed on Desktop presentation: {res['results']}"
        )
        for check in res["results"]:
            self.assertTrue(
                check["passed"],
                f"Check '{check['name']}' failed on Desktop presentation: {check['detail']}"
            )

    def test_verify_presentation_nonexistent_file(self):
        """Verify that verify_presentation_file handles non-existent file cleanly."""
        res = verify_presentation_file("nonexistent_presentation_file.pptx")
        self.assertFalse(res["all_passed"])
        self.assertTrue(any(not r["passed"] for r in res["results"]))


if __name__ == "__main__":
    unittest.main()

