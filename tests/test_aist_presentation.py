"""
Unit tests verifying the AIST 2026 conference presentation slide deck integrity,
PDF compilation artifacts, page count, and desktop synchronization.
"""

import hashlib
import os
import re
import unittest


class TestAISTPresentation(unittest.TestCase):
    """Validation test suite for AIST 2026 oral presentation deck."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.tex_path = os.path.join(cls.repo_root, "presentation", "aist2026_presentation.tex")
        cls.pdf_path = os.path.join(cls.repo_root, "presentation", "aist2026_presentation.pdf")
        cls.log_path = os.path.join(cls.repo_root, "presentation", "aist2026_presentation.log")
        cls.desktop_pdf = os.path.join(os.path.expanduser("~"), "Desktop", "AIST2026_Presentation_Daulet_Anekesh.pdf")

    def test_source_tex_exists_and_valid(self):
        """Verify LaTeX source exists, has 17 frames, and has zero emojis."""
        self.assertTrue(os.path.isfile(self.tex_path), f"Source tex not found at {self.tex_path}")
        with open(self.tex_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check total frames
        frames = re.findall(r"\\begin\{frame\}", content)
        self.assertEqual(len(frames), 17, f"Expected 17 frames, but found {len(frames)}")

        # Check aspect ratio 16:9
        self.assertIn("aspectratio=169", content, "Document must specify aspectratio=169")

        # Invariance: strictly zero decorative emojis
        emoji_pattern = re.compile(
            "[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]",
            flags=re.UNICODE,
        )
        emojis_found = emoji_pattern.findall(content)
        self.assertEqual(len(emojis_found), 0, f"Found decorative emojis in presentation: {emojis_found}")

    def test_pdf_exists_and_has_17_pages(self):
        """Verify compiled PDF exists and has exactly 17 pages."""
        self.assertTrue(os.path.isfile(self.pdf_path), f"Compiled PDF not found at {self.pdf_path}")
        self.assertGreater(os.path.getsize(self.pdf_path), 50000, "PDF file is unusually small")

        import pypdf
        reader = pypdf.PdfReader(self.pdf_path)
        self.assertEqual(len(reader.pages), 17, f"Expected 17 pages, but found {len(reader.pages)}")

        if os.path.isfile(self.log_path):
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_text = f.read()
            self.assertIn("Output written on aist2026_presentation.pdf (17 pages)", log_text)

    def test_compilation_log_clean(self):
        """Verify compilation log has zero fatal errors and zero missing characters if present."""
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                log = f.read()
            self.assertNotIn("Fatal error", log)
            self.assertNotIn("Missing character:", log)

    def test_desktop_synchronization(self):
        """Verify synchronized Desktop PDF exists and matches master compiled PDF."""
        self.assertTrue(os.path.isfile(self.desktop_pdf), f"Desktop PDF not found at {self.desktop_pdf}")

        def get_md5(path):
            hasher = hashlib.md5()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()

        master_hash = get_md5(self.pdf_path)
        desktop_hash = get_md5(self.desktop_pdf)
        self.assertEqual(
            master_hash,
            desktop_hash,
            "Desktop PDF hash does not match presentation/aist2026_presentation.pdf",
        )


if __name__ == "__main__":
    unittest.main()
